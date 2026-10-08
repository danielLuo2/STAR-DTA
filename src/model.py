"""STAR-DTA model: multi-view drug encoder, multi-scale residue encoder and drug/protein cross-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool


def gin_mlp(in_ch, out_ch):
    return Sequential(Linear(in_ch, out_ch), ReLU(), Linear(out_ch, out_ch))


def length_mask(lengths, max_len, device):
    return torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)


def masked_mean(x, mask, dim=1):
    mask_f = mask.unsqueeze(-1).float()
    return (x * mask_f).sum(dim=dim) / mask_f.sum(dim=dim).clamp(min=1.0)


def masked_max(x, mask, dim=1):
    return x.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=dim).values


class DrugGINEncoder(nn.Module):
    """Five-layer GIN over the molecular graph (GraphDTA style) with sum + max readout."""

    def __init__(self, in_dim=78, dim=32, output_dim=128, dropout=0.2, dual_pool=True):
        super().__init__()
        self.dropout = dropout
        self.dual_pool = dual_pool

        self.conv1 = GINConv(gin_mlp(in_dim, dim))
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = GINConv(gin_mlp(dim, dim))
        self.bn2 = nn.BatchNorm1d(dim)
        self.conv3 = GINConv(gin_mlp(dim, dim))
        self.bn3 = nn.BatchNorm1d(dim)
        self.conv4 = GINConv(gin_mlp(dim, dim))
        self.bn4 = nn.BatchNorm1d(dim)
        self.conv5 = GINConv(gin_mlp(dim, dim))
        self.bn5 = nn.BatchNorm1d(dim)
        fc_in = dim * 2 if dual_pool else dim
        self.fc = Linear(fc_in, output_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = F.relu(self.bn5(self.conv5(x, edge_index)))
        if self.dual_pool:
            x = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        else:
            x = global_add_pool(x, batch)
        x = F.relu(self.fc(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SmilesCNNEncoder(nn.Module):
    """Character-level SMILES CNN; returns one feature vector per SMILES token."""

    def __init__(self, vocab_size=80, embed_dim=64, hidden_dim=128, max_len=200, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.dropout = dropout

    def forward(self, smiles_ids):
        x = self.embed(smiles_ids)
        x = x.transpose(1, 2)
        x = F.dropout(F.relu(self.conv1(x)), p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.conv2(x)), p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.conv3(x)), p=self.dropout, training=self.training)
        return x.transpose(1, 2)


class MorganEncoder(nn.Module):
    """MLP over the 1024-bit Morgan fingerprint (radius 2)."""

    def __init__(self, n_bits=1024, output_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            Linear(n_bits, 512),
            ReLU(),
            nn.Dropout(dropout),
            Linear(512, output_dim),
        )

    def forward(self, fp):
        return F.relu(self.net(fp))


class MultiScaleResidueEncoder(nn.Module):
    """Parallel residue convolutions with kernel sizes 5 / 7 / 9, concatenated and projected."""

    def __init__(self, embed_dim=128, hidden_dim=128, kernels=(5, 7, 9)):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, hidden_dim, k, padding=k // 2) for k in kernels]
        )
        self.proj = nn.Linear(hidden_dim * len(kernels), hidden_dim)

    def forward(self, emb):
        x = emb.transpose(1, 2)
        parts = [F.relu(conv(x)) for conv in self.convs]
        z = torch.cat(parts, dim=1).transpose(1, 2)
        return F.relu(self.proj(z))


class STARDTA(nn.Module):
    """
    STAR-DTA affinity model.

    Drug: GIN graph branch + SMILES-CNN branch + Morgan-fingerprint branch, fused into one vector.
    Protein: GraphDTA-style global convolution + multi-scale (5/7/9) residue convolutions.
    Interaction: drug-query-to-residue attention and protein-query-to-atom (SMILES token) attention.
    The affinity head reads [z_d, z_p, z_d * z_p, c_dr, c_ps].

    The drug-gated per-residue output (residue_head) and the affinity-prior branch are optional
    extras; they are not used by any result reported in the paper (see the README).
    """

    def __init__(
        self,
        drug_in_dim=78,
        hidden_dim=128,
        max_protein_len=1000,
        smiles_max_len=200,
        morgan_bits=1024,
        cross_heads=4,
        dropout=0.15,
        use_interaction=True,
        use_region_head=True,
        use_affinity_prior=False,
        use_morgan=True,
        use_smiles_cnn=True,
        use_multiscale_residue=True,
        use_drug_res_attention=True,
        use_protein_smiles_attention=True,
    ):
        super().__init__()
        self.use_interaction = use_interaction
        self.use_region_head = use_region_head
        self.use_affinity_prior = use_affinity_prior
        self.use_morgan = use_morgan
        self.use_smiles_cnn = use_smiles_cnn
        self.use_multiscale_residue = use_multiscale_residue
        self.use_drug_res_attention = use_drug_res_attention
        self.use_protein_smiles_attention = use_protein_smiles_attention
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # drug encoder
        self.drug_gin = DrugGINEncoder(
            in_dim=drug_in_dim, output_dim=hidden_dim, dropout=dropout, dual_pool=True
        )
        self.smiles_enc = SmilesCNNEncoder(
            hidden_dim=hidden_dim, max_len=smiles_max_len, dropout=dropout
        )
        self.smiles_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.morgan_enc = MorganEncoder(n_bits=morgan_bits, output_dim=hidden_dim, dropout=dropout)
        n_drug_branches = 1 + int(use_smiles_cnn) + int(use_morgan)
        self.drug_fuse = nn.Sequential(
            nn.Linear(hidden_dim * n_drug_branches, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # protein encoder
        self.protein_embed = nn.Embedding(27, hidden_dim, padding_idx=0)
        self.protein_global_conv = nn.Conv1d(
            in_channels=max_protein_len, out_channels=32, kernel_size=8
        )
        self.protein_global_fc = Linear(32 * (hidden_dim - 8 + 1), hidden_dim)
        self.residue_encoder = MultiScaleResidueEncoder(embed_dim=hidden_dim, hidden_dim=hidden_dim)
        self.protein_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # cross-attention (drug_to_res also gates the optional residue head)
        self.drug_to_res = nn.Linear(hidden_dim, hidden_dim)
        self.residue_head = nn.Linear(hidden_dim, 1)

        self.drug_res_attn = nn.MultiheadAttention(
            hidden_dim, cross_heads, dropout=dropout, batch_first=True
        )
        self.prot_smiles_attn = nn.MultiheadAttention(
            hidden_dim, cross_heads, dropout=dropout, batch_first=True
        )
        self.cross_ln = nn.LayerNorm(hidden_dim * 2)

        self.prior_encoder = None
        if use_affinity_prior:
            self.prior_encoder = nn.Sequential(
                nn.Linear(7, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # affinity head
        fusion_dim = hidden_dim * 5 if use_interaction else hidden_dim * 3
        if use_affinity_prior:
            fusion_dim += hidden_dim
        self.fusion_ln = nn.LayerNorm(fusion_dim)
        self.fc1 = nn.Linear(fusion_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)

    def _smiles_features(self, smiles_ids):
        tokens = self.smiles_enc(smiles_ids)
        mask = smiles_ids > 0
        pooled = torch.cat([masked_mean(tokens, mask), masked_max(tokens, mask)], dim=-1)
        return tokens, mask, self.smiles_pool(pooled)

    def _protein_features(self, protein, lengths):
        emb = self.protein_embed(protein)
        global_x = self.protein_global_conv(emb)
        global_x = global_x.view(global_x.size(0), -1)
        global_vec = F.relu(self.protein_global_fc(global_x))

        if self.use_multiscale_residue:
            residue_feat = self.residue_encoder(emb)
        else:
            # ablation: keep token-level embeddings for attention, drop the multi-kernel convolutions
            residue_feat = emb
        mask = length_mask(lengths, residue_feat.size(1), residue_feat.device)
        residue_feat = residue_feat * mask.unsqueeze(-1).float()
        if self.use_multiscale_residue:
            residue_mean = masked_mean(residue_feat, mask)
            residue_max = masked_max(residue_feat, mask)
        else:
            # ablation: protein summary comes only from the global branch (same head width)
            residue_mean = torch.zeros_like(global_vec)
            residue_max = torch.zeros_like(global_vec)
        pooled = torch.cat([global_vec, residue_mean, residue_max], dim=-1)
        return self.protein_fuse(pooled), residue_feat, mask

    def _cross_interaction(self, drug_vec, smiles_tokens, smiles_mask, residue_feat, residue_mask):
        batch_size = drug_vec.size(0)
        if self.use_drug_res_attention:
            drug_query = self.drug_to_res(drug_vec).unsqueeze(1)
            drug_res, _ = self.drug_res_attn(
                drug_query, residue_feat, residue_feat, key_padding_mask=~residue_mask
            )
            drug_res = drug_res.squeeze(1)
        else:
            drug_res = drug_vec.new_zeros(batch_size, self.hidden_dim)

        if self.use_protein_smiles_attention:
            protein_query = masked_mean(residue_feat, residue_mask).unsqueeze(1)
            prot_smiles, _ = self.prot_smiles_attn(
                protein_query, smiles_tokens, smiles_tokens, key_padding_mask=~smiles_mask
            )
            prot_smiles = prot_smiles.squeeze(1)
        else:
            prot_smiles = drug_vec.new_zeros(batch_size, self.hidden_dim)

        cross = self.cross_ln(torch.cat([drug_res, prot_smiles], dim=-1))
        drug_res_cross, prot_smiles_cross = cross.split(self.hidden_dim, dim=-1)
        if not self.use_drug_res_attention:
            drug_res_cross = torch.zeros_like(drug_res_cross)
        if not self.use_protein_smiles_attention:
            prot_smiles_cross = torch.zeros_like(prot_smiles_cross)
        return torch.cat([drug_res_cross, prot_smiles_cross], dim=-1)

    def forward(self, batch):
        b = batch.num_graphs
        protein = batch.protein.view(b, -1)
        lengths = batch.prot_len.view(-1).clamp(min=1, max=protein.size(1))
        smiles_ids = batch.smiles_ids.view(b, -1)
        morgan = batch.morgan_fp.view(b, -1)

        graph_vec = self.drug_gin(batch.x, batch.edge_index, batch.batch)
        smiles_tokens, smiles_mask, smiles_vec = self._smiles_features(smiles_ids)
        drug_parts = [graph_vec]
        if self.use_smiles_cnn:
            drug_parts.append(smiles_vec)
        if self.use_morgan:
            drug_parts.append(self.morgan_enc(morgan))
        drug_vec = self.drug_fuse(torch.cat(drug_parts, dim=-1))

        protein_vec, residue_feat, residue_mask = self._protein_features(protein, lengths)

        residue_logits, residue_prob = None, None
        if self.use_region_head:
            drug_gate = self.drug_to_res(drug_vec).unsqueeze(1)
            residue_logits = self.residue_head(torch.tanh(residue_feat + drug_gate)).squeeze(-1)
            residue_logits = residue_logits.masked_fill(~residue_mask, 0.0)
            residue_prob = torch.sigmoid(residue_logits)

        parts = [drug_vec, protein_vec, drug_vec * protein_vec]
        if self.use_interaction:
            parts.append(
                self._cross_interaction(
                    drug_vec, smiles_tokens, smiles_mask, residue_feat, residue_mask
                )
            )
        if self.use_affinity_prior:
            prior = batch.affinity_prior.view(b, -1)
            parts.append(self.prior_encoder(prior))
        h = self.fusion_ln(torch.cat(parts, dim=-1))
        h = F.dropout(F.relu(self.fc1(h)), p=self.dropout, training=self.training)
        h = F.dropout(F.relu(self.fc2(h)), p=self.dropout, training=self.training)
        return self.out(h).squeeze(-1), residue_logits, residue_prob
