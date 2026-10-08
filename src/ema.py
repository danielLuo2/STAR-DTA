import copy

import torch


class ModelEMA:
    """Exponential moving average of model weights for stabler evaluation."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        for s_buffer, buffer in zip(self.shadow.buffers(), model.buffers()):
            s_buffer.copy_(buffer)

    def state_dict(self):
        return self.shadow.state_dict()
