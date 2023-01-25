import torch
import torch.nn as nn
from super_image import Trainer, TrainingArguments, EdsrModel, EdsrConfig


class NLSA(nn.Module):
    def __init__(self,  n_attention):
        self.n_attentions = n_attention
        self.linear = nn.Linear(3, 1)

    def forward(self, inputs):
        if len(inputs.size()) == 4:
            n_batch, n_channels, h, w = inputs.size()
        elif len(inputs.size()) == 3:
            n_channels, h, w = inputs.size()
            n_batch = None
        else:
            raise ValueError(f"Expect (N,C,H,W) or (C,H,W) dims of inputs. Incoming inputs dims are {inputs.size()}")

        m = min(h*w/144, 128)
        if n_batch:
            random_matrices = torch.randn(n_batch,n_channels, m)
        else:
            random_matrices = torch.randn(n_channels, m)

        hash_code = torch.matmul(inputs.permute(0), random_matrices)



class NLSN(EdsrModel):
    def __init__(self):
        super(EdsrModel).__init__()
