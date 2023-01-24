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
        else:
            raise ValueError(f"Expect (N,C,H,W) or (C,H,W) dims of inputs. Incoming inputs dims are {inputs.size()}")

        m = min(h*w/144, 128)
        random_matrices = torch.randn()



class NLSN(EdsrModel):
    def __init__(self):
        super(EdsrModel).__init__()
