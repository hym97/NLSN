import torch
import torch.nn as nn
from super_image import Trainer, TrainingArguments, EdsrModel, EdsrConfig


class NLSA(nn.Module):
    def __init__(self,  n_attention, k, n_channels):
        self.n_attentions = n_attention
        self.k = k
        self.n_channels = n_channels
        self.linear = nn.Linear(3, 1)

    def forward(self, inputs):
        if len(inputs.size()) == 4:
            n_batch, c, h, w = inputs.size()
        else:
            raise ValueError(f"Expect (N,C,H,W) dims of inputs. Incoming inputs dims are {inputs.size()}")


        inputs_flatten = torch.flatten(inputs, 2, 3).permute(0,2,1)

        m = min(h*w/144, 128)

        random_matrices = torch.randn(self.n_attentions, c , m)

        rotated_inputs = torch.matmul(inputs_flatten, random_matrices)

        hash_code = torch.max(rotated_inputs, 3).values
        sorted_hash, sorted_index = torch.sort(hash_code, 2, descending=True)

        attention_buckets = torch.split(sorted_index, self.k, dim=2) # dims=(N,n_attentions,k)
        output = torch.zeros(n_batch,h*w,c)
        for bucket in attention_buckets:
            pass













class NLSN(EdsrModel):
    def __init__(self):
        super(EdsrModel).__init__()
