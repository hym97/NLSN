import torch
import torch.nn as nn
from super_image import Trainer, TrainingArguments, EdsrModel, EdsrConfig


class NLSA(nn.Module):
    def __init__(self,  n_attention, k, n_channels, n_channels_out):
        super().__init__()
        self.n_attentions = n_attention
        self.k = k
        self.n_channels = n_channels
        self.theta = nn.Linear(n_channels,n_channels_out)
        self.g = nn.Linear(n_channels, n_channels_out)

    def forward(self, inputs):
        if len(inputs.size()) == 4:
            n_batch, c, h, w = inputs.size()
        else:
            raise ValueError(f"Expect (N,C,H,W) dims of inputs. Incoming inputs dims are {inputs.size()}")


        inputs_flatten = torch.flatten(inputs, 2, 3).permute(0,2,1)

        m = min(h*w//self.k, 128)

        random_matrices = torch.randn(self.n_attentions, c , m)

        rotated_inputs = torch.matmul(inputs_flatten.unsqueeze(1), random_matrices)

        hash_code = torch.max(rotated_inputs, 3).values
        sorted_hash, sorted_index = torch.sort(hash_code, 2, descending=True)

        attention_buckets = torch.split(sorted_index, self.k, dim=2) # dims=(N,n_attentions,k)
        output = torch.zeros(self.n_attentions,n_batch,h*w,c)
        for bucket in attention_buckets:
            for i in range(self.n_attentions):
                curr_idx = bucket[:,i,:]
                gathered_values = torch.stack([inputs_flatten[i].index_select(0, curr_idx[i]) for i in range(inputs_flatten.size(0))], dim=0)
                m1 = self.theta(gathered_values)
                m2 = self.g(gathered_values)

                c = m1.repeat_interleave(144, dim=1)
                d = m1

                # reshape the tensors
                c = c.view(10, 144, 144, 3)
                # d = d.view(10,  144, 1, 3)
                #
                # # Perform element-wise multiplication
                # result = c * d
                # print(result.size())
                # print(m1[0,0] * m1[0] == result[0,0,:,:])

                d2 = d.view(10,1,144,3)
                result2 = c * d2
                print(m1[0,0] * m1[0] == result2[0,0,:,:])

                theta_g = result2*m2.view(10,1,144,3)
                print(theta_g[2,1] == result2[2,1]*m2[2])

                sum_att = result2.sum(2)
                att_i = (theta_g / sum_att.view(10,144,1,3)).sum(2)
                print(att_i[1,2] == (result2[1,2]*m2[1] / result2[1,2].sum(0)).sum(0))


                output[i].scatter_(1, curr_idx.unsqueeze(-1).expand(-1, -1, 3), att_i)

a = torch.randn(10,3,100,100)
model = NLSA(4,144,3,3)
model(a)

#
#
#
#
#
# class NLSN(EdsrModel):
#     def __init__(self):
#         super(EdsrModel).__init__()
