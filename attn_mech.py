import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class AttnMODULE(nn.Module):
    def __init__(self, numhead, d_model, dropout=0.0):
        super().__init__()

        assert d_model % numhead == 0

        self.num_head = numhead
        self.d_model = d_model
        self.d_k = d_model // numhead

        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

    @staticmethod
    def attn(q_, k_, v_, dropout, mask=None):

        d_k = q_.shape[-1]

        scores = (q_ @ k_.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

            scores = scores.masked_fill(~mask, float('-inf'))

        scores = F.softmax(scores, dim=-1)
        scores = torch.nan_to_num(scores, nan=0.0)
        scores = dropout(scores)
        
        return scores @ v_

    def forward(self, query, key, value, mask=None):

        batch, seq_q, _ = query.shape
        _, seq_k, _ = key.shape

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(batch, seq_q, self.num_head, self.d_k).transpose(1, 2)

        k = k.view(batch, seq_k, self.num_head, self.d_k).transpose(1, 2)

        v = v.view(batch, seq_k, self.num_head, self.d_k).transpose(1, 2)

        out = self.attn(q, k, v, self.dropout, mask)

        out = (
            out.transpose(1, 2)
               .contiguous()
               .view(batch, seq_q, self.d_model)
        )

        return self.wo(out)
    
class AttnWrapper(nn.Module):
    def __init__(self , in_channel , numhead):
        super().__init__()
        self.channel = in_channel
        self.attn = AttnMODULE(numhead , in_channel)

        self.norm = nn.LayerNorm(in_channel)

    def forward(self , x):
        b , c , h , w = x.shape
        x_flat = x.view(b , c , -1).permute(0 , 2 , 1)

        x_norm = self.norm(x_flat)

        out = self.attn(x_norm , x_norm , x_norm)

        out += x_flat
        return out.permute(0 , 2 , 1).view(b , c , h , w)