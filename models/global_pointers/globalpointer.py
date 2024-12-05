import torch
import torch.nn as nn
import logging

from .globalpointer_utils import add_mask_tril, lengths_to_mask


logger = logging.getLogger(__name__)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, unsqueeze_dim=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("n , d -> n d", t, inv_freq)
        if unsqueeze_dim:
            freqs = freqs.unsqueeze(unsqueeze_dim)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)

    def forward(self, t, seqlen=-2, past_key_value_length=0):
        # t shape [bs, dim, seqlen, seqlen]
        sin, cos = (
            self.sin[past_key_value_length : past_key_value_length + seqlen, :],
            self.cos[past_key_value_length : past_key_value_length + seqlen, :],
        )
        t1, t2 = t[..., 0::2], t[..., 1::2]
        # 奇偶交错
        return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(
            -2, -1
        )


INF = 1e4
EPSILON = 1e-5


class LinearRelu(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearRelu, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))
    


# from https://github.com/JunnYu/GPLinker_pytorch/blob/dev/duee_v1/models.py
class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, gp_heads=12, num_dense=1, head_size=64, RoPE=True, tril_mask=True, max_length=512):
        super().__init__()
        self.gp_heads = gp_heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        if num_dense == 1:
            self.dense = nn.Linear(hidden_size, gp_heads * 2 * head_size)
        else:
            self.dense = nn.Sequential(
                *[LinearRelu(hidden_size, hidden_size) for _ in range(num_dense - 1)],
                nn.Linear(hidden_size, gp_heads * 2 * head_size),
            )
        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length, unsqueeze_dim=-2)  # n1d


    def forward(self, inputs, attention_mask=None):
        inputs = self.dense(inputs)
        bs, seqlen = inputs.shape[:2]

        # method 1
        inputs = inputs.reshape(bs, seqlen, self.gp_heads, 2, self.head_size)
        qw, kw = inputs.unbind(axis=-2)

        # method 2
        # inputs = inputs.reshape(bs, seqlen, self.heads, 2 * self.head_size)
        # qw, kw = inputs.chunk(2, axis=-1)

        # original
        # inputs = inputs.chunk(self.heads, axis=-1)
        # inputs = torch.stack(inputs, axis=-2)
        # qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]

        # RoPE编码
        if self.RoPE:
            qw, kw = self.rotary(qw, seqlen), self.rotary(kw, seqlen)

        # 计算内积
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # 排除padding
        # huggingface's attention_mask
        mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
        )
        logits = logits - mask * INF

        # 排除下三角
        if self.tril_mask:
            # 排除下三角
            # torch.tril 无法导出ONNX opet12
            # tril_mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            tril_mask = lengths_to_mask(torch.arange(seqlen), seqlen)[None, None, :, :].expand(bs, 1, -1, -1).to(
                inputs.device)  # boolean

            logits = logits - tril_mask * INF

            # 合并两个mask，方便后续做负采样
            mask = torch.logical_or(mask, tril_mask)  # boolean

        # scale返回
        return logits / self.head_size ** 0.5, mask


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(self, hidden_size, heads=12, head_size=64, RoPE=True, tril_mask=True, max_length=512):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, head_size * 2)
        self.dense2 = nn.Linear(head_size * 2, heads * 2)
        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length)

    def forward(self, inputs, attention_mask=None):
        bs, seqlen = inputs.shape[:2]
        inputs = self.dense1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            qw, kw = self.rotary(qw, seqlen), self.rotary(kw, seqlen)

        # 计算内积
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.head_size ** 0.5
        bias = self.dense2(inputs).transpose(1, 2) / 2  # 'bnh->bhn'
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # 排除padding
        # huggingface's attention_mask
        mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
        )
        logits = logits - mask * INF

        # 排除下三角
        if self.tril_mask:
            # 排除下三角
            # torch.tril 无法导出ONNX opet12
            # tril_mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            tril_mask = lengths_to_mask(torch.arange(seqlen), seqlen)[None, None, :, :].expand(bs, 1, -1, -1).to(
                inputs.device)  # boolean

            logits = logits - tril_mask * INF

            # 合并两个mask
            mask = torch.logical_or(mask, tril_mask)  # boolean

        return logits, mask