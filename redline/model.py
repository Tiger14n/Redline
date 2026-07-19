"""Frame-level timing model: mel -> conv frontend -> rotary transformer -> per-frame heads.

Encoder-only, ~45M params, no temporal downsampling (output is per-frame at 50 fps).
Heads:
  beat   : beat-here logit
  down   : downbeat (measure start) logit
  off    : sub-frame offset in [-OFFSET_RANGE, OFFSET_RANGE] frames (recovers ms precision)
  tempo  : BPM-bin logits (per-frame tempo curve)
  change : MAJOR-redline logit (audible tempo/meter/phase changes)
  red    : ALL-redline (mapper notation) logit — where a human writes a redline, including
           the convention re-anchors that carry no audible change (red_head=True models)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from .targets import OFFSET_RANGE, N_TEMPO_BINS


class ConvFrontend(nn.Module):
    """[B,F,M] mel -> [B,F,d_model]. Conv pools frequency; time (F) is preserved. The
    (5,3) kernels give 5 frames of local temporal context to keep transients sharp."""
    def __init__(self, n_mels, d_model):
        super().__init__()
        ch = [1, 16, 32, 64, 64]
        layers = []
        for i in range(4):
            layers += [
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=(5, 3), stride=(1, 2), padding=(2, 1)),
                nn.BatchNorm2d(ch[i + 1]),
                nn.GELU(),
            ]
        self.conv = nn.Sequential(*layers)
        freq_out = n_mels // 16                       # 4 strides of 2
        self.proj = nn.Linear(ch[-1] * freq_out, d_model)

    def forward(self, mel):                            # mel: [B,F,M]
        x = mel.unsqueeze(1)                           # [B,1,F,M]
        x = self.conv(x)                               # [B,C,F,M/16]
        b, c, f, m = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, f, c * m)
        return self.proj(x)                            # [B,F,d_model]


class ContrastiveChangeHead(nn.Module):
    """Detects boundaries by comparing the past 1 second directly to the future 1 second."""
    def __init__(self, d_model):
        super().__init__()
        # 50 frames @ 50fps = 1 second
        self.past_pool = nn.Conv1d(d_model, d_model, kernel_size=50, padding=49, groups=d_model)
        self.future_pool = nn.Conv1d(d_model, d_model, kernel_size=50, padding=49, groups=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):                              # x: [B,T,D]
        x_t = x.transpose(1, 2)
        past = self.past_pool(x_t)[..., :-49]          # shifted left
        future = self.future_pool(x_t)[..., 49:]       # shifted right
        feat = torch.cat([past.transpose(1, 2), future.transpose(1, 2)], dim=-1)
        return self.classifier(feat).squeeze(-1)


class RotaryAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.h = n_heads
        self.hd = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryEmbedding(dim=self.hd)
        self.drop = dropout

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # [B,H,T,hd]
        q = self.rotary.rotate_queries_or_keys(q)
        k = self.rotary.rotate_queries_or_keys(k)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        o = o.transpose(1, 2).reshape(b, t, d)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = RotaryAttention(d_model, n_heads, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ffn, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class TimingModel(nn.Module):
    """redline-v1 architecture. `conv_v2`/`change_head_v2` are accepted for checkpoint-args
    compatibility and must be True (the only shipped configuration). `sum_head` folds the
    downbeat logit into the beat prediction (Beat This!, ISMIR'24) — off in redline-v1,
    kept so older specialist checkpoints load."""

    def __init__(self, n_mels=128, d_model=384, depth=10, n_heads=6, ffn=1536, dropout=0.1,
                 sum_head=False, conv_v2=True, change_head_v2=True, red_head=False):
        super().__init__()
        assert conv_v2 and change_head_v2, "redline ships only the conv_v2 + change_head_v2 architecture"
        self.sum_head = sum_head
        self.frontend = ConvFrontend(n_mels, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ffn, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.beat_head = nn.Linear(d_model, 1)
        self.down_head = nn.Linear(d_model, 1)
        self.off_head = nn.Linear(d_model, 1)
        self.tempo_head = nn.Linear(d_model, N_TEMPO_BINS)   # per-frame BPM-bin logits
        self.change_head = ContrastiveChangeHead(d_model)
        # notation head: trained on ALL GT redlines (major_only=False), unlike the change
        # head which sees only major (audible) changes. Perception (beat/down) says where
        # the beats ARE; this head says where a human NOTATES the grid.
        self.use_red = red_head
        if red_head:
            self.red_head = nn.Linear(d_model, 1)

    def forward(self, mel):                            # mel: [B,F,M]
        x = self.frontend(mel)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        beat_logit = self.beat_head(x).squeeze(-1)
        down_logit = self.down_head(x).squeeze(-1)
        ret = {
            # sum head: downbeat logit ADDS into the beat prediction -> downbeats are always beats
            "beat": (beat_logit + down_logit) if self.sum_head else beat_logit,
            "down": down_logit,
            "off": torch.tanh(self.off_head(x).squeeze(-1)) * OFFSET_RANGE,
            "tempo": self.tempo_head(x),               # [B,F,N_BINS]
            "change": self.change_head(x).squeeze(-1),  # MAJOR-redline logits
        }
        if self.use_red:
            ret["red"] = self.red_head(x).squeeze(-1)  # ALL-redline (notation) logits
        return ret

    def param_groups(self):
        """(muon, adamw) split: Muon takes the 2-D trunk matrices; heads, frontend and all
        non-2-D tensors go to AdamW."""
        muon, adamw = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_head_or_frontend = any(x in name for x in ("frontend", "head"))
            if p.ndim == 2 and not is_head_or_frontend:
                muon.append(p)
            else:
                adamw.append(p)
        return muon, adamw


if __name__ == "__main__":
    m = TimingModel(d_model=512, depth=14, n_heads=8, ffn=2048, red_head=True)
    n = sum(p.numel() for p in m.parameters())
    muon, adamw = m.param_groups()
    out = m(torch.randn(2, 1500, 128))
    print(f"params: {n/1e6:.1f}M  (Muon {sum(p.numel() for p in muon)/1e6:.1f}M / "
          f"AdamW {sum(p.numel() for p in adamw)/1e6:.1f}M)")
    print("out:", {k: tuple(v.shape) for k, v in out.items()})
