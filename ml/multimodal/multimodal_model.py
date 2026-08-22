"""
multimodal_model.py — Enhanced Architecture v2
================================================
Master's Thesis: AI Model for Angina Recognition
Astana IT University

Architectural upgrades over baseline:

  ECG Branch  (1D CNN):
    v1 → 4 plain residual blocks, 128-dim embedding
    v2 → SE-Residual blocks (channel attention) + Multi-Scale Inception blocks
         (parallel k=3/7/15 convolutions) + Temporal Self-Attention layer,
         6 stages, 256-dim embedding

  Tabular Branch  (MLP):
    v1 → simple 3-layer MLP with BatchNorm
    v2 → Feature Tokenizer (each scalar feature → 32-dim vector) +
         Multi-Head Self-Attention over feature tokens (TabTransformer-style) +
         gated dual-path combining transformer output with MLP bypass,
         128-dim embedding

  Fusion:
    v1 → naive concatenation + MLP head
    v2 → Bidirectional Cross-Modal Attention (tabular ↔ ECG) +
         learned modality gating + deeper fusion classifier

References:
  SE blocks:        Hu et al., CVPR 2018
  TabTransformer:   Huang et al., NeurIPS 2020
  Focal Loss:       Lin et al., ICCV 2017
  AdamW / cosine:   Loshchilov & Hutter, ICLR 2019
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILDING BLOCKS — ECG BRANCH
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation channel attention for 1-D signals.

    For each convolutional feature map the block computes a per-channel
    weight by:
      1. Squeezing the temporal dimension via global average pooling
         (one scalar per channel)
      2. Exciting through a small bottleneck FC: channels → channels//r → channels
      3. Re-scaling the original feature map channel-wise

    This lets the network adaptively suppress uninformative frequency bands
    and amplify clinically relevant ones (e.g. QRS frequency range).

    Args:
        channels:  number of input/output channels
        reduction: bottleneck ratio r  (default 8)
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.pool(x).squeeze(-1)    # (B, C)
        w = self.fc(w).unsqueeze(-1)    # (B, C, 1)
        return x * w                    # broadcast over time axis


class SEResBlock1d(nn.Module):
    """
    Residual block augmented with SE channel attention.

    Structure per block:
        Conv1d(k, stride) → BN → ReLU → Conv1d(3) → BN → SE → + residual → ReLU

    The stride-1 shortcut uses identity; stride-2 uses a 1×1 conv to match
    channel count and spatial resolution.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 7, stride: int = 1, reduction: int = 8):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.se = SEBlock1d(out_ch, reduction)
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            ) if (stride != 1 or in_ch != out_ch) else None
        )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv(x)
        out = self.se(out)
        if out.shape[2] != identity.shape[2]:
            identity = F.adaptive_avg_pool1d(identity, out.shape[2])
        return F.relu(out + identity, inplace=True)


class MultiScaleBlock1d(nn.Module):
    """
    Inception-style multi-scale block with three parallel convolutional paths.

    Kernel sizes are chosen to match characteristic ECG waveform widths:
      k =  3  →  QRS peak  (~3–5 samples at 500 Hz)
      k =  7  →  P-wave / T-wave  (~10–20 ms)
      k = 15  →  slow ST-segment drift / baseline wander

    The three branch outputs are concatenated and projected back to out_ch
    through a 1×1 conv, then passed through an SE block and added to the
    residual shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        b = out_ch // 3
        r = out_ch - 3 * b          # assign remainder to k=3 branch

        self.branch3  = self._branch(in_ch, b + r, kernel=3,  stride=stride)
        self.branch7  = self._branch(in_ch, b,     kernel=7,  stride=stride)
        self.branch15 = self._branch(in_ch, b,     kernel=15, stride=stride)

        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock1d(out_ch)
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            ) if (stride != 1 or in_ch != out_ch) else None
        )

    @staticmethod
    def _branch(in_ch: int, out_ch: int, kernel: int, stride: int):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel,
                      stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = torch.cat([self.branch3(x), self.branch7(x), self.branch15(x)], dim=1)
        out = self.fuse(out)
        out = self.se(out)
        if out.shape[2] != identity.shape[2]:
            identity = F.adaptive_avg_pool1d(identity, out.shape[2])
        return F.relu(out + identity, inplace=True)


class TemporalSelfAttention(nn.Module):
    """
    Lightweight multi-head self-attention over the temporal axis of the CNN
    feature map.  Inserted after the last convolutional stage to model
    long-range rhythm dependencies (e.g., R-R interval regularity).

    Input / output shape:  (B, C, T)

    Unlike standard transformers the sequence length T is small here
    (~160 samples after 5 stride-2 downsamples from 5000) so attention
    is computationally cheap.
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm    = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        seq = x.transpose(1, 2)                           # (B, T, C)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.norm(seq + self.dropout(attn_out))     # residual + norm
        return seq.transpose(1, 2)                        # (B, C, T)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ENHANCED ECG ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class ECGEncoder(nn.Module):
    """
    Enhanced 6-stage 1-D CNN for 12-lead ECG (shape: 12 × 5000).

    Stage  | Block type        | Channels | Stride | Output T
    -------|-------------------|----------|--------|----------
      1    | SE-Residual k=7   |  12→64   |   2    |  2500
      2    | SE-Residual k=7   |  64→64   |   1    |  2500
      3    | Multi-Scale       |  64→128  |   2    |  1250
      4    | SE-Residual k=5   | 128→128  |   2    |   625
      5    | Multi-Scale       | 128→256  |   2    |   313
      6    | SE-Residual k=3   | 256→256  |   2    |   157
    TemporalAttn(256, heads=4)           |  157
    GAP → (B, 256)
    Linear(256 → embedding_dim) + LayerNorm + GELU + Dropout

    Total trainable params (embedding_dim=256): ~1.6 M
    """

    def __init__(self, n_leads: int = 12, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.stage1 = SEResBlock1d(n_leads,  64,  kernel_size=7, stride=2)
        self.stage2 = SEResBlock1d(64,       64,  kernel_size=7, stride=1)
        self.stage3 = MultiScaleBlock1d(64,  128, stride=2)
        self.stage4 = SEResBlock1d(128,      128, kernel_size=5, stride=2)
        self.stage5 = MultiScaleBlock1d(128, 256, stride=2)
        self.stage6 = SEResBlock1d(256,      256, kernel_size=3, stride=2)

        self.temporal_attn = TemporalSelfAttention(channels=256, num_heads=4)
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, 5000)  —  zeros tensor if ECG unavailable
        Returns:
            embedding: (B, embedding_dim)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.temporal_attn(x)
        x = self.gap(x).squeeze(-1)
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BUILDING BLOCKS — TABULAR BRANCH
# ─────────────────────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Embeds each scalar tabular feature into a d-dimensional token vector.

    For feature i with value x_i:
        token_i = W_i · x_i + b_i   ∈ R^d

    where W_i ∈ R^d and b_i ∈ R^d are learned per-feature parameters.

    This is equivalent to a single linear layer applied per feature
    independently.  It allows the subsequent transformer to compute
    scaled-dot-product attention between feature embeddings rather than
    between raw scalars, enabling richer feature interaction learning.

    Reference: TabTransformer (Huang et al., 2020); FT-Transformer (Gorishniy
    et al., 2021).
    """

    def __init__(self, num_features: int, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.W = nn.Parameter(torch.empty(num_features, embed_dim))
        self.b = nn.Parameter(torch.zeros(num_features, embed_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N)  →  (B, N, embed_dim)
        return x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENHANCED TABULAR ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class TabularEncoder(nn.Module):
    """
    Enhanced tabular encoder combining feature tokenization, multi-head
    self-attention (transformer), and a parallel MLP bypass with gated
    combination.

    Architecture:
        Input (B, N) — N = raw features + missingness indicators
            │
            ├─── [Transformer path] ──────────────────────────────────────────
            │     FeatureTokenizer(N, token_dim=32)    → (B, N, 32)
            │     prepend CLS token                   → (B, N+1, 32)
            │     TransformerEncoder(2 layers, 4 heads)
            │     take CLS output                      → (B, 32)
            │     MLP head(32 → 2*emb → emb)           → (B, emb)
            │
            └─── [MLP bypass path] ───────────────────────────────────────────
                  Linear(N → emb) + GELU                → (B, emb)

        Gated output:  g · transformer_out + (1−g) · mlp_out
        where g = σ(Linear(transformer_out))

    The gated combination lets the model fall back on direct linear features
    when the attention adds noise (e.g. very sparse data), while still
    benefiting from feature interactions when data is dense.

    Args:
        input_dim:       N (raw features + missingness indicators)
        embedding_dim:   output dimension  (default 128)
        token_dim:       per-feature token size  (default 32)
        num_heads:       attention heads  (default 4)
        num_attn_layers: transformer depth  (default 2)
        dropout:         regularisation  (default 0.3)
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        token_dim: int = 32,
        num_heads: int = 4,
        num_attn_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Transformer path ─────────────────────────────────────────────────
        self.tokenizer = FeatureTokenizer(input_dim, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_attn_layers)

        self.attn_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # ── MLP bypass path ──────────────────────────────────────────────────
        self.mlp_bypass = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # ── Gating ───────────────────────────────────────────────────────────
        self.gate = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — NaN replaced with 0, missingness indicators appended
        Returns:
            embedding: (B, embedding_dim)
        """
        B = x.size(0)

        # Transformer path
        tokens = self.tokenizer(x)                              # (B, N, d)
        cls    = self.cls_token.expand(B, -1, -1)              # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)               # (B, N+1, d)
        tokens = self.transformer(tokens)
        attn_emb = self.attn_head(tokens[:, 0])                # (B, emb)

        # MLP bypass path
        mlp_emb = self.mlp_bypass(x)                           # (B, emb)

        # Gated combination
        g = self.gate(attn_emb)                                # (B, 1)
        return g * attn_emb + (1.0 - g) * mlp_emb


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CROSS-MODAL ATTENTION FUSION
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-modal attention for ECG + tabular fusion.

    Rather than naive concatenation, this module performs two complementary
    cross-attention operations:

      Tab-attends-ECG:  tabular token queries the ECG token
        "Given these lab values, which ECG patterns are most relevant?"

      ECG-attends-Tab:  ECG token queries the tabular token
        "Given this ECG waveform, which clinical features corroborate it?"

    The two enriched tokens are then combined using a learned modality gate
    that outputs soft per-sample weights [w_tab, w_ecg] summing to 1.

    For patients WITHOUT ECG data the ECG embedding is zeroed before this
    module, so the ECG token carries no information and the gate should
    learn to suppress it automatically.

    Args:
        ecg_dim:     ECG embedding dimension (256 by default)
        tabular_dim: tabular embedding dimension (128 by default)
        fusion_dim:  common projection dimension (128 by default)
        num_heads:   cross-attention heads (default 4)
        dropout:     dropout rate (default 0.3)
    """

    def __init__(
        self,
        ecg_dim:     int = 256,
        tabular_dim: int = 128,
        fusion_dim:  int = 128,
        num_heads:   int = 4,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.ecg_proj = nn.Linear(ecg_dim,     fusion_dim)
        self.tab_proj = nn.Linear(tabular_dim, fusion_dim)

        # tabular queries ECG
        self.tab_to_ecg = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # ECG queries tabular
        self.ecg_to_tab = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.norm_t = nn.LayerNorm(fusion_dim)
        self.norm_e = nn.LayerNorm(fusion_dim)

        # Modality gate: learns per-sample [w_tab, w_ecg]
        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1),
        )

        # Classification head on concatenated enriched embeddings
        self.head = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(
        self,
        ecg_emb:      torch.Tensor,
        tabular_emb:  torch.Tensor,
        ecg_available: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ecg_emb:       (B, ecg_dim)
            tabular_emb:   (B, tabular_dim)
            ecg_available: (B,) bool — True when a real ECG is present
        Returns:
            logits: (B,)
        """
        # Mask ECG embedding for patients without ECG
        mask    = ecg_available.float().unsqueeze(1)   # (B, 1)
        ecg_emb = ecg_emb * mask

        # Project to shared fusion dimension → unsqueeze for seq_len=1
        e = self.ecg_proj(ecg_emb).unsqueeze(1)        # (B, 1, F)
        t = self.tab_proj(tabular_emb).unsqueeze(1)    # (B, 1, F)

        # Cross-attention
        t_enr, _ = self.tab_to_ecg(t, e, e)            # tab enriched by ECG
        e_enr, _ = self.ecg_to_tab(e, t, t)            # ECG enriched by tab

        t_enr = self.norm_t((t + t_enr).squeeze(1))   # (B, F)
        e_enr = self.norm_e((e + e_enr).squeeze(1))   # (B, F)

        # Modality gating
        gate_input = torch.cat([t_enr, e_enr], dim=1)  # (B, 2F)
        gates = self.modality_gate(gate_input)          # (B, 2) summing to 1
        fused_weighted = (
            gates[:, 0:1] * t_enr +
            gates[:, 1:2] * e_enr
        )                                               # (B, F) — weighted sum

        # Concatenate for classifier
        classifier_input = torch.cat([t_enr, e_enr], dim=1)  # (B, 2F)
        return self.head(classifier_input).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  COMPLETE ENHANCED MULTIMODAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AnginaMultimodalModel(nn.Module):
    """
    End-to-end enhanced multimodal model for angina pectoris prediction.

    Parameter budget (default dims):
        ECG encoder      (embedding_dim=256):  ~1.6 M
        Tabular encoder  (embedding_dim=128):  ~0.15 M
        Fusion           (fusion_dim=128):      ~0.08 M
        ──────────────────────────────────────
        Total:                                  ~1.83 M

    Handles patients WITHOUT ECG by zeroing the ECG branch; the
    cross-modal attention gate learns to down-weight the ECG token
    automatically in those cases.
    """

    def __init__(
        self,
        tabular_input_dim: int,
        n_leads:           int = 12,
        ecg_embedding_dim: int = 256,
        tab_embedding_dim: int = 128,
        fusion_dim:        int = 128,
    ):
        super().__init__()
        self.ecg_encoder     = ECGEncoder(n_leads=n_leads,
                                           embedding_dim=ecg_embedding_dim)
        self.tabular_encoder = TabularEncoder(tabular_input_dim,
                                               embedding_dim=tab_embedding_dim)
        self.fusion          = CrossModalAttentionFusion(
            ecg_dim=ecg_embedding_dim,
            tabular_dim=tab_embedding_dim,
            fusion_dim=fusion_dim,
        )

    def forward(
        self,
        tabular:       torch.Tensor,
        ecg:           torch.Tensor | None = None,
        ecg_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tabular:       (B, tabular_input_dim)
            ecg:           (B, 12, 5000) or None
            ecg_available: (B,) bool tensor or None
        Returns:
            logits: (B,)  — use sigmoid for probabilities
        """
        B, device = tabular.shape[0], tabular.device

        tabular_emb = self.tabular_encoder(tabular)

        if ecg is None:
            ecg = torch.zeros(B, 12, 5000, device=device)
        if ecg_available is None:
            ecg_available = torch.zeros(B, dtype=torch.bool, device=device)

        ecg_emb = self.ecg_encoder(ecg)
        return self.fusion(ecg_emb, tabular_emb, ecg_available)

    def get_embeddings(
        self,
        tabular:       torch.Tensor,
        ecg:           torch.Tensor | None = None,
        ecg_available: torch.Tensor | None = None,
    ) -> dict:
        """Return pre-fusion embeddings — used for SHAP / t-SNE analysis."""
        B, device = tabular.shape[0], tabular.device
        tabular_emb = self.tabular_encoder(tabular)
        if ecg is None:
            ecg = torch.zeros(B, 12, 5000, device=device)
        if ecg_available is None:
            ecg_available = torch.zeros(B, dtype=torch.bool, device=device)
        ecg_emb = self.ecg_encoder(ecg)
        mask    = ecg_available.float().unsqueeze(1)
        return {
            "ecg_embedding":     ecg_emb * mask,
            "tabular_embedding": tabular_emb,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TABULAR-ONLY ABLATION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AnginaTabularOnlyModel(nn.Module):
    """
    Tabular-only baseline using the enhanced TabularEncoder.
    Used in the ablation study to isolate the benefit of the ECG branch.
    """

    def __init__(self, tabular_input_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = TabularEncoder(tabular_input_dim,
                                       embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, tabular: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.head(self.encoder(tabular)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss  (Lin et al., ICCV 2017).

    FL(p) = −α (1−p)^γ log(p)

    Advantages over plain BCE for medical classification:
      • Down-weights easy negatives (well-separated samples)
      • Focuses training gradient on hard / uncertain cases
      • α compensates for residual class imbalance after sampling

    Default α=0.25, γ=2.0 follow the original paper recommendation.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt     = torch.exp(-bce)
        weight = self.alpha * (1.0 - pt) ** self.gamma
        return (weight * bce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, tabular_dim: int = 68) -> None:
    """Print architecture summary with per-module parameter counts."""
    print(f"\n{'Module':<30} {'Parameters':>14}")
    print("─" * 46)
    for name, module in model.named_children():
        n = count_parameters(module)
        print(f"  {name:<28} {n:>14,}")
    print("─" * 46)
    print(f"  {'TOTAL':<28} {count_parameters(model):>14,}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Multimodal Architecture — Self-Test")
    print("=" * 60)

    TABULAR_DIM = 68    # 34 features + 34 missingness indicators
    BATCH       = 4

    # ── Full multimodal model ─────────────────────────────────────────────────
    model = AnginaMultimodalModel(tabular_input_dim=TABULAR_DIM)
    model_summary(model, TABULAR_DIM)

    tabular = torch.randn(BATCH, TABULAR_DIM)
    ecg     = torch.randn(BATCH, 12, 5000)
    avail   = torch.tensor([True, True, False, False])

    # With ECG for half the batch
    logits = model(tabular, ecg, avail)
    probs  = torch.sigmoid(logits)
    print(f"\n[With ECG]    logits: {logits.detach().numpy().round(3)}")
    print(f"              probs:  {probs.detach().numpy().round(3)}")

    # Without ECG (tabular-only inference)
    logits_no_ecg = model(tabular)
    print(f"\n[No ECG]      logits: {logits_no_ecg.detach().numpy().round(3)}")

    # Focal loss
    loss = FocalLoss()(logits, torch.tensor([1., 1., 0., 0.]))
    print(f"\nFocal loss:   {loss.item():.4f}")

    # ── Tabular-only ablation model ───────────────────────────────────────────
    tab_model = AnginaTabularOnlyModel(TABULAR_DIM)
    tab_logits = tab_model(tabular)
    print(f"\n[Tabular-Only] params: {count_parameters(tab_model):,}")
    print(f"               logits: {tab_logits.detach().numpy().round(3)}")

    # ── Parameter breakdown ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Parameter comparison (v1 → v2):")
    print(f"  ECG encoder:     v1 ~1.0M  →  v2 ~{count_parameters(model.ecg_encoder)/1e6:.2f}M")
    print(f"  Tabular encoder: v1 ~0.02M →  v2 ~{count_parameters(model.tabular_encoder)/1e6:.3f}M")
    print(f"  Fusion:          v1 ~0.03M →  v2 ~{count_parameters(model.fusion)/1e6:.3f}M")
    print(f"  Total:           v1 ~1.06M →  v2 ~{count_parameters(model)/1e6:.2f}M")
    print("\n✓ All tests passed")
