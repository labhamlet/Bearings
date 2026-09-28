"""SELD probe that fuses frozen GRAM-T (mono/semantic) embeddings with
SphereV5 (spatial) embeddings.

Fixes relative to the previous version:

1. `inject_spatial_tokens="Learn"` no longer crashes when `spatial_encoder`
   is None (D_sp/F_sp were read off the encoder unconditionally).
2. Arbitrary sequence lengths. SphereV5's `pass_through_encoder` has a fixed
   positional-embedding grid (target_length frames, e.g. 200 = 2 s). SELD
   sequences are usually longer, which made the pos-embed addition blow up.
   Both streams are now run in non-overlapping `target_length` windows folded
   into the batch dim, then re-concatenated along the token-time axis (the
   zero-padded tail tokens are trimmed).
3. GRAM-T is chunked the same way (`MonoEncoderSpec.chunk_len`), since it was
   also only ever pretrained on `target_length` crops.
4. `mono_spec_from_sphere()` / `build_sphere_seld()` helpers so the mono spec
   (dims, n_freq, chunk length) is pulled straight from the SphereV5
   checkpoint instead of being hand-specified.
5. Input layout is asserted (B, 7, T, F) with ch0=W logmel, ch1:4=YZX logmel,
   ch4:7=AIV -- the same 7-channel layout `_extract_sphere_features` caches.
6. NEW: the spatial half lives in `sphere_backbone.SpatialStreamMixin`, shared
   with the SPEAR/Dasheng probe, and two further arms are available:
   `"Finetune"` (pre-trained encoder, weights updated) and `"Scratch"` (same
   architecture, random init).  See sphere_backbone.py.

Usage
-----
    from sphere_seld_fusion import build_sphere_seld
    from sphere.models.sphere_v5 import SphereV5

    sphere = SphereV5.load_from_checkpoint(
        params['sphere_ckpt'], map_location='cpu',
        patch_strategy=patch_strategy,   # ignored by save_hyperparameters
        strict=False,                    # gramt_null_token differs across arms
    )
    sphere.eval()

    model = build_sphere_seld(data_out, params, sphere,
                              inject_spatial_tokens="True").to(device)

For the frozen arm the optimizer may still filter on requires_grad, but
prefer `model.parameter_groups(...)`, which puts the encoder in its own group
with its own LR -- required by the Finetune/Scratch arms.

IMPORTANT (feature side): SphereV5 pretraining normalizes every clip by the
W-channel RMS *before* the mel front end. `_extract_sphere_features` in
cls_feature_class.py must do the same, otherwise the 4 log-mel channels are
offset by a per-file constant relative to pretraining (the AIV channels are
gain-invariant and unaffected):

    audio = audio[:self._nb_channels].float()
    if int(orig_sr) != self._fs:
        audio = resample(audio=audio, orig_sr=int(orig_sr), target_sr=self._fs)
    rms = torch.sqrt(audio[0].pow(2).mean() + 1e-8)   # W-channel RMS
    audio = audio / rms

Also make sure the data generator feeds this probe the *unnormalized* sphere
feature cache (`get_sphere_feat_dir()`), not the StandardScaler-normalized
baseline features. Cache layout is (T, 7*F), channel-major:

    x = rearrange(feat, 't (c f) -> c t f', c=7)      # then batch -> (B,7,T,F)
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from einops import rearrange

# ConvBlock / LearntSpatialFrontEnd are re-exported for callers that imported
# them from here before the split.
from sphere_backbone import (  # noqa: F401
    SPATIAL_MODES,
    ConvBlock,
    LearntSpatialFrontEnd,
    SpatialStreamMixin,
    set_sphere_trainable,
    sphere_encoder_parameters,
)

# =============================================================================
# Mono encoder spec
# =============================================================================

@dataclass
class MonoEncoderSpec:
    name: str
    model: nn.Module                # frozen W-only backbone
    embed_dim: int                  # D_mono (per-frequency token dim)
    n_freq: int                     # F_mono
    chunk_len: Optional[int] = None  # frames per forward pass; None = whole seq
    # (model, w_logmel:(B,1,T,F)) -> (B, T_mono, F_mono*D_mono), F-major flatten
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] = \
        field(default=lambda m, w: m(w, strategy="raw"))


def build_gramt(model_id: str = "labhamlet/gramt-mono") -> nn.Module:
    """Standalone frozen GRAM-T with pretrained weights (local cache first)."""
    try:
        gram = transformers.AutoModel.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True)
    except OSError:
        gram = transformers.AutoModel.from_pretrained(
            model_id, trust_remote_code=True)
    for p in gram.parameters():
        p.requires_grad_(False)
    return gram.eval()


def mono_spec_from_sphere(sphere, name: str = "gram-t",
                          gramt_model_id: str = "labhamlet/gramt-mono",
                          ) -> MonoEncoderSpec:
    """Mono GRAM-T spec for fusion with a SphereV5 spatial encoder.

    Works for all checkpoint flavours:
      * GRAM-conditioned sphere  -> reuse its frozen `gram` (weights already
        restored from the Lightning checkpoint) and its cached geometry.
      * Unconditioned sphere     -> build GRAM-T fresh from the hub/cache and
        probe the token geometry with a dummy forward.
      * Scratch sphere           -> same as unconditioned: the spatial encoder
        is random, the mono encoder stays pre-trained on purpose.
    """
    gram = getattr(sphere, "gram", None)
    if gram is not None:
        embed_dim, n_freq = sphere.gramt_native_dim, sphere.gramt_n_freq
    else:
        gram = build_gramt(gramt_model_id)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, sphere.target_length, sphere.num_mel_bins)
            flat_dim = gram(dummy, strategy="raw").shape[2]
        n_freq = sphere.p_f_dim
        assert flat_dim % n_freq == 0, \
            f"GRAM-T raw dim {flat_dim} not divisible by p_f_dim={n_freq}"
        embed_dim = flat_dim // n_freq

    return MonoEncoderSpec(
        name=name,
        model=gram,
        embed_dim=embed_dim,
        n_freq=n_freq,
        chunk_len=sphere.target_length,   # GRAM-T only ever saw 2 s crops
    )


# =============================================================================
# SELD head (unchanged)
# =============================================================================

class SeldHead(nn.Module):
    def __init__(self, in_dim, out_shape, params):
        super().__init__()
        self.gru = nn.GRU(in_dim, params['rnn_size'], params['nb_rnn_layers'],
                          batch_first=True,
                          dropout=(params['dropout_rate']
                                   if params['nb_rnn_layers'] > 1 else 0.0),
                          bidirectional=True)
        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(
                params['rnn_size'], params['nb_heads'],
                dropout=params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(params['rnn_size']))
        self.fnn_list = nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(
                    params['fnn_size'] if fc_cnt else params['rnn_size'],
                    params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(
            params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'],
            out_shape[-1], bias=True))

    def forward(self, x):                              # (B, T_seld, in_dim)
        x, _ = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]
        for mhsa, ln in zip(self.mhsa_block_list, self.layer_norm_list):
            xin = x
            x, _ = mhsa(xin, xin, xin)
            x = ln(x + xin)
        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        return torch.tanh(self.fnn_list[-1](x))


# =============================================================================
# Fusion probe
# =============================================================================

class SphereV5SELD(SpatialStreamMixin, nn.Module):
    """SELD probe fusing GRAM-T (mono) + SphereV5 (spatial).

    inject_spatial_tokens:
      "True"     -> frozen pre-trained SphereV5 encoder          (ours)
      "Finetune" -> pre-trained SphereV5, encoder weights update (ours, ft)
      "Scratch"  -> same architecture, random init, trained      (no pretrain)
      "False"    -> mono stream only                             (baseline ii)
      "Learn"    -> trainable 2-Conv2D spatial front-end         (baseline iii)

    NOTE on "Scratch": only the SPATIAL stream is randomly initialized. The
    mono stream stays the pre-trained frozen GRAM-T, which is the point --
    this isolates the value of spatial pretraining. For a fully-from-scratch
    system use the bare SELDNet arm.

    Input x: (B, 7, T, F) with ch0 = W log-mel, ch1:4 = Y,Z,X log-mel,
    ch4:7 = normalized active-intensity vector, i.e. exactly what
    FeatureClass._extract_sphere_features caches (after un-flattening).
    """

    def __init__(self, out_shape, params, mono_spec: MonoEncoderSpec,
                 spatial_encoder: Optional[nn.Module] = None,
                 inject_spatial_tokens: str = "True",
                 d_sp: Optional[int] = None, f_sp: Optional[int] = None):
        super().__init__()
        self.params = params
        self.T_seld = out_shape[-2]
        proj_dim = 256
        dropout = params.get('dropout_rate', 0.1)

        # ---- Mono stream (always frozen) ----
        self.mono = mono_spec
        self.add_module("mono_model", mono_spec.model)  # registered for .to()/.eval()
        for p in mono_spec.model.parameters():
            p.requires_grad = False
        mono_spec.model.eval()
        self.D_gr, self.F_gr = mono_spec.embed_dim, mono_spec.n_freq

        # ---- Spatial stream ----
        # Must come AFTER the mono stream is frozen: in the GRAM arm
        # mono_spec.model IS sphere.gram, and set_sphere_trainable() re-freezes
        # it as its last act, so the ordering is belt-and-braces either way.
        self._init_spatial_stream(inject_spatial_tokens, spatial_encoder,
                                  params, proj_dim, dropout, d_sp, f_sp)

        # ---- Mono projection plumbing ----
        self.input_norm_gr = nn.LayerNorm(self.F_gr * self.D_gr)
        self.q_gr = nn.Parameter(torch.randn(self.D_gr) * 0.02)
        self.k_proj_gr = nn.Linear(self.D_gr, self.D_gr, bias=False)
        self.gram_proj = nn.Sequential(
            nn.LayerNorm(self.D_gr), nn.Linear(self.D_gr, proj_dim),
            nn.GELU(), nn.Dropout(dropout))

        # fuse: concat only when a spatial stream is present
        self.feat_proj = nn.Linear(2 * proj_dim, proj_dim) \
            if self.has_spatial else None

        self.head = SeldHead(proj_dim, out_shape, params)

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        self.mono.model.eval()                          # always frozen
        self._sync_spatial_train_mode()                 # see mixin docstring
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _freq_pool(z_flat, query, k_proj, F_dim, D_dim):
        B, T, _ = z_flat.shape
        z = z_flat.view(B, T, F_dim, D_dim)
        attn = (k_proj(z) @ query) / (D_dim ** 0.5)
        attn = attn.softmax(dim=-1).unsqueeze(-1)       # over frequency
        return (z * attn).sum(dim=2)                    # (B, T, D)

    def _avg_pool_time(self, z):                        # (B,T,D)->(B,T_seld,D)
        return F.adaptive_avg_pool1d(
            z.transpose(1, 2), self.T_seld).transpose(1, 2)

    # ------------------------------------------------------------------
    def _mono_tokens(self, w_logmel: torch.Tensor) -> torch.Tensor:
        """w_logmel: (B, 1, T, F) -> (B, T_mono, F_gr*D_gr).

        Runs GRAM in pretraining-length windows so its positional grid always
        matches, then re-concatenates along token time."""
        with torch.no_grad():
            self.mono.model.eval()
            Tc = self.mono.chunk_len
            if Tc is None or w_logmel.shape[2] <= Tc:
                return self.mono.forward_fn(self.mono.model, w_logmel)
            B = w_logmel.shape[0]
            wc, n, T = self._chunk_time(w_logmel, Tc)   # from the mixin
            z = self.mono.forward_fn(self.mono.model, wc)   # (B*n, Tg, F*D)
            Tg = z.shape[1]
            z = rearrange(z, "(b n) t fd -> b (n t) fd", b=B, n=n)
            # drop tokens that cover only zero padding
            frames_per_tok = Tc / Tg
            t_valid = math.ceil(T / frames_per_tok)
            return z[:, :t_valid]

    # ------------------------------------------------------------------
    def forward(self, x):
        """x: (B, 7, T, F): ch0=W, ch1:4=YZX, ch4:7=AIV."""
        assert x.dim() == 4 and x.shape[1] == 7, \
            f"expected (B, 7, T, F), got {tuple(x.shape)}"

        # ---- Mono stream (frozen GRAM-T on the W log-mel channel) ----
        z_gr = self._mono_tokens(x[:, 0:1])
        z_gr = self.input_norm_gr(z_gr)
        z_gr = self._freq_pool(z_gr, self.q_gr, self.k_proj_gr,
                               self.F_gr, self.D_gr)
        z_gr = self.gram_proj(self._avg_pool_time(z_gr))     # (B,T_seld,256)

        if not self.has_spatial:                             # baseline (ii)
            return self.head(z_gr)

        # ---- Spatial stream (SphereV5 frozen/ft/scratch, or learnt) ----
        z_sp = self._spatial_tokens(x)
        z_sp = self.input_norm_sp(z_sp)
        z_sp = self._freq_pool(z_sp, self.q_sp, self.k_proj_sp,
                               self.F_sp, self.D_sp)
        z_sp = self.sphere_proj(self._avg_pool_time(z_sp))   # (B,T_seld,256)

        # ---- Fuse ----
        z = self.feat_proj(torch.cat([z_sp, z_gr], dim=-1))  # (B,T_seld,256)
        return self.head(z)


# =============================================================================
# Factory
# =============================================================================

def build_sphere_seld(out_shape, params, sphere,
                      inject_spatial_tokens: str = "True") -> SphereV5SELD:
    """One-liner: fuse the GRAM-T living inside a SphereV5 checkpoint with the
    SphereV5 spatial encoder itself.

    `sphere` is required in every arm, including "Learn"/"False", because the
    mono spec reads its token geometry (p_f_dim, target_length, num_mel_bins)
    off it."""
    mono = mono_spec_from_sphere(sphere)
    return SphereV5SELD(
        out_shape, params, mono,
        spatial_encoder=sphere,
        inject_spatial_tokens=inject_spatial_tokens,
    )