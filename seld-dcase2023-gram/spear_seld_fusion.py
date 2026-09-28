"""SELD probe that fuses frozen SPEAR (mono/semantic, Zipformer on raw 16 kHz
waveform, https://arxiv.org/abs/2510.25955) or Dasheng embeddings with
SphereV5 (spatial) embeddings.

Refactored to mirror sphere_seld_fusion.py (the GRAM-T / SphereV5 probe):

1. `inject_spatial_tokens` in {"True", "Finetune", "Scratch", "Learn",
   "False"} instead of the old `freeze_backbone` flag, so every arm (ours /
   ours fine-tuned / no-pretraining control / mono-only baseline ii / learnt
   front-end baseline iii) lives in one class. "Learn" no longer needs a
   SphereV5 instance at all (pass d_sp/f_sp explicitly).
2. Arbitrary sequence lengths on the spatial stream. SphereV5's
   `pass_through_encoder` has a fixed positional-embedding grid
   (target_length frames, e.g. 200 = 2 s); SELD sequences are usually longer.
   The spatial input is run in non-overlapping `target_length` windows folded
   into the batch dim, then re-concatenated along the token-time axis (the
   zero-padded tail tokens are trimmed).  That logic now lives in
   `sphere_backbone.SpatialStreamMixin`, shared with the GRAM probe.
3. Optional chunking on the mono stream too (`MonoEncoderSpec.chunk_len`, in
   *samples* since SPEAR consumes the raw waveform). Unlike GRAM-T, the
   Zipformer uses relative/convolutional positioning and accepts a `wav_len`,
   so it does not hard-crash on long inputs -- chunk_len therefore defaults
   to None (whole sequence in one pass). Set it to SPEAR's pretraining crop
   length (in samples) if you observe long-context degradation.
4. `spear_mono_spec()` / `build_spear_sphere_seld()` helpers so wiring the
   fusion up is a one-liner. Note that unlike GRAM-T, SPEAR does *not* live
   inside the SphereV5 checkpoint, so the builder instantiates (or accepts)
   the HF module itself.
5. Input layout asserted as (B, 7, T, F) with ch0=W logmel, ch1:4=YZX logmel,
   ch4:7=AIV -- the same 7-channel layout `_extract_sphere_features` caches.

SPEAR differs from GRAM-T in two ways the structure accommodates:
  * it has no frequency axis (F_mono = 1), so the mono stream skips the
    attention freq-pool entirely (norm -> time-align -> project);
  * it consumes its own input -- the raw 16 kHz W-channel waveform -- passed
    to forward() as a second argument, rather than being sliced off channel 0
    of the 7-channel sphere features.

Usage
-----
    from spear_seld_fusion import build_spear_sphere_seld
    from sphere.models.sphere_v5 import SphereV5

    sphere = SphereV5.load_from_checkpoint(
        params['sphere_ckpt'], map_location='cpu',
        patch_strategy=patch_strategy,
        strict=False,
    )
    sphere.eval()

    model = build_spear_sphere_seld(data_out, params, sphere,
                                    inject_spatial_tokens="True").to(device)

    out = model(x, spear)   # x: (B,7,T,F) sphere cache, spear: waveform

For the frozen arm the optimizer may still filter on requires_grad, but
prefer `model.parameter_groups(...)`, which puts the encoder in its own group
with its own LR -- required by the Finetune/Scratch arms.

IMPORTANT (feature side):

* Spatial stream: feed the *unnormalized* sphere feature cache
  (`get_sphere_feat_dir()`), NOT the StandardScaler-normalized baseline
  features. `_extract_sphere_features` must apply the per-clip W-channel RMS
  normalization before the mel front end (see sphere_seld_fusion.py header).
  Cache layout is (T, 7*F), channel-major:

      x = rearrange(feat, 't (c f) -> c t f', c=7)   # batch -> (B,7,T,F)

* Mono stream: `SpearFeatureExtractor` caches the resampled 16 kHz W-channel
  on the feature-frame grid, on-disk shape (1, samples_per_frame, T). The
  data generator rides these rows through the circular buffer and flattens
  (T, spf) back into a contiguous waveform per sequence. forward() accepts
  either the flattened (B, n_samples) or the un-flattened (B, T, spf) layout
  and flattens the latter itself (row-major, so sample order is preserved).
  The cached waveform is *not* gain-normalized; if SPEAR pretraining applied
  per-clip normalization, mirror it in SpearFeatureExtractor, not here.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel

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
    model: nn.Module                 # frozen mono backbone
    embed_dim: int                   # D_mono (per-token dim)
    n_freq: int                      # F_mono (1 for waveform/mel-frame encoders)
    chunk_len: Optional[int] = None  # units of the encoder's input time axis
                                     # (SAMPLES for SPEAR); None = whole seq.
                                     # Only supported for input_layout="wave".
    # "wave"   : forward gets (B, N) waveform; (B, T, spf) buffers flattened
    # "frames" : forward gets the mono input untouched (e.g. Dasheng mel
    #            (B, 64, T) straight off the generator)
    input_layout: str = "wave"
    # (model, mono_input) -> (B, T_mono, F_mono*D_mono), F-major flatten
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] = \
        field(default=lambda m, w: m(w))

    def __post_init__(self):
        assert self.input_layout in ("wave", "frames")
        assert self.chunk_len is None or self.input_layout == "wave", \
            "chunking is only implemented for waveform-input encoders"


# =============================================================================
# Injectable mono-encoder: SPEAR (frozen, F_mono = 1)
#   Zipformer backbone (93M params), 512-d embeddings at ~50 Hz.
#   Consumes the *raw 16 kHz waveform* (B, n_samples) and exposes features
#   under outputs["encoder_out"].
# =============================================================================

class SpearMono(nn.Module):
    """Frozen SPEAR backbone returning per-frame embeddings (B, T_mono, D)."""

    SR = 16000

    def __init__(self, hf_id="marcoyang/spear-base-speech-audio-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        self.embed_dim = 512                            # 512 (base)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()                               # always frozen
        return self

    @torch.no_grad()
    def forward(self, wav, wav_len=None):               # wav -> (B, T_mono, D)
        if wav_len is None:                             # fixed-length clips
            wav_len = wav.new_full((wav.shape[0],), wav.shape[-1],
                                   dtype=torch.long)
        return self.model(wav, wav_len)["encoder_out"]  # (B, T_mono, D)


def spear_mono_spec(spear_module: SpearMono, name: str = "spear",
                    chunk_len: Optional[int] = None) -> MonoEncoderSpec:
    """SPEAR wrapped as a MonoEncoderSpec (no frequency axis -> n_freq = 1).

    chunk_len is in SAMPLES (e.g. 10 * 16000 for 10-s windows). Leave None to
    run the whole sequence in one Zipformer pass (safe: relative positioning,
    no fixed pos-embed grid)."""
    return MonoEncoderSpec(
        name=name,
        model=spear_module,
        embed_dim=spear_module.embed_dim,               # D_mono
        n_freq=1,                                       # F_mono
        chunk_len=chunk_len,
        forward_fn=lambda m, wav: m(wav),               # (B, T_mono, D_mono)
    )


# =============================================================================
# Injectable mono-encoder: Dasheng (frozen, F_mono = 1)
#   ViT-style audio encoder, 768-d (base) per-frame embeddings. Consumes its
#   own cached mel front-end (B, 64, T) at 100 fps, cached by
#   DashengFeatureExtractor in cls_feature_class.py.
# =============================================================================

class DashengMono(nn.Module):
    """Frozen Dasheng backbone returning per-frame embeddings (B, T_mono, D)."""

    def __init__(self, hf_id="mispeech/dasheng-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            hf_id, outputdim=None, trust_remote_code=True)
        self.embed_dim = 768                            # 768 (base)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()                               # always frozen
        return self

    @torch.no_grad()
    def forward(self, mel):                             # (B, 64, T) -> (B, T_mono, D)
        return self.model(input_values=mel).hidden_states


def dasheng_mono_spec(dasheng_module: DashengMono,
                      name: str = "dasheng") -> MonoEncoderSpec:
    """Dasheng wrapped as a MonoEncoderSpec (no frequency axis -> n_freq = 1).

    input_layout="frames": the generator's (B, 64, T) mel batch is handed to
    the encoder untouched (no waveform flatten, no chunking -- Dasheng handles
    the 2-s sequences whole)."""
    return MonoEncoderSpec(
        name=name,
        model=dasheng_module,
        embed_dim=dasheng_module.embed_dim,             # D_mono
        n_freq=1,                                       # F_mono
        chunk_len=None,
        input_layout="frames",
        forward_fn=lambda m, mel: m(mel),               # (B, T_mono, D_mono)
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

    def forward(self, x):                               # (B, T_seld, in_dim)
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

class SpearSphereSELD(SpatialStreamMixin, nn.Module):
    """SELD probe fusing a mono encoder (SPEAR / Dasheng) + SphereV5 (spatial).

    inject_spatial_tokens:
      "True"     -> frozen pre-trained SphereV5 encoder          (ours)
      "Finetune" -> pre-trained SphereV5, encoder weights update (ours, ft)
      "Scratch"  -> same architecture, random init, trained      (no pretrain)
      "False"    -> mono stream only                             (baseline ii)
      "Learn"    -> trainable 2-Conv2D spatial front-end         (baseline iii)

    NOTE on "Scratch": only the SPATIAL stream is randomly initialized. The
    mono stream stays pre-trained and frozen, which is the point -- this
    isolates the value of spatial pretraining.

    forward(x, mono):
      x     : (B, 7, T, F) with ch0 = W log-mel, ch1:4 = Y,Z,X log-mel,
              ch4:7 = normalized active-intensity vector, i.e. exactly what
              FeatureClass._extract_sphere_features caches (un-flattened).
              Unused (may be None) when inject_spatial_tokens == "False".
      mono  : the mono encoder's own input -- SPEAR: raw 16 kHz W-channel
              waveform, (B, n_samples) or (B, T, spf) straight off the
              circular buffer; Dasheng: mel batch (B, 64, T).
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
        self.D_mo, self.F_mo = mono_spec.embed_dim, mono_spec.n_freq
        assert self.F_mo == 1, \
            "this probe has no mono freq-pool; use the GRAM probe for F_mono > 1"

        # ---- Spatial stream ----
        self._init_spatial_stream(inject_spatial_tokens, spatial_encoder,
                                  params, proj_dim, dropout, d_sp, f_sp)

        # ---- Mono projection plumbing (no freq-pool, F_mono = 1) ----
        self.input_norm_mo = nn.LayerNorm(self.D_mo)
        self.mono_proj = nn.Sequential(
            nn.LayerNorm(self.D_mo), nn.Linear(self.D_mo, proj_dim),
            nn.GELU(), nn.Dropout(dropout))

        # fuse: concat only when a spatial stream is present
        self.feat_proj = nn.Linear(2 * proj_dim, proj_dim) \
            if self.has_spatial else None

        self.head = SeldHead(proj_dim, out_shape, params)

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        self.mono.model.eval()                          # mono always frozen
        self._sync_spatial_train_mode()                 # see mixin docstring
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_wave(w: torch.Tensor, N_chunk: int):
        """(B, N) -> (B*n, N_chunk); zero-pads the tail.

        Returns (chunked, n_chunks, N_orig)."""
        B, N = w.shape
        n = max(1, math.ceil(N / N_chunk))
        pad = n * N_chunk - N
        if pad:
            w = F.pad(w, (0, pad))
        return w.view(B * n, N_chunk), n, N

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
    def _mono_tokens(self, mono_in: torch.Tensor) -> torch.Tensor:
        """mono_in -> (B, T_mono, D_mo).

        input_layout="frames" (Dasheng): the input -- e.g. a (B, 64, T) mel
        batch -- is passed to the encoder untouched.
        input_layout="wave" (SPEAR): accepts (B, n_samples) or the (B, T, spf)
        circular-buffer layout (flattened row-major), optionally running the
        encoder in fixed-length sample windows (chunk_len) and trimming
        tokens that cover only zero padding."""
        if self.mono.input_layout == "frames":
            with torch.no_grad():
                self.mono.model.eval()
                return self.mono.forward_fn(self.mono.model, mono_in.float())

        wav = mono_in
        if wav.dim() == 3:                              # (B, T, spf) buffer
            wav = wav.reshape(wav.shape[0], -1)         # row-major -> (B, N)
        wav = wav.float()
        with torch.no_grad():
            self.mono.model.eval()
            Nc = self.mono.chunk_len
            if Nc is None or wav.shape[-1] <= Nc:
                return self.mono.forward_fn(self.mono.model, wav)
            B = wav.shape[0]
            wc, n, N = self._chunk_wave(wav, Nc)
            z = self.mono.forward_fn(self.mono.model, wc)   # (B*n, Tz, D)
            Tz = z.shape[1]
            z = rearrange(z, "(b n) t d -> b (n t) d", b=B, n=n)
            # drop tokens that cover only zero padding
            samples_per_tok = Nc / Tz
            t_valid = math.ceil(N / samples_per_tok)
            return z[:, :t_valid]

    # ------------------------------------------------------------------
    def forward(self, x, mono):
        """x: (B,7,T,F) ch0=W, ch1:4=YZX, ch4:7=AIV (ignored when mono-only);
        mono: the mono encoder's input -- SPEAR: raw 16 kHz waveform
        (B, n_samples) or (B, T, spf); Dasheng: mel batch (B, 64, T)."""

        # ---- Mono stream (frozen encoder on the W channel) ----
        z_mo = self._mono_tokens(mono)                  # (B, T_mono, D_mo)
        z_mo = self.input_norm_mo(z_mo)
        z_mo = self.mono_proj(self._avg_pool_time(z_mo))    # (B,T_seld,256)

        if not self.has_spatial:                        # baseline (ii)
            return self.head(z_mo)

        assert x is not None and x.dim() == 4 and x.shape[1] == 7, \
            f"expected (B, 7, T, F), got {None if x is None else tuple(x.shape)}"

        # ---- Spatial stream (SphereV5 frozen/ft/scratch, or learnt) ----
        z_sp = self._spatial_tokens(x)
        z_sp = self.input_norm_sp(z_sp)
        z_sp = self._freq_pool(z_sp, self.q_sp, self.k_proj_sp,
                               self.F_sp, self.D_sp)
        z_sp = self.sphere_proj(self._avg_pool_time(z_sp))  # (B,T_seld,256)

        # ---- Fuse ----
        z = self.feat_proj(torch.cat([z_sp, z_mo], dim=-1))  # (B,T_seld,256)
        return self.head(z)


# =============================================================================
# Factory
# =============================================================================

def build_spear_sphere_seld(out_shape, params, sphere=None,
                            spear: Optional[SpearMono] = None,
                            hf_id: str = "marcoyang/spear-base-speech-audio-v2",
                            inject_spatial_tokens: str = "True",
                            spear_chunk_len_s: Optional[float] = None,
                            d_sp: Optional[int] = None,
                            f_sp: Optional[int] = None) -> SpearSphereSELD:
    """One-liner: fuse a frozen SPEAR mono encoder with the SphereV5 spatial
    encoder. Unlike GRAM-T, SPEAR is not stored inside the SphereV5
    checkpoint, so it is downloaded from HF (or passed in via `spear`).

    spear_chunk_len_s: optional window length in seconds for the SPEAR
    forward pass (converted to samples at 16 kHz). None = single pass."""
    if spear is None:
        spear = SpearMono(hf_id)
    chunk = int(round(spear_chunk_len_s * SpearMono.SR)) \
        if spear_chunk_len_s else None
    mono = spear_mono_spec(spear, chunk_len=chunk)
    return SpearSphereSELD(
        out_shape, params, mono,
        spatial_encoder=sphere,
        inject_spatial_tokens=inject_spatial_tokens,
        d_sp=d_sp, f_sp=f_sp,
    )


def build_dasheng_sphere_seld(out_shape, params, sphere=None,
                              dasheng: Optional[DashengMono] = None,
                              hf_id: str = "mispeech/dasheng-base",
                              inject_spatial_tokens: str = "True",
                              d_sp: Optional[int] = None,
                              f_sp: Optional[int] = None) -> SpearSphereSELD:
    """One-liner: fuse a frozen Dasheng mono encoder with the SphereV5 spatial
    encoder. forward(x, mono) takes the generator's (B, 64, T) mel batch as
    the mono input."""
    if dasheng is None:
        dasheng = DashengMono(hf_id)
    mono = dasheng_mono_spec(dasheng)
    return SpearSphereSELD(
        out_shape, params, mono,
        spatial_encoder=sphere,
        inject_spatial_tokens=inject_spatial_tokens,
        d_sp=d_sp, f_sp=f_sp,
    )


# The probe is mono-encoder-agnostic (anything wrapped in a MonoEncoderSpec
# with F_mono == 1); alias for readability at call sites.
MonoSphereSELD = SpearSphereSELD