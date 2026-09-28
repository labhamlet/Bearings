"""Shared spatial-stream plumbing for the SELD probes.

Both `sphere_seld_fusion.SphereV5SELD` (GRAM-T mono) and
`spear_seld_fusion.SpearSphereSELD` (SPEAR / Dasheng mono) used to carry
byte-identical copies of the spatial half: the `_chunk_time` folding, the
`pass_through_encoder` call, the CLS drop, the freq-major regroup and the
tail trim.  That code now lives here once, together with the freeze /
unfreeze machinery the fine-tuned and from-scratch arms need.

inject_spatial_tokens:
  "True"        frozen pre-trained SphereV5 encoder, probe only   (ours)
  "Finetune"    pre-trained SphereV5, encoder weights updated     (ours, ft)
  "Scratch"     same architecture, random init, trained           (no pretraining)
  "Learn"       trainable 2-Conv2D front-end                      (baseline iii)
  "Handcrafted" parameter-free framewise AIV statistics           (baseline iv)
  "False"       mono stream only                                  (baseline ii)

"Scratch" and "Finetune" are one code path here: the difference lives in
_make_sphere() in train_seldnet.py, which either loads the checkpoint or
constructs the module fresh from the checkpoint's architecture hparams.
They stay distinct strings because the LR groups, the unfreeze schedule and
the run naming all key off them.

"Handcrafted" replaces the spatial token producer with fixed statistics of
the cached AIV channels (raw + 50/350 ms smoothed AIVs and their two-scale
magnitudes, i.e. proxies of the pre-training targets' inputs).  Everything
downstream -- input_norm_sp, freq pooling, sphere_proj, fusion, head -- is
byte-identical to the other arms, so the arm's only spatial-side trainable
parameters are the same ~3.5k of plumbing every arm trains.  It joins
neither USES_SPHERE nor TRAINS_SPHERE: no encoder is registered, the
checkpoint is only ever read for the mono-spec token geometry.
"""

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

SPATIAL_MODES = ('True', 'Finetune', 'Scratch', 'Learn', 'False', 'Handcrafted')
USES_SPHERE = ("True", "Finetune", "Scratch")
TRAINS_SPHERE = ("Finetune", "Scratch")

# Parameters on the pass_through_encoder path, and nothing else.
# Deliberately excluded:
#   pos_embed        fixed 2-D sincos, constructed with requires_grad=False
#   decoder.*        pretraining heads; no gradient reaches them here
#   gram.*           the frozen mono conditioner -- in the GRAM arm this is
#                    the SAME module object as the probe's mono stream, so
#                    unfreezing sphere.parameters() wholesale would silently
#                    start training the mono encoder and void the comparison
#   gramt_proj.*, gramt_null_token, route_a.*, melspec.*
SPHERE_ENCODER_PREFIXES = (
    "patch_embed.",
    "cls_token",
    "encoder_blocks.",
    "encoder_norm.",
)


def sphere_encoder_parameters(sphere):
    """(name, param) pairs for the encoder proper."""
    for name, p in sphere.named_parameters():
        if name.startswith(SPHERE_ENCODER_PREFIXES):
            yield name, p


def set_sphere_trainable(sphere, flag: bool):
    """Freeze everything, then re-enable only the encoder.  The GRAM
    conditioner is force-frozen last, unconditionally."""
    for p in sphere.parameters():
        p.requires_grad_(False)
    if flag:
        for _, p in sphere_encoder_parameters(sphere):
            p.requires_grad_(True)
    gram = getattr(sphere, "gram", None)
    if gram is not None:
        for p in gram.parameters():
            p.requires_grad_(False)
        gram.eval()
    return sphere


# =============================================================================
# Learnt spatial front-end baseline (iii)
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class LearntSpatialFrontEnd(nn.Module):
    """Two Conv2D layers on 7-ch FOA -> (B, T, F_sp*D_sp), matching the
    frozen spatial stream's token layout exactly."""

    def __init__(self, in_ch, D_sp, F_sp, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or D_sp
        self.conv1 = ConvBlock(in_ch, mid_ch)
        self.conv2 = ConvBlock(mid_ch, D_sp)
        self.F_sp, self.D_sp = F_sp, D_sp

    def forward(self, x):                              # (B, 7, T, F)
        x = self.conv2(self.conv1(x))                  # (B, D_sp, T, F)
        B, D, T, _ = x.shape
        x = F.adaptive_avg_pool2d(x, (T, self.F_sp))   # (B, D_sp, T, F_sp)
        x = x.permute(0, 2, 3, 1).contiguous()         # (B, T, F_sp, D_sp)
        return x.reshape(B, T, self.F_sp * self.D_sp)


# =============================================================================
# Hand-crafted spatial front-end baseline (iv)
# =============================================================================

class HandcraftedSpatialFrontEnd(nn.Module):
    """Zero-parameter spatial tokens from the cached 7-ch features.

    Per (frame, mel-band), D_hc = 12:

        raw AIV                  (3)  cached channels 4:7, untouched
        50 ms smoothed AIV       (3)  uniform moving average, tau_s of Eq. (5)
        350 ms smoothed AIV      (3)  tau_l of Eq. (5)
        ||smoothed short||       (1)  ~ directionality = 1 - Psi_short
        ||smoothed long||        (1)  ~ 1 - Psi_long
        short - long magnitude   (1)  multi-scale diffuseness contrast

    The cached AIV is per-band energy-normalized (Eq. 2), so the smoothed
    magnitudes are the standard DirAC-style proxy of Eq. (5) diffuseness,
    not Eq. (5) verbatim (that would need raw intensity and energy averaged
    separately).  AIVs are gain-invariant, so the W-RMS clip normalization
    needs no special handling.

    Input  x: (B, 7, T, F) -- ch0 = W logmel, ch1:4 = YZX logmel, ch4:7 = AIV.
    Output  : (B, T, F * D_hc), F-major flatten (index = f * D + d), i.e.
              exactly what the hosts' _freq_pool expects via
              .view(B, T, F_sp, D_sp).

    No positional grid -> no chunking needed; works at arbitrary T (so it is
    seq_len-agnostic: 20- and 60-frame probes see the same features).  The
    module has no parameters and no buffers with train-mode behaviour, so it
    needs no case in _sync_spatial_train_mode().
    """

    D_hc = 12

    def __init__(self, short_frames: int = 5, long_frames: int = 35):
        super().__init__()
        assert short_frames % 2 == 1 and long_frames % 2 == 1, \
            "use odd window lengths so the moving average is length-preserving"
        self.short_frames = short_frames
        self.long_frames = long_frames

    @staticmethod
    def _smooth(flat: torch.Tensor, k: int) -> torch.Tensor:
        # flat: (B, 3*F, T) -> same shape; uniform k-frame moving average,
        # edge-corrected (padding excluded from the mean).
        return F.avg_pool1d(flat, kernel_size=k, stride=1,
                            padding=k // 2, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 7, T, F)
        aiv = x[:, 4:7]                                   # (B, 3, T, F)
        flat = rearrange(aiv, 'b c t f -> b (c f) t')
        i_s = rearrange(self._smooth(flat, self.short_frames),
                        'b (c f) t -> b c t f', c=3)
        i_l = rearrange(self._smooth(flat, self.long_frames),
                        'b (c f) t -> b c t f', c=3)
        m_s = i_s.norm(dim=1, keepdim=True)               # (B, 1, T, F)
        m_l = i_l.norm(dim=1, keepdim=True)
        feats = torch.cat([aiv, i_s, i_l, m_s, m_l, m_s - m_l], dim=1)
        return rearrange(feats, 'b d t f -> b t (f d)')   # F-major, like mono


# =============================================================================
# Mixin
# =============================================================================

class SpatialStreamMixin:
    """Spatial half of the SELD probes.

    The host class must be an nn.Module and must call
    `_init_spatial_stream()` after `nn.Module.__init__()` has run.  It
    provides `_chunk_time` and `_spatial_tokens`; the host keeps its own
    mono-side helpers.
    """

    def _init_spatial_stream(self, mode, spatial_encoder, params, proj_dim,
                             dropout, d_sp=None, f_sp=None):
        assert mode in SPATIAL_MODES, \
            f"inject_spatial_tokens={mode!r} not in {SPATIAL_MODES}"
        self.inject = mode
        self.has_spatial = mode != "False"
        self.uses_sphere = mode in USES_SPHERE
        self.sphere_frozen = mode == "True"
        self.backbone_trainable = mode in TRAINS_SPHERE
        self.sphere_grad_checkpoint = bool(
            params.get("sphere_grad_checkpoint", False))

        self.sphere = None
        self.learnt_front = None
        self.handcrafted_front = None
        if not self.has_spatial:
            return

        if self.uses_sphere:
            assert spatial_encoder is not None, \
                f"inject_spatial_tokens={mode!r} needs a SphereV5 encoder"
            self.sphere = spatial_encoder
            self.D_sp = spatial_encoder.encoder_embedding_dim
            self.F_sp = spatial_encoder.p_f_dim
            set_sphere_trainable(self.sphere, self.backbone_trainable)
        elif mode == "Handcrafted":
            # Tokens live on the raw mel grid, NOT the patch grid: F_sp is
            # the mel-band count, and the hosts' freq-attention pool absorbs
            # the different F_sp transparently.  The sphere module (when
            # given) is only read for that number -- it is NOT registered.
            self.handcrafted_front = HandcraftedSpatialFrontEnd(
                short_frames=int(params.get('hc_short_frames', 5)),
                long_frames=int(params.get('hc_long_frames', 35)))
            self.D_sp = HandcraftedSpatialFrontEnd.D_hc
            if spatial_encoder is not None:
                self.F_sp = int(spatial_encoder.num_mel_bins)
            else:
                assert f_sp is not None or 'nb_mel_bins' in params, \
                    "'Handcrafted' without an encoder needs f_sp or " \
                    "params['nb_mel_bins']"
                self.F_sp = int(f_sp if f_sp is not None
                                else params['nb_mel_bins'])
        else:                                   # "Learn"
            if spatial_encoder is not None:
                self.D_sp = spatial_encoder.encoder_embedding_dim
                self.F_sp = spatial_encoder.p_f_dim
            else:
                assert d_sp is not None and f_sp is not None, \
                    "'Learn' without an encoder needs d_sp and f_sp"
                self.D_sp, self.F_sp = d_sp, f_sp
            self.learnt_front = LearntSpatialFrontEnd(
                in_ch=7, D_sp=self.D_sp, F_sp=self.F_sp,
                mid_ch=params.get("nb_cnn2d_filt", None))

        self.input_norm_sp = nn.LayerNorm(self.F_sp * self.D_sp)
        self.q_sp = nn.Parameter(torch.randn(self.D_sp) * 0.02)
        self.k_proj_sp = nn.Linear(self.D_sp, self.D_sp, bias=False)
        self.sphere_proj = nn.Sequential(
            nn.LayerNorm(self.D_sp), nn.Linear(self.D_sp, proj_dim),
            nn.GELU(), nn.Dropout(dropout))

    # ------------------------------------------------------------------
    def set_backbone_trainable(self, flag: bool) -> bool:
        """Toggle encoder gradients (used for the freeze-then-unfreeze
        schedule).  No-op for the frozen / Learn / Handcrafted / mono-only
        arms.

        The backbone params stay in the optimizer's param group either way:
        a frozen param produces no .grad, and torch optimizers skip those,
        so nothing is updated and the probe's Adam state is preserved."""
        if self.sphere is None or self.sphere_frozen:
            return False
        self.backbone_trainable = bool(flag)
        set_sphere_trainable(self.sphere, self.backbone_trainable)
        return self.backbone_trainable

    def _sync_spatial_train_mode(self):
        """Called from train().  The encoder is held in eval() even while
        fine-tuning: nothing on the pass_through_encoder path is train-mode
        dependent (encoder_dropout p=0.0, timm Blocks built without
        drop-path), and eval() guarantees SphereV5's pretraining EMA buffers
        (leveldiff/level stats, wp_log_ref) can never be updated by a probe
        run.  Module mode does not affect gradient flow.
        Revisit this if drop-path is ever enabled in the encoder blocks.
        (The Handcrafted front-end is stateless: nothing to sync.)"""
        if self.sphere is not None:
            self.sphere.eval()

    def parameter_groups(self, lr, backbone_lr=None, weight_decay=0.0,
                         backbone_weight_decay=None):
        """Two groups so the pre-trained encoder can run at its own LR.

        For the "Handcrafted" arm self.sphere is None, so the backbone group
        is empty by construction and the probe group is mono-only plumbing
        plus ~3.5k spatial-side params (LN + 12-dim query/key + 12->256
        projection) -- the train script's group printout is the check."""
        if self.sphere is not None and not self.sphere_frozen:
            backbone = [p for _, p in sphere_encoder_parameters(self.sphere)]
        else:
            backbone = []
        backbone_ids = {id(p) for p in backbone}
        probe = [p for p in self.parameters()
                 if p.requires_grad and id(p) not in backbone_ids]

        groups = [{"name": "probe", "params": probe, "lr": lr,
                   "weight_decay": weight_decay}]
        if backbone:
            groups.append({
                "name": "sphere",
                "params": backbone,
                "lr": backbone_lr if backbone_lr is not None else 0.1 * lr,
                "weight_decay": (weight_decay if backbone_weight_decay is None
                                 else backbone_weight_decay),
            })
        return groups

    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_time(x: torch.Tensor, T_chunk: int):
        """(B, C, T, F) -> (B*n, C, T_chunk, F); zero-pads the tail.

        Returns (chunked, n_chunks, T_orig)."""
        B, C, T, Fm = x.shape
        n = max(1, math.ceil(T / T_chunk))
        pad = n * T_chunk - T
        if pad:
            x = F.pad(x, (0, 0, 0, pad))                # pad time dim at end
        x = x.view(B, C, n, T_chunk, Fm).permute(0, 2, 1, 3, 4)
        return x.reshape(B * n, C, T_chunk, Fm), n, T

    def _run_sphere(self, logmel4, aiv):
        fn = self.sphere.pass_through_encoder
        if (self.backbone_trainable and self.sphere_grad_checkpoint
                and torch.is_grad_enabled()):
            # One recompute of the whole encoder; the chunked batch (B*n)
            # is what makes activations expensive on long sequences.
            return checkpoint(fn, logmel4, aiv, use_reentrant=False)
        return fn(logmel4, aiv)

    def _spatial_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 7, T, F) -> (B, T_sp, F_sp*D_sp)."""
        if self.inject == "Handcrafted":
            return self.handcrafted_front(x.float())    # stateless, any T
        if self.inject == "Learn":
            return self.learnt_front(x)                 # trainable front-end

        train_backbone = self.backbone_trainable and torch.is_grad_enabled()
        ctx = contextlib.nullcontext() if train_backbone else torch.no_grad()
        with ctx:
            self._sync_spatial_train_mode()
            B = x.shape[0]
            Tc = self.sphere.target_length
            xc, n, T = self._chunk_time(x.float(), Tc)
            z = self._run_sphere(xc[:, 0:4], xc[:, 4:7])   # (B*n, 1+P, D)
            z = z[:, 1:, :]                                # drop CLS
            z = rearrange(z, "(b n) (f t) d -> b (n t) (f d)",
                          b=B, n=n, f=self.F_sp)
            t_valid = math.ceil(T / self.sphere.tshape)
            z = z[:, :t_valid]
        return z