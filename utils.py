def get_identity_from_cfg(cfg):
    # ---- derive the ablation arm from config state (no extra keys needed) ----
    gram = ("none" if cfg.model.gramt_model_id is None
            else "ctx-masked" if cfg.model.gramt_mask_context
            else "ctx-full")
    if cfg.model.gramt_model_id is None:
        arm = "no-gramt"
    elif cfg.loss.q == 0.0:
        arm = "no-q"
    elif cfg.loss.diffuseness == 0.0:
        arm = "no-psi"
    elif cfg.route_a.n_grid != 256:
        arm = f"grid{cfg.route_a.n_grid}"
    else:
        arm = "full"

    parts = [
        f"Abl={arm}",
        f"Gram={gram}",
        f"Loss=q{cfg.loss.q}-psi{cfg.loss.diffuseness}",
        f"Grid={cfg.route_a.n_grid}-k{cfg.route_a.vmf_kappa:g}",
        f"Rot={cfg.augmentation.rotation_mode}-p{cfg.augmentation.rotation_prob}",
        f"MaskR={cfg.masking.ratio}",
        f"LR={cfg.optimizer.lr}_BS={cfg.trainer.batch_size}x{cfg.trainer.samples_per_clip}",
        f"Seed={cfg.seed}",
    ]
    return "_".join(parts)      # '/'-tree via your existing replace('_','/')