# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='/projects/0/prjs1261/seld/TAU2021/2_1_dev_split0_accdoa_foa_model.h5',

        # INPUT PATH
        dataset_dir='/projects/0/prjs1261/seld/TAU2021/',

        # OUTPUT PATHS
        feat_label_dir='/projects/0/prjs1261/seld/TAU2021/TAU2021_labels_gram',

        model_dir='/projects/0/prjs1261/seld/TAU2021/TAU2021_saved_models_gram',            # Dumps the trained models and training curves in this folder
        dcase_output_dir='/projects/0/prjs1261/seld/TAU2021/TAU2021_results_gram',    # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=32000,
        hop_len_s=0.01,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=128,

        # We do not use salsalite
        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=60,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer
        temporal_mode = "none",
        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,              # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,

        # ---- two-stream probe ----
        # inject_spatial_tokens:
        #   'True'      frozen pre-trained Bearings/SphereV5 encoder   (ours)
        #   'Finetune'  pre-trained encoder, weights updated           (ours, ft)
        #   'Scratch'   same architecture, random init, trained        (control)
        #   'Learn'     trainable 2-Conv2D front-end                   (baseline iii)
        #   'False'     mono stream only                               (baseline ii)
        mono_encoder='spear-base',
        inject_spatial_tokens='True',

        learnt_token_dim=384,
        learnt_n_freq=8,
        sphere_ckpt="/gpfs/work5/0/prjs1261/experiments/sphere/Abl=no-gramt/step=50000.ckpt",
        use_sphere_feat=True,
        mono_ckpts=dict(
            gram='labhamlet/gramt-mono',
            spear_base='marcoyang/spear-base-speech-audio-v2',
            spear_large='marcoyang/spear-large-speech-audio-v2',
            dasheng='mispeech/dasheng-base',
        ),

        sphere_lr=1e-4,
        sphere_weight_decay=0.01,
        sphere_freeze_epochs=0,
        sphere_grad_checkpoint=False,
        weight_decay=0.0,
        grad_clip=1.0,
        sphere_fshape=16,
        sphere_tshape=8,
        sphere_n_mels=128,
        sphere_target_length=200,
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '21':                         # "full"  = OURS (frozen probe)
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'True'
        params['condition'] = 'full'
        params['multi_accdoa'] = False

    elif argv == '22':                         # "ca_only" = baseline (ii) learnt conv tokens
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'Learn'
        params['condition'] = 'ca_only' #0.72 in epoch etc
        params['multi_accdoa'] = False

    elif argv == '23':                         # "gram_only" = baseline (iii) mono-only
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'False'
        params['condition'] = 'gram_only'
        params['multi_accdoa'] = False

    elif argv == '24':                         # "bare" = baseline (iv) SELDNet from scratch
        params['dataset'] = 'foa'
        params['inject_spatial_tokens'] = 'None'   # signals: build SeldModel, no encoders
        params['condition'] = 'bare'
        params['multi_accdoa'] = False

    elif argv == '25':                         # "full_ft" = OURS, fine-tuned
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'Finetune'
        params['condition'] = 'full_ft'
        params['multi_accdoa'] = False

    elif argv == '26':                         # "scratch" = architecture control,
        params['dataset'] = 'foa'              # no spatial pretraining
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'Scratch'
        params['condition'] = 'scratch'
        params['multi_accdoa'] = False
        params['sphere_freeze_epochs'] = 0     # nothing to preserve
        params['sphere_lr'] = params['lr']     # random init trains at full LR

    elif argv == '27':                         # "random_frozen" = Table 'Random, frozen':
        params['dataset'] = 'foa'              # Scratch init, but the encoder never trains.
        params['mono_encoder'] = 'gram'        # freeze_epochs > nb_epochs keeps
        params['inject_spatial_tokens'] = 'Scratch'   # set_backbone_trainable(False)
        params['condition'] = 'random_frozen'  # in effect for the whole run.
        params['multi_accdoa'] = False
        params['sphere_freeze_epochs'] = 10**9   # never unfrozen
        # sphere_lr is irrelevant: frozen params produce no grads, AdamW skips them.

    # --- mono-encoder sweep, all with OURS (inject='True') ---
    elif argv == '31':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-base'
        params['inject_spatial_tokens'] = 'True'
        params['multi_accdoa'] = False

    elif argv == '32':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-large'
        params['inject_spatial_tokens'] = 'True'
        params['multi_accdoa'] = False

    elif argv == '33':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'dasheng'
        params['inject_spatial_tokens'] = 'True'
        params['multi_accdoa'] = False

    # --- fine-tuned / from-scratch spatial stream, SPEAR mono ---
    elif argv == '34':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-base'
        params['inject_spatial_tokens'] = 'Finetune'
        params['condition'] = 'full_ft'
        params['multi_accdoa'] = False

    elif argv == '35':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-base'
        params['inject_spatial_tokens'] = 'Scratch'
        params['condition'] = 'scratch'
        params['multi_accdoa'] = False
        params['sphere_freeze_epochs'] = 0
        params['sphere_lr'] = params['lr']

    elif argv == '36':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'dasheng'
        params['inject_spatial_tokens'] = 'Finetune'
        params['condition'] = 'full_ft'
        params['multi_accdoa'] = False

    elif argv == '37':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'dasheng'
        params['inject_spatial_tokens'] = 'Scratch'
        params['condition'] = 'scratch'
        params['multi_accdoa'] = False
        params['sphere_freeze_epochs'] = 0
        params['sphere_lr'] = params['lr']

    # --- same sweep, mono-only (to isolate each backbone's own ceiling) ---
    elif argv == '41':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-base'
        params['inject_spatial_tokens'] = 'False'
        params['multi_accdoa'] = False

    elif argv == '42':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-large'
        params['inject_spatial_tokens'] = 'False'
        params['multi_accdoa'] = False

    elif argv == '43':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'dasheng'
        params['inject_spatial_tokens'] = 'False'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '51':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'True'
        params['condition'] = 'full'
        params['multi_accdoa'] = False
        params["sphere_ckpt"] = "/projects/0/prjs1261/experiments/sphere/full_model/step=50000.ckpt"

    elif argv == '52': 
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'True'
        params['condition'] = 'full'
        params['multi_accdoa'] = False
        params["sphere_ckpt"] = "/gpfs/work5/0/prjs1261/experiments/sphere/Abl=no-gramt/step=50000.ckpt"

    elif argv == '53': 
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'True'
        params['condition'] = 'full'
        params['multi_accdoa'] = False
        params["sphere_ckpt"] = "/gpfs/work5/0/prjs1261/experiments/sphere/Abl=no-psi/Gram=ctx-full/Loss=q1.0-psi0.0/Grid=256-k40/Rot=so3-p1.0/MaskR=0.8/LR=0.0002/BS=256x2/Seed=1234/step=50000.ckpt"

    elif argv == '54':  
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'True'
        params['condition'] = 'full'
        params['multi_accdoa'] = False
        params["sphere_ckpt"] = "/gpfs/work5/0/prjs1261/experiments/sphere/Abl=no-q/Gram=ctx-full/Loss=q0.0-psi1.0/Grid=256-k40/Rot=so3-p1.0/MaskR=0.8/LR=0.0002/BS=256x2/Seed=1234/step=50000.ckpt"

    elif argv == '55':  
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'True'
        params['condition'] = 'full'
        params['multi_accdoa'] = False
        params["sphere_ckpt"] = "/gpfs/work5/0/prjs1261/experiments/sphere/Abl=grid512/Gram=ctx-full/Loss=q1.0-psi1.0/Grid=512-k81.49/Rot=so3-p1.0/MaskR=0.8/LR=0.0002/BS=256x2/Seed=1234/step=50000.ckpt"

    elif argv == '56':                         # '54' architecture, no pretraining
        params['dataset'] = 'foa'              # (ckpt read for hparams only)
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'Scratch'
        params['condition'] = 'scratch'
        params['multi_accdoa'] = False
        params['sphere_freeze_epochs'] = 0
        params['sphere_lr'] = params['lr']
        params["sphere_ckpt"] = "/projects/0/prjs1261/experiments/sphere/full_model/step=100000.ckpt"

    elif argv == '57':                         # '54' architecture, random & frozen
        params['dataset'] = 'foa'              # Table 'Random, frozen' for the
        params['mono_encoder'] = 'gram'        # main checkpoint's geometry
        params['inject_spatial_tokens'] = 'Scratch'
        params['condition'] = 'random_frozen'
        params['multi_accdoa'] = False
        params['sphere_freeze_epochs'] = 10**9   # never unfrozen
        params["sphere_ckpt"] = "/projects/0/prjs1261/experiments/sphere/full_model/step=100000.ckpt"

    elif argv == '121':                     # 121 with hand-crafted spatial stream
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'True'
        params['condition'] = 'full'
        params['multi_accdoa'] = False
        params['inject_spatial_tokens'] = 'Handcrafted'
    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached
    params['feature_sequence_length'] = params['label_sequence_length'] * 10

    # feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    # params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    # params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13


    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params