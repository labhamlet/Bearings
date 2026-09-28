import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import parameters
import time
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates

import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")


from sphere_backbone import SPATIAL_MODES, TRAINS_SPHERE, USES_SPHERE
from sphere_seld_fusion import build_sphere_seld
from spear_seld_fusion import (build_spear_sphere_seld, build_dasheng_sphere_seld,
                                 SpearMono, DashengMono)

import random

def _set_seed(seed: int):
    """The probe head, the freq-pool queries and (in the Scratch arm) the
    whole spatial encoder are randomly initialized, and the generator
    shuffles. cuDNN autotuning still makes runs non-bitwise-identical, but
    init and data order dominate the spread."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pop_seed_arg(argv):
    """Strip an optional '--seed N' (or '--seed=N') from the CLI.

    Returns (argv_without_seed, seed_or_None).  Keeping the seed out of the
    positional slots means the existing `task_id [job_id]` call style is
    untouched, e.g.:

        python3 train_seldnet.py 21 --seed 1
        python3 train_seldnet.py 21 --seed=2
    """
    argv = list(argv)
    seed = None
    for i, a in enumerate(argv):
        if a == '--seed':
            if i + 1 >= len(argv):
                raise ValueError("'--seed' expects an integer value")
            seed = int(argv[i + 1])
            del argv[i:i + 2]
            break
        if a.startswith('--seed='):
            seed = int(a.split('=', 1)[1])
            del argv[i]
            break
    return argv, seed


def _mono_kind(params):
    """'gram' | 'spear' | 'dasheng' -- which mono encoder this run uses.

    Raises on anything else (e.g. leftover 'atst' configs) instead of
    silently falling through to GRAM."""
    enc = str(params.get('mono_encoder', '')) or 'gram'
    if enc.startswith('spear'):
        return 'spear'
    if enc.startswith('dasheng'):
        return 'dasheng'
    if enc.startswith('gram'):
        return 'gram'
    raise ValueError(
        f"Unsupported mono_encoder '{enc}': expected 'gram', "
        f"'spear-base'/'spear-large', or 'dasheng'")


def _inject_mode(params):
    """The spatial-stream arm, validated once up front.

    'True' | 'Finetune' | 'Scratch' | 'Learn' | 'False'."""
    mode = str(params.get('inject_spatial_tokens', 'True'))
    if mode not in SPATIAL_MODES:
        raise ValueError(
            f"inject_spatial_tokens={mode!r} is not one of {SPATIAL_MODES}. "
            "(The 'None' bare-SELDNet arm is not wired up in this script.)")
    return mode


def _uses_second_stream(params):
    """True when the data generator yields (feat, mono, label) 3-tuples,
    i.e. the mono encoder consumes its own cached input (SPEAR waveform /
    Dasheng mel) instead of channel 0 of the sphere features (GRAM)."""
    return _mono_kind(params) in ('spear', 'dasheng')


def _load_mono_encoder(params, device):
    """Load the frozen mono backbone once (HF download / cache hit).
    Returns None for GRAM, which lives inside the SphereV5 checkpoint (or is
    built from the hub by mono_spec_from_sphere for unconditioned/Scratch
    spheres)."""
    kind = _mono_kind(params)
    if kind == 'spear':
        hf_key = params['mono_encoder'].replace('-', '_')   # 'spear-base' -> 'spear_base'
        return SpearMono(params['mono_ckpts'][hf_key]).to(device)
    if kind == 'dasheng':
        return DashengMono(
            params['mono_ckpts'].get('dasheng', 'mispeech/dasheng-base')
        ).to(device)
    return None


def _build_model(data_out, params, sphere, mono_module, device):
    """Construct the probe matching params['mono_encoder'] /
    params['inject_spatial_tokens']."""
    inject = _inject_mode(params)
    kind = _mono_kind(params)
    print(kind)
    if kind == 'spear':
        model = build_spear_sphere_seld(
            data_out, params, sphere,
            spear=mono_module,
            inject_spatial_tokens=inject,
        )
    elif kind == 'dasheng':
        model = build_dasheng_sphere_seld(
            data_out, params, sphere,
            dasheng=mono_module,
            inject_spatial_tokens=inject,
        )
    else:   # 'gram' -- the mono encoder is pulled out of the SphereV5 ckpt
        model = build_sphere_seld(
            data_out, params, sphere,
            inject_spatial_tokens=inject,
        )
    return model.to(device)


# =============================================================================
# SphereV5 construction
# =============================================================================

def _sphere_patch_strategy(params):
    """SphereV5 is saved with save_hyperparameters(ignore=['patch_strategy']),
    so the strategy must be reconstructed identically at load time."""
    from src.patching import PatchStrategy
    return PatchStrategy(
        fshape=params.get('sphere_fshape', 16),
        tshape=params.get('sphere_tshape', 8),
        fstride=params.get('sphere_fshape', 16),   # SphereV5 requires stride==shape
        tstride=params.get('sphere_tshape', 8),
        input_fdim=params.get('sphere_n_mels', 128),
        input_tdim=params.get('sphere_target_length', 200),
    )


def _sphere_ckpt_hparams(params):
    """Architecture hparams of the pre-trained checkpoint, so the Scratch arm
    is shape-identical to the pre-trained one (depth, width, heads, patch
    grid, n_grid, ...).  Weights are NOT read."""
    ckpt = torch.load(params['sphere_ckpt'], map_location='cpu',
                      weights_only=False)
    hp = dict(ckpt.get('hyper_parameters', {}) or {})
    del ckpt
    hp.pop('patch_strategy', None)
    hp.pop('kwargs', None)
    return hp


def _make_sphere(params, device, mode):
    """mode in {'True','Finetune','Scratch','Learn','False'}.

    'Scratch' constructs the architecture with random init; every other mode
    that needs a sphere loads the checkpoint.  ('Learn'/'False' still get one
    because the GRAM mono spec reads its token geometry off it.)"""
    from src.model import SphereV5
    ps = _sphere_patch_strategy(params)

    if mode == 'Scratch':
        hp = _sphere_ckpt_hparams(params) if params.get('sphere_ckpt') else {}
        # The decoder conditioner is unused by pass_through_encoder; disabling
        # it avoids pulling a second copy of GRAM-T into memory.  The GRAM mono
        # stream is then built from the hub by mono_spec_from_sphere(), with
        # the same pre-trained weights the conditioned checkpoint carries.
        hp['gramt_model_id'] = None
        sphere = SphereV5(patch_strategy=ps, **hp)
        print('sphere: RANDOM INIT (architecture from {})'.format(
            params.get('sphere_ckpt', '<defaults>')), flush=True)
    else:
        sphere = SphereV5.load_from_checkpoint(
            params['sphere_ckpt'],
            map_location='cpu',
            patch_strategy=ps,
            strict=False,   # gramt_null_token exists only in the mask-ctx arm
        )
        print('sphere: loaded {}'.format(params['sphere_ckpt']), flush=True)

    sphere.eval()
    return sphere.to(device)


# =============================================================================
# ACCDOA helpers (unchanged)
# =============================================================================

def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5

    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 6*nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames
    test_filelist = data_generator.get_filelist()

    use_mono = _uses_second_stream(params)

    nb_test_batches, test_loss = 0, 0.
    nb_classes = params['unique_classes']
    mag_min = np.full(nb_classes, np.inf, dtype=np.float32)
    mag_max = np.full(nb_classes, -np.inf, dtype=np.float32)
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for batch in data_generator.generate():
            # load one batch of data
            if use_mono:
                data, mono, target = batch
                data = torch.tensor(data).to(device).float()
                mono = torch.tensor(mono).to(device).float()
                target = torch.tensor(target).to(device).float()
                output = model(data, mono)
            else:
                data, target = batch
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
                output = model(data)

            loss = criterion(output, target)

            # --- track per-class ACCDOA magnitude range over predictions ---
            out_np = output.detach().cpu().numpy()
            if params['multi_accdoa'] is True:
                mags = []
                for tr in range(3):
                    base = tr * 3 * nb_classes
                    x_t = out_np[..., base               : base +   nb_classes]
                    y_t = out_np[..., base +   nb_classes: base + 2*nb_classes]
                    z_t = out_np[..., base + 2*nb_classes: base + 3*nb_classes]
                    mags.append(np.sqrt(x_t**2 + y_t**2 + z_t**2))  # [B, T, C]
                # stack tracks -> [B, T, num_tracks, C] then reduce all but class axis
                mag = np.stack(mags, axis=-2)
                batch_min = mag.reshape(-1, nb_classes).min(axis=0)
                batch_max = mag.reshape(-1, nb_classes).max(axis=0)
            else:
                x_t = out_np[..., :nb_classes]
                y_t = out_np[..., nb_classes:2*nb_classes]
                z_t = out_np[..., 2*nb_classes:3*nb_classes]
                mag = np.sqrt(x_t**2 + y_t**2 + z_t**2)  # [B, T, C]
                batch_min = mag.reshape(-1, nb_classes).min(axis=0)
                batch_max = mag.reshape(-1, nb_classes).max(axis=0)
            mag_min = np.minimum(mag_min, batch_min)
            mag_max = np.maximum(mag_max, batch_max)
            # ---------------------------------------------------------------

            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(out_np, params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(out_np, params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+params['unique_classes']], doa_pred[frame_cnt][class_cnt+2*params['unique_classes']]])
            data_generator.write_output_format_file(output_file, output_dict)

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break


        test_loss /= nb_test_batches

    return test_loss, mag_min, mag_max


def train_epoch(data_generator, optimizer, model, criterion, params, device):
    nb_train_batches, train_loss = 0, 0.
    use_mono = _uses_second_stream(params)
    grad_clip = params.get('grad_clip', 0.0)
    model.train()
    for batch in data_generator.generate():
        # load one batch of data
        if use_mono:
            data, mono, target = batch
            data = torch.tensor(data).to(device).float()
            mono = torch.tensor(mono).to(device).float()
            target = torch.tensor(target).to(device).float()
        else:
            data, target = batch
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

        optimizer.zero_grad()

        # process the batch of data based on chosen mode
        output = model(data, mono) if use_mono else model(data)

        loss = criterion(output, target)
        loss.backward()
        if grad_clip:
            # Only relevant to the Finetune/Scratch arms, where gradients
            # actually reach the ViT; harmless (and skipped) otherwise.
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.grad is not None],
                grad_clip)
        optimizer.step()

        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break

    train_loss /= nb_train_batches

    return train_loss

def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects up to two optional positional inputs and one option.
        first input:  task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1
        --seed N:     (optional) explicit RNG seed for this run. Overrides the
                      params['seed'] + job_id scheme. The seed is appended to
                      unique_name, so two runs of the same task_id with
                      different seeds never overwrite each other's model /
                      DCASE outputs.

        e.g.  python3 train_seldnet.py 21 --seed 1
    """
    print(argv, flush=True)
    argv, cli_seed = _pop_seed_arg(argv)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # torch.autograd.set_detect_anomaly(True)   # debug only: slows training a lot

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        if '2020' in params['dataset_dir']:
            test_splits = [1]
            val_splits = [2]
            train_splits = [[3]]

        elif '2021' in params['dataset_dir']:
            test_splits = [6]
            val_splits = [5]
            train_splits = [[1,2,3,4]]

        elif '2022' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]]

        elif '2023' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1,2,3]]
        else:
            print('ERROR: Unknown dataset splits', flush=True)
            exit()

    if cli_seed is not None:
        # Explicit --seed wins: used verbatim, so the bash script controls
        # exactly which seeds each comparison runs with (e.g. 1 and 2).
        seed = cli_seed
    else:
        seed = int(params.get('seed', 1234)) + (int(job_id) if str(job_id).isdigit() else 0)
    _set_seed(seed)
    print('seed: {} ({})'.format(
        seed, 'from --seed' if cli_seed is not None else 'params+job_id'),
        flush=True)
    inject = _inject_mode(params)
    trains_backbone = inject in TRAINS_SPHERE
    needs_sphere = True   # even 'Learn'/'False' read token geometry off it

    shared_sphere = None if trains_backbone else _make_sphere(params, device, inject)
    mono_module = _load_mono_encoder(params, device)
    print('mono_encoder: {} ({}), inject_spatial_tokens: {}{}'.format(
        params.get('mono_encoder', 'gram'), _mono_kind(params), inject,
        '  [backbone trainable]' if trains_backbone else ''), flush=True)

    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------', flush=True)
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split), flush=True)
        print('---------------------------------------------------------------------------------------------------', flush=True)

        # Unique name for the run
        loc_feat = params['dataset']
        if params['dataset'] == 'mic':
            if params['use_salsalite']:
                loc_feat = '{}_salsa'.format(params['dataset'])
            else:
                loc_feat = '{}_gcc'.format(params['dataset'])
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_split{}_{}_{}_{}_seed{}'.format(
            task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat,
            inject.lower(), seed
        )
        model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
        print("unique_name: {}\n".format(unique_name), flush=True)

        # Load train and validation data
        print('Loading training dataset:', flush=True)
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        print('Loading validation dataset:', flush=True)
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True
        )

        data_in, data_out = data_gen_train.get_data_sizes()

        # Fresh backbone per split when its weights are trained.
        sphere = _make_sphere(params, device, inject) if trains_backbone \
            else shared_sphere
        model = _build_model(data_out, params, sphere, mono_module, device)

        # Dump results in DCASE output format for calculating final scores
        dcase_output_val_folder = os.path.join(params['dcase_output_dir'], '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        print('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder), flush=True)

        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)

        # start training
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999
        patience_cnt = 0

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']

        # Keep the pre-trained encoder frozen while the randomly-initialized
        # probe settles, otherwise the first epochs' noise gradients wash out
        # the pretrained features at any usable LR.
        freeze_epochs = int(params.get('sphere_freeze_epochs', 0)) \
            if trains_backbone else 0
        if freeze_epochs:
            model.set_backbone_trainable(False)

        param_groups = model.parameter_groups(
            lr=params['lr'],
            backbone_lr=params.get('sphere_lr', 0.1 * params['lr']),
            weight_decay=params.get('weight_decay', 0.0),
            backbone_weight_decay=params.get('sphere_weight_decay', None),
        )
        # AdamW with weight_decay=0.0 is numerically identical to the Adam
        # this script used before.
        optimizer = optim.AdamW(param_groups)
        criterion = nn.MSELoss()

        for g in param_groups:
            print('  {:<7s} {:>3d} tensors  {:>12,d} params  lr={:g}  wd={:g}'.format(
                g['name'], len(g['params']),
                sum(p.numel() for p in g['params']),
                g['lr'], g['weight_decay']), flush=True)
        if freeze_epochs:
            print('  sphere encoder frozen for the first {} epoch(s)'.format(
                freeze_epochs), flush=True)

        for epoch_cnt in range(nb_epoch):
            # Unfreeze the backbone once the probe has warmed up. No optimizer
            # rebuild needed: the backbone params are already in their group,
            # and a param with requires_grad=False yields no .grad, which the
            # optimizer skips.
            if freeze_epochs and epoch_cnt == freeze_epochs:
                model.set_backbone_trainable(True)
                print('  unfroze sphere encoder at epoch {}'.format(epoch_cnt),
                      flush=True)

            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            train_loss = train_epoch(data_gen_train, optimizer, model, criterion, params, device)
            train_time = time.time() - start_time

            # ---------------------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------------------
            start_time = time.time()
            val_loss, val_mag_min, val_mag_max = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)

            # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
            val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_folder)

            val_time = time.time() - start_time

            # Save model if loss is good
            if val_seld_scr <= best_seld_scr:
                best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr
                torch.save(model.state_dict(), model_name)

            # Print stats
            print(
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                # 'train_loss: {:0.2f}, val_loss: {:0.2f}, '
                'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                'ER/F/LE/LR/SELD: {}, '
                'best_val_epoch: {} {}'.format(
                    epoch_cnt, train_time, val_time,
                    train_loss, val_loss,
                    '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr),
                    best_val_epoch, '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE, best_LR, best_seld_scr))
            , flush=True)
            print('  val ACCDOA magnitude per class:', flush=True)
            print('  Class\tmin\tmax', flush=True)
            for cls_cnt in range(params['unique_classes']):
                print('  {}\t{:0.4f}\t{:0.4f}'.format(cls_cnt, val_mag_min[cls_cnt], val_mag_max[cls_cnt]), flush=True)

            if params['average'] == 'macro':
                print('  Classwise val results:', flush=True)
                print('  Class\tER\tF\tLE\tLR\tSELD', flush=True)
                for cls_cnt in range(params['unique_classes']):
                    print('  {}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                        cls_cnt,
                        classwise_val_scr[0][cls_cnt],
                        classwise_val_scr[1][cls_cnt],
                        classwise_val_scr[2][cls_cnt],
                        classwise_val_scr[3][cls_cnt],
                        classwise_val_scr[4][cls_cnt],
                    ), flush=True)
                patience_cnt += 1
                if patience_cnt > params['patience']:
                    break

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        print('Load best model weights', flush=True)
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        print('Loading unseen test dataset:', flush=True)
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
        )

        # Dump results in DCASE output format for calculating final scores
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder), flush=True)

        test_loss, test_mag_min, test_mag_max = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)
        print('Test ACCDOA magnitude per class:', flush=True)
        print('Class\tmin\tmax', flush=True)
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.4f}\t{:0.4f}'.format(cls_cnt, test_mag_min[cls_cnt], test_mag_max[cls_cnt]), flush=True)

        use_jackknife=True
        print("Getting Test Results")
        test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )
        print('\nTest Loss', flush=True)
        print('SELD score (early stopping metric): {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''), flush=True)
        print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(test_ER[0]  if use_jackknife else test_ER, '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0], test_ER[1][1]) if use_jackknife else '', 100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''), flush=True)
        print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '', 100*test_LR[0]  if use_jackknife else 100*test_LR,'[{:0.2f}, {:0.2f}]'.format(100*test_LR[1][0], 100*test_LR[1][1]) if use_jackknife else ''), flush=True)
        if params['average']=='macro':
            print('Classwise results on unseen test data', flush=True)
            print('Class\tER\tF\tLE\tLR\tSELD_score', flush=True)
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                     cls_cnt,
                     classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''), flush=True)



if __name__ == "__main__":
    sys.exit(main(sys.argv))