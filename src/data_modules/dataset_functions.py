import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def pad_or_truncate_1d(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[1]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad1d((0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:, :target_length]
    return feature

def pad_or_truncate(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[1]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad2d((0, 0, 0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:, :target_length, :]
    return feature



def pad_or_truncate_batch(x: torch.Tensor, target_length: int) -> torch.Tensor:
    B, C, T, F_bins = x.shape
    if T == target_length:
        return x
    elif T > target_length:
        return x[:, :, :target_length, :]
    else:
        # F.pad format: (last_dim_left, last_dim_right, second_last_dim_left, second_last_dim_right)
        # We want to pad the T dimension (second to last) on the right by (target_length - T)
        return F.pad(x, (0, 0, 0, target_length - T))



def instance_normalize(feature: torch.Tensor) -> torch.Tensor:
    """
    Normalize a feature tensor using the specified mean and standard deviation.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to normalize.
    mean : float
        The mean value for normalization.
    std : float
        The standard deviation value for normalization.

    Returns
    -------
    torch.Tensor
        A tensor where each element is normalized as:
        `(feature - mean) / (std)`.

    Notes
    -----
    This normalization scales the data to have a mean of 0 and reduces the amplitude
    by the factor of `2 * std`.
    """
    return (feature - feature.mean()) / (feature.std() + 1e-8)


def pre_process_audio_visage(waveform, audio_sr, native_sr):
    """Return exactly five seconds of four-channel native-rate audio."""

    if waveform.ndim != 2 or waveform.shape[0] != 4:
        raise ValueError(
            f"Expected waveform shaped (4, T), got {tuple(waveform.shape)}"
        )

    if audio_sr != native_sr:
        raise ValueError(
            f"Shard sample rate {audio_sr} != expected native rate "
            f"{native_sr}; re-shard or bucket mixed sample rates."
        )

    target_len = native_sr * 5
    current_len = waveform.shape[-1]

    if current_len < target_len:
        waveform = F.pad(
            waveform,
            (0, target_len - current_len),
            mode="constant",
            value=0,
        )

    elif current_len > target_len:
        waveform = waveform[:, :target_len].clone()

    return waveform





