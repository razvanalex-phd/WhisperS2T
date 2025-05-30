import os
import subprocess
import tempfile
import wave
from multiprocessing.dummy import Pool
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from whisper_s2t import BASE_PATH
from whisper_s2t.configs import *

silent_file = f"{BASE_PATH}/assets/silent.mp3"

RESAMPLING_ENGINE = "soxr"

with tempfile.TemporaryDirectory() as tmpdir:
    ffmpeg_install_link = (
        "https://github.com/shashikg/WhisperS2T?tab=readme-ov-file#for-ubuntu"
    )

    try:
        subprocess.check_output(["ffmpeg", "-version"])
    except:
        raise RuntimeError(
            f"Seems 'ffmpeg' is not installed. Please install ffmpeg before using this package!\nCheck: {ffmpeg_install_link}"
        )

    ret_code = os.system(
        f'ffmpeg -hide_banner -loglevel panic -i "{silent_file}" -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar 1600 "{tmpdir}/tmp.wav" -y'
    )

    if ret_code != 0:
        print(f"'ffmpeg' failed with soxr resampler, trying 'swr' resampler.")
        RESAMPLING_ENGINE = "swr"

        ret_code = os.system(
            f'ffmpeg -hide_banner -loglevel panic -i "{silent_file}" -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar 1600 "{tmpdir}/tmp.wav" -y'
        )

        if ret_code != 0:
            raise RuntimeError(
                f"Seems 'ffmpeg' is not installed properly. Please uninstall and install it again.\nCheck: {ffmpeg_install_link}"
            )
        else:
            print(f"Using 'swr' resampler. This may degrade performance.")


def load_audio(
    input_file: str,
    sr: int = 16000,
    return_duration: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """
    Load and preprocess an audio file.

    Args:
        input_file: Path to the audio file
        sr: Target sample rate in Hz
        return_duration: Whether to return the audio duration

    Returns:
        Audio signal as numpy array or tuple of (audio signal, duration in seconds)
    """
    try:
        with wave.open(input_file, "rb") as wf:
            if (wf.getframerate() != sr) or (wf.getnchannels() != 1):
                raise Exception("Not a 16kHz wav mono channel file!")

            frames = wf.getnframes()
            x = wf.readframes(int(frames))
    except:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_file = f"{tmpdir}/tmp.wav"
            ret_code = os.system(
                f'ffmpeg -hide_banner -loglevel panic -i "{input_file}" -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar {sr} "{wav_file}" -y'
            )
            if ret_code != 0:
                raise RuntimeError(
                    "ffmpeg failed to resample the input audio file, make sure ffmpeg is compiled properly!"
                )

            with wave.open(wav_file, "rb") as wf:
                frames = wf.getnframes()
                x = wf.readframes(int(frames))

    audio_signal = np.frombuffer(x, np.int16).flatten().astype(np.float32) / 32768.0
    audio_duration = len(audio_signal) / sr

    if return_duration:
        return audio_signal, audio_duration
    else:
        return audio_signal


THREAD_POOL_AUDIO_LOADER = Pool(2)


def audio_batch_generator(
    audio_files: list[str],
) -> Iterator[np.ndarray | tuple[np.ndarray, float]]:
    """
    Generate audio data in batches using thread pool for parallel loading.

    Args:
        audio_files: List of paths to audio files

    Returns:
        Iterator of loaded audio signals
    """
    return THREAD_POOL_AUDIO_LOADER.imap(load_audio, audio_files)


def pad_or_trim(
    array: torch.Tensor | np.ndarray,
    length: int = N_SAMPLES,
    *,
    axis: int = -1,
) -> torch.Tensor | np.ndarray:
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.

    Args:
        array: Audio array (torch tensor or numpy array)
        length: Target length for the array dimension specified by axis
        axis: Dimension to pad or trim

    Returns:
        Padded or trimmed array of the same type as input
    """

    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


class TorchSTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            window=self.window,  # type: ignore
            return_complex=True,
        )


class LogMelSpectogram(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        padding: int = 0,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.padding = padding

        mel_filters = np.load(os.path.join(BASE_PATH, "assets/mel_filters.npz"))
        mel_filters = torch.from_numpy(mel_filters[f"mel_{n_mels}"])
        self.register_buffer("mel_filters", mel_filters)

        self.stft = TorchSTFT(n_fft, hop_length)

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        seq_len = torch.floor(seq_len / self.hop_length)
        return seq_len.to(dtype=torch.long)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        seq_len = self.get_seq_len(seq_len.float())

        if self.padding > 0:
            x = F.pad(x, (0, self.padding))

        x = self.stft(x)

        x = x[..., :-1].abs() ** 2
        x = self.mel_filters @ x  # mels

        x = torch.clamp(x, min=1e-10).log10()  # log_mels
        x = torch.maximum(x, torch.amax(x, dim=(1, 2), keepdim=True) - 8.0)  # clip
        x = (x + 4.0) / 4.0  # scale

        return x, seq_len
