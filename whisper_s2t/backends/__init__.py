from abc import ABC, abstractmethod
from typing import Any, TypeVar

import torch
from tqdm import tqdm

from whisper_s2t.audio import LogMelSpectogram
from whisper_s2t.configs import *
from whisper_s2t.data import WhisperDataLoader
from whisper_s2t.speech_segmenter import SpeechSegmenter

T = TypeVar("T")


class NoneTokenizer:
    def __init__(self) -> None:
        self.sot_prev: int = 0
        self.silent_token: int = 0
        self.no_timestamps: int = 0
        self.timestamp_begin: int = 0

    def sot_sequence(
        self,
        task: Any | None = None,
        lang: str | None = None,
    ) -> list[Any]:
        return [task, lang]

    def encode(self, _text: str) -> list[int]:
        return [0]


def fix_batch_param(
    param: list[T] | T | None,
    default_value: T | None,
    N: int,
) -> list[T]:
    if param is None:
        param = N * [default_value]  # type: ignore
    elif type(param) == type(default_value):
        param = N * [param]  # type: ignore
    elif len(param) != N:  # type: ignore
        param = N * [param[0]]  # type: ignore

    return param  # type: ignore


class WhisperModel(ABC):
    def __init__(
        self,
        tokenizer: Any | None = None,
        vad_model: Any | None = None,
        n_mels: int = 80,
        align_n_mels: int = 80,
        device: str = "cuda",
        device_index: int = 0,
        compute_type: str = "float16",
        merge_chunks: bool = True,
        dta_padding: float = 3.0,
        use_dynamic_time_axis: bool = False,
        max_speech_len: float = 29.0,
        max_text_token_len: int = MAX_TEXT_TOKEN_LENGTH,
        without_timestamps: bool = True,
        speech_segmenter_options: dict[str, Any] = {},
    ) -> None:
        # Configure Params
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type

        self.n_mels = n_mels
        self.align_n_mels = align_n_mels
        self.merge_chunks = merge_chunks
        self.max_speech_len = max_speech_len

        self.dta_padding = dta_padding
        self.use_dynamic_time_axis = use_dynamic_time_axis

        self.without_timestamps = without_timestamps
        self.max_text_token_len = max_text_token_len

        self.vad_model = vad_model
        self.speech_segmenter_options = speech_segmenter_options
        self.speech_segmenter_options["max_seg_len"] = self.max_speech_len

        # Tokenizer
        if tokenizer is None:
            tokenizer = NoneTokenizer()

        self.tokenizer = tokenizer

        self._init_dependables()

    def _init_dependables(self) -> None:
        # Rescaled Params
        self.dta_padding = int(self.dta_padding * SAMPLE_RATE)
        self.max_initial_prompt_len = self.max_text_token_len // 2 - 1

        # Load Pre Processor
        self.align_preprocessor = LogMelSpectogram(n_mels=self.align_n_mels).to(
            self.device
        )
        self.preprocessor = LogMelSpectogram(n_mels=self.n_mels).to(self.device)

        # Load Speech Segmenter
        self.speech_segmenter = SpeechSegmenter(
            self.vad_model, device=self.device, **self.speech_segmenter_options
        )

        # Load Data Loader
        self.data_loader = WhisperDataLoader(
            self.device,
            self.tokenizer,
            self.speech_segmenter,
            dta_padding=self.dta_padding,
            without_timestamps=self.without_timestamps,
            max_speech_len=self.max_speech_len,
            max_initial_prompt_len=self.max_initial_prompt_len,
            use_dynamic_time_axis=self.use_dynamic_time_axis,
            merge_chunks=self.merge_chunks,
        )

    def update_params(self, params: dict[str, Any] = {}) -> None:
        for key, value in params.items():
            setattr(self, key, value)

        self._init_dependables()

    @abstractmethod
    def generate_segment_batched(
        self,
        features: torch.Tensor,
        prompts: list[list[int]],
        seq_lens: torch.Tensor,
        seg_metadata: list[dict[str, Any]],
        *,
        align_features: torch.Tensor,
        align_seq_lens: torch.Tensor,
    ) -> list[dict[str, Any]]:
        raise NotImplemented()

    @torch.no_grad()
    def transcribe(
        self,
        audio_files: list[str],
        lang_codes: list[str] | str | None = None,
        tasks: list[str] | str | None = None,
        initial_prompts: list[str | None] | str | None = None,
        batch_size: int = 8,
        use_vad: bool = False,
    ) -> list[list[dict[str, Any]]]:
        lang_codes = fix_batch_param(lang_codes, "en", len(audio_files))
        tasks = fix_batch_param(tasks, "transcribe", len(audio_files))
        initial_prompts = fix_batch_param(initial_prompts, None, len(audio_files))

        responses = [[] for _ in audio_files]

        pbar_pos = 0
        with tqdm(total=len(audio_files) * 100, desc=f"Transcribing") as pbar:
            dataloader = self.data_loader(
                audio_files,
                lang_codes,
                tasks,
                initial_prompts,
                batch_size=batch_size,
                use_vad=use_vad,
            )

            for data in dataloader:
                signals, prompts, seq_len, seg_metadata, pbar_update = data

                mels, seq_lens = self.preprocessor(signals, seq_len)
                align_mels, align_seq_lens = self.align_preprocessor(signals, seq_len)
                res = self.generate_segment_batched(
                    mels.to(self.device),
                    prompts,
                    seq_lens,
                    seg_metadata,
                    align_features=align_mels.to(self.device),
                    align_seq_lens=align_seq_lens,
                )

                for res_idx, _seg_metadata in enumerate(seg_metadata):
                    responses[_seg_metadata["file_id"]].append(
                        {
                            **res[res_idx],
                            "start_time": round(_seg_metadata["start_time"], 3),
                            "end_time": round(_seg_metadata["end_time"], 3),
                        }
                    )

                if (pbar_pos) <= pbar.total:
                    pbar_pos += pbar_update
                    pbar.update(pbar_update)

            pbar.update(pbar.total - pbar_pos)

        return responses

    @torch.no_grad()
    def transcribe_with_vad(
        self,
        audio_files: list[str],
        lang_codes: list[str] | str | None = None,
        tasks: list[str] | str | None = None,
        initial_prompts: list[str | None] | str | None = None,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        return self.transcribe(
            audio_files,
            lang_codes=lang_codes,
            tasks=tasks,
            initial_prompts=initial_prompts,
            batch_size=batch_size,
            use_vad=True,
        )
