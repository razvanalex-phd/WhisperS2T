import os
from typing import Any

import ctranslate2
import numpy as np
import tokenizers
import torch

from whisper_s2t.backends import WhisperModel
from whisper_s2t.backends.hf_utils import download_model
from whisper_s2t.backends.tensorrt.engine_builder import (
    TRTBuilderConfig,
    build_trt_engine,
    load_trt_build_config,
)
from whisper_s2t.backends.tensorrt.trt_model import WhisperTRT
from whisper_s2t.backends.tokenizer import Tokenizer
from whisper_s2t.configs import *

FAST_ASR_OPTIONS: dict[str, Any] = {
    "beam_size": 1,
    "best_of": 1,  # Placeholder
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4,  # Placeholder
    "log_prob_threshold": -1.0,  # Placeholder
    "no_speech_threshold": 0.5,  # Placeholder
    "prefix": None,  # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False,  # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": "tiny",
}


BEST_ASR_CONFIG: dict[str, Any] = {
    "beam_size": 5,
    "best_of": 1,  # Placeholder
    "patience": 2,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4,  # Placeholder
    "log_prob_threshold": -1.0,  # Placeholder
    "no_speech_threshold": 0.5,  # Placeholder
    "prefix": None,  # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False,  # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": "tiny",
}


class WhisperModelTRT(WhisperModel):
    def __init__(
        self,
        model_name_or_path: str,
        cpu_threads: int = 4,
        num_workers: int = 1,
        device: str = "cuda",
        device_index: int = 0,
        compute_type: str = "float16",
        max_text_token_len: int = MAX_TEXT_TOKEN_LENGTH,
        force_trt_build: bool = False,
        asr_options: dict[str, Any] = {},
        **model_kwargs: Any,
    ) -> None:
        # ASR Options
        self.asr_options = FAST_ASR_OPTIONS
        self.asr_options.update(asr_options)

        # Get local model path or build a new engine
        if os.path.isdir(model_name_or_path):
            self.model_path = model_name_or_path
            trt_build_args = load_trt_build_config(self.model_path)
        else:
            trt_build_args = model_kwargs.get("trt_build_args", None)
            if trt_build_args is None:
                print(
                    f"'trt_build_args' not provided in model_kwargs, using default configs."
                )
                trt_build_args = TRTBuilderConfig(
                    max_output_len=max_text_token_len,
                    max_beam_width=self.asr_options["beam_size"],
                )

            self.model_path = build_trt_engine(
                model_name=model_name_or_path,
                args=trt_build_args,
                force=force_trt_build,
            )

        if "trt_build_args" in model_kwargs:
            del model_kwargs["trt_build_args"]

        self.trt_build_args = trt_build_args

        # Update params according to TRT Build Args
        if max_text_token_len > self.trt_build_args.max_output_len:
            print(
                f"'max_text_token_len' cannot be larger than 'self.trt_build_args.max_output_len'. Setting 'max_text_token_len' to {self.trt_build_args.max_output_len}."
            )
            max_text_token_len = self.trt_build_args.max_output_len

        if self.asr_options["beam_size"] > self.trt_build_args.max_beam_width:
            print(
                f"'beam_size' cannot be larger than 'self.trt_build_args.max_beam_width'. Setting 'beam_size' to {self.trt_build_args.max_beam_width}."
            )
            self.asr_options["beam_size"] = self.trt_build_args.max_beam_width

        # Load model
        self.model = WhisperTRT(
            self.model_path,
            runtime_args=model_kwargs.get("runtime_args", None),
        )

        if "runtime_args" in model_kwargs:
            del model_kwargs["runtime_args"]

        # Load tokenizer
        tokenizer_file = os.path.join(self.model_path, "tokenizer.json")
        tokenizer = Tokenizer(
            tokenizers.Tokenizer.from_file(tokenizer_file),
            self.model.is_multilingual,
        )

        if self.asr_options["word_timestamps"]:
            self.aligner_model_path = download_model(
                self.asr_options["word_aligner_model"]
            )
            self.aligner_model = ctranslate2.models.Whisper(
                self.aligner_model_path,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                intra_threads=cpu_threads,
                inter_threads=num_workers,
            )

        self.max_output_len = max_text_token_len
        self.generate_kwargs = {
            "end_id": tokenizer.eot,
            "pad_id": tokenizer.eot,
            "max_new_tokens": max_text_token_len,
            "length_penalty": self.asr_options["length_penalty"],
            "repetition_penalty": self.asr_options["repetition_penalty"],
            "num_beams": self.asr_options["beam_size"],
            "temperature": self.asr_options["sampling_temperature"],
        }

        super().__init__(
            tokenizer=tokenizer,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs,
        )
        self.tokenizer: Tokenizer

    def update_generation_kwargs(self, params: dict[str, Any] = {}) -> None:
        self.generate_kwargs.update(params)

        if "max_text_token_len" in params:
            self.update_params(
                params={"max_text_token_len": params["max_text_token_len"]}
            )

    def encode(self, features: torch.Tensor) -> tuple:
        """
        [Not Used]
        """

        return self.model.encode(features)

    def assign_word_timings(
        self,
        alignments: list[tuple[int, int]],
        text_token_probs: list[float],
        words: list[str],
        word_tokens: list,
    ) -> list[dict[str, float | str]]:
        text_indices = np.array([pair[0] for pair in alignments])
        time_indices = np.array([pair[1] for pair in alignments])

        if len(word_tokens) <= 1:
            return []

        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        if len(word_boundaries) <= 1:
            return []

        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * TIME_PRECISION
        start_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]
        word_probs = [
            np.mean(text_token_probs[i:j])
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ]

        return [
            dict(
                word=word,
                start=float(round(start, 2)),
                end=float(round(end, 2)),
                prob=float(round(prob, 2)),
            )
            for word, start, end, prob in zip(words, start_times, end_times, word_probs)
        ]

    def align_words(
        self,
        features: np.ndarray | torch.Tensor,
        texts: list[str],
        text_tokens: list[list[int]],
        sot_seqs: list[tuple[int, ...]],
        seq_lens: torch.Tensor,
        seg_metadata: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        lang_codes = [_["lang_code"] for _ in seg_metadata]
        word_tokens = self.tokenizer.split_to_word_tokens_batch(
            texts, text_tokens, lang_codes
        )

        start_seq_wise_req = {}
        for _idx, _sot_seq in enumerate(sot_seqs):
            try:
                start_seq_wise_req[_sot_seq].append(_idx)
            except:
                start_seq_wise_req[_sot_seq] = [_idx]

        token_alignments = [[] for _ in seg_metadata]
        for start_seq, req_idx in start_seq_wise_req.items():
            res = self.aligner_model.align(
                ctranslate2.StorageView.from_array(features[req_idx]),
                start_sequence=list(start_seq),
                text_tokens=[text_tokens[_] for _ in req_idx],
                num_frames=list(seq_lens[req_idx].detach().cpu().numpy()),
                median_filter_width=7,
            )

            for _res, _req_idx in zip(res, req_idx):
                token_alignments[_req_idx] = _res

        word_timings = []
        for _idx, _seg_metadata in enumerate(seg_metadata):
            _word_timings = self.assign_word_timings(
                token_alignments[_idx].alignments,  # type: ignore
                token_alignments[_idx].text_token_probs,  # type: ignore
                word_tokens[_idx][0],
                word_tokens[_idx][1],
            )

            stitched_seg = _seg_metadata["stitched_seg"]

            current_seg_idx = 0
            current_offset = _seg_metadata["start_time"]

            for w in _word_timings:
                while (w["start"] + current_offset) >= stitched_seg[current_seg_idx][1]:
                    current_seg_idx += 1
                    current_offset += (
                        stitched_seg[current_seg_idx][0]
                        - stitched_seg[current_seg_idx - 1][1]
                    )

                w["start"] += current_offset
                w["end"] += current_offset

            word_timings.append(_word_timings)

        return word_timings

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
        if self.compute_type == "float16":
            features = features.half()

        prompt_len = max(map(len, prompts))
        max_tokens = self.generate_kwargs["max_new_tokens"] + prompt_len
        if self.max_output_len < max_tokens:
            self.generate_kwargs["max_new_tokens"] = self.max_output_len - prompt_len

        result = self.model.generate(features, prompts, **self.generate_kwargs)
        texts = self.tokenizer.decode_batch([x[0] for x in result])

        response = []
        for idx, _r in enumerate(result):
            response.append({"text": texts[idx].strip()})

        if self.asr_options["word_timestamps"]:
            text_tokens = [
                [_t for _t in x[0] if _t < self.tokenizer.eot] + [self.tokenizer.eot]
                for x in result
            ]
            sot_seqs = [tuple(_[-4:]) for _ in prompts]
            word_timings = self.align_words(
                align_features,
                texts,
                text_tokens,
                sot_seqs,
                align_seq_lens,
                seg_metadata,
            )

            for _response, _word_timings in zip(response, word_timings):
                _response["word_timestamps"] = _word_timings

        return response
