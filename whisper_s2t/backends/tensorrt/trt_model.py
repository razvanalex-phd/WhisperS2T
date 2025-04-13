# Source: https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/whisper/run.py

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import tensorrt_llm
import torch
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.bindings import GptJsonConfig, KVCacheType
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo

from whisper_s2t.backends.tensorrt.engine_builder import load_trt_build_config

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def remove_tensor_padding(
    input_tensor: torch.Tensor,
    input_tensor_lengths: torch.Tensor | None = None,
    pad_value: int = 0,
) -> torch.Tensor:
    if input_tensor.dim() == 2:
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] != pad_value
        ), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    elif input_tensor.dim() == 3:
        # Audio tensor case: batch, seq_len, feature_len
        assert (
            input_tensor_lengths is not None
        ), "input_tensor_lengths must be provided for 3D input_tensor"
        batch_size, _seq_len, _feature_len = input_tensor.shape

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(batch_size):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length, :])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)

    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

    return output_tensor


def read_config(component: str, engine_dir: Path) -> OrderedDict:
    config_path = engine_dir / component / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config["pretrained_config"])
    model_config.update(config["build_config"])
    return model_config


class WhisperEncoding:

    def __init__(self, engine_dir: Path) -> None:
        self.session: Session = self.get_session(engine_dir)
        config = read_config("encoder", engine_dir)
        self.n_mels: int = config["n_mels"]
        self.dtype: str = config["dtype"]
        self.num_languages: int = config["num_languages"]
        self.encoder_config: OrderedDict = config

    def get_session(self, engine_dir: Path) -> Session:
        serialize_path = engine_dir / "encoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(
        self,
        mel: torch.Tensor,
        mel_input_lengths: torch.Tensor,
        encoder_downsampling_factor: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_config["plugin_config"]["remove_input_padding"]:
            # mel B,D,T -> B,T,D -> BxT, D
            mel = mel.transpose(1, 2)
            mel = remove_tensor_padding(mel, mel_input_lengths)

        inputs = OrderedDict()
        inputs["input_features"] = mel
        inputs["input_lengths"] = mel_input_lengths

        output_list = [
            TensorInfo("input_features", str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo(
                "input_lengths", str_dtype_to_trt("int32"), mel_input_lengths.shape
            ),
        ]

        output_info = (self.session).infer_shapes(output_list)

        outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda"
            )
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, "Engine execution failed"
        stream.synchronize()
        encoder_output = outputs["encoder_output"]
        encoder_output_lengths = mel_input_lengths // encoder_downsampling_factor

        return encoder_output, encoder_output_lengths


class WhisperDecoding:

    def __init__(
        self,
        engine_dir: Path,
        runtime_mapping: tensorrt_llm.Mapping,
        debug_mode: bool = False,
    ) -> None:
        self.decoder_config: OrderedDict = read_config("decoder", engine_dir)
        self.decoder_generation_session: tensorrt_llm.runtime.GenerationSession = (
            self.get_session(engine_dir, runtime_mapping, debug_mode)
        )

    def get_session(
        self,
        engine_dir: Path,
        runtime_mapping: tensorrt_llm.Mapping,
        debug_mode: bool = False,
    ) -> tensorrt_llm.runtime.GenerationSession:
        serialize_path = engine_dir / "decoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config["max_batch_size"],
            max_beam_width=self.decoder_config["max_beam_width"],
            num_heads=self.decoder_config["num_attention_heads"],
            num_kv_heads=self.decoder_config["num_attention_heads"],
            hidden_size=self.decoder_config["hidden_size"],
            vocab_size=self.decoder_config["vocab_size"],
            cross_attention=True,
            num_layers=self.decoder_config["num_hidden_layers"],
            gpt_attention_plugin=self.decoder_config["plugin_config"][
                "gpt_attention_plugin"
            ],
            remove_input_padding=self.decoder_config["plugin_config"][
                "remove_input_padding"
            ],
            kv_cache_type=(
                KVCacheType.PAGED
                if self.decoder_config["plugin_config"]["paged_kv_cache"] == True
                else KVCacheType.CONTINUOUS
            ),
            has_position_embedding=self.decoder_config["has_position_embedding"],
            dtype=self.decoder_config["dtype"],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode,
        )

        return decoder_generation_session

    def generate(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_max_input_length: int,
        encoder_input_lengths: torch.Tensor,
        eot_id: int,
        max_new_tokens: int = 40,
        num_beams: int = 1,
        stop_words_list: list[list[int]] | None = None,
        bad_words_list: list[list[int]] | None = None,
        temperature: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
    ) -> list[list[int]]:
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = (
            torch.ones([batch_size, 1, encoder_max_input_length]).int().cuda()
        )

        # generation config
        sampling_config = SamplingConfig(
            end_id=eot_id,
            pad_id=eot_id,
            num_beams=num_beams,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,  # type: ignore
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_max_input_length,
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config["plugin_config"]["remove_input_padding"]:
            # 50256 is the index of <pad> for all whisper models' decoder
            WHISPER_PAD_TOKEN_ID = 50256
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=WHISPER_PAD_TOKEN_ID
            )
            if encoder_outputs.dim() == 3:
                encoder_output_lens = torch.full(
                    (encoder_outputs.shape[0],),
                    encoder_outputs.shape[1],
                    dtype=torch.int32,
                    device="cuda",
                )

                encoder_outputs = remove_tensor_padding(
                    encoder_outputs, encoder_output_lens
                )
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()  # type: ignore
        return output_ids


class WhisperTRT:

    def __init__(
        self,
        engine_dir: str | Path,
        debug_mode: bool = False,
        assets_dir: str | Path | None = None,
        use_py_session: bool = False,
    ):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)
        encoder_config = read_config("encoder", engine_dir)
        decoder_config = read_config("decoder", engine_dir)
        self.n_mels = encoder_config["n_mels"]
        self.num_languages = encoder_config["num_languages"]
        self.is_multilingual = decoder_config["vocab_size"] >= 51865
        if use_py_session:
            self.encoder = WhisperEncoding(engine_dir)
            self.decoder = WhisperDecoding(
                engine_dir, runtime_mapping, debug_mode=debug_mode
            )
        else:
            json_config = GptJsonConfig.parse_file(
                engine_dir / "decoder" / "config.json"
            )
            assert json_config.model_config.supports_inflight_batching
            trt_build_args = load_trt_build_config(engine_dir)
            runner_kwargs = dict(
                engine_dir=engine_dir,
                is_enc_dec=True,
                kv_cache_free_gpu_memory_fraction=0.9,
                max_batch_size=trt_build_args.max_batch_size,
                max_input_len=trt_build_args.max_input_len_enc,
                max_output_len=trt_build_args.max_output_len,
                max_beam_width=trt_build_args.max_beam_width,
                debug_mode=trt_build_args.debug_mode,
            )
            self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)  # type: ignore
        self.use_py_session = use_py_session

    def process_batch(
        self,
        mel: torch.Tensor,
        mel_input_lengths: torch.Tensor,
        prompt: list[torch.Tensor],
        num_beams: int = 1,
        max_new_tokens: int = 96,
        end_id: int = 0,
        pad_id: int = 0,
        stop_words_list: list[list[int]] | None = None,
        bad_words_list: list[list[int]] | None = None,
        temperature: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
    ) -> list[list[int]]:
        if self.use_py_session:
            encoder_output, encoder_output_lengths = self.encoder.get_audio_features(
                mel, mel_input_lengths
            )
            encoder_max_input_length = torch.max(encoder_output_lengths).item()
            output_ids = self.decoder.generate(
                prompt,
                encoder_output,
                encoder_max_input_length,
                encoder_output_lengths,
                end_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                bad_words_list=bad_words_list,
                stop_words_list=stop_words_list,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )
        else:
            with torch.no_grad():
                outputs = self.model_runner_cpp.generate(
                    batch_input_ids=prompt,
                    encoder_input_features=mel.transpose(1, 2),
                    encoder_output_lengths=mel_input_lengths // 2,
                    max_new_tokens=max_new_tokens,
                    end_id=end_id,
                    pad_id=pad_id,
                    num_beams=num_beams,
                    output_sequence_lengths=True,
                    return_dict=True,
                    bad_words_list=bad_words_list,
                    stop_words_list=stop_words_list,
                    temperature=temperature,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                )
                torch.cuda.synchronize()
                output_ids = outputs["output_ids"].cpu().numpy().tolist()  # type: ignore
        return output_ids

    def encode(
        self,
        mel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel_input_lengths = torch.full(
            (mel.shape[0],),
            mel.shape[2],
            dtype=torch.int32,
            device=mel.device,
        )
        return (
            *self.encoder.get_audio_features(mel, mel_input_lengths),
            mel_input_lengths,
        )

    def generate(
        self,
        features: torch.Tensor,
        prompts: list[list[int]],
        **generate_kwargs: Any,
    ) -> list[list[int]]:
        features_lengths = torch.full(
            (features.shape[0],),
            features.shape[2],
            dtype=torch.int32,
            device=features.device,
        )
        prompts = list(map(lambda x: torch.tensor(x, dtype=torch.int32), prompts))
        output_ids = self.process_batch(
            features,
            features_lengths,
            prompts,
            **generate_kwargs,
        )
        return output_ids
