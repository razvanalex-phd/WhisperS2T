# Modified from: https://github.com/NVIDIA/TensorRT-LLM/blob/d51ae53940e2a259375069c57fca7656dc9c5af2/examples/models/core/whisper/convert_checkpoint.py

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

import argparse
import json
import logging
import os
import sys
import time
from argparse import Namespace

import torch
from safetensors.torch import save_file
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.models.convert_utils import weight_only_quantize_dict
from tensorrt_llm.quantization import QuantAlgo

from whisper_s2t.backends.tensorrt.engine_builder import (
    TRTBuilderConfig,
    load_trt_build_config,
)
from whisper_s2t.backends.tensorrt.engine_builder.model_utils import sinusoids

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
)


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="assets")
    parser.add_argument("--quant_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default="large-v2",
        choices=[
            "large-v3-turbo",
            "large-v3",
            "large-v2",
            "medium",
            "small",
            "base",
            "tiny",
            "medium.en",
            "small.en",
            "base.en",
            "tiny.en",
            "distil-large-v3",
            "distil-large-v2",
            "distil-medium.en",
            "distil-small.en",
        ],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--logits_dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    # NOTE: this is the only one used.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tllm_checkpoint",
        help="The path to save the TensorRT-LLM checkpoint",
    )
    parser.add_argument(
        "--use_weight_only",
        default=False,
        action="store_true",
        help="Quantize weights for the various GEMMs to INT4/INT8."
        "See --weight_only_precision to set the precision",
    )
    parser.add_argument(
        "--weight_only_precision",
        const="int8",
        type=str,
        nargs="?",
        default="int8",
        choices=["int8", "int4"],
        help="Define the precision for the weights when using weight-only quantization."
        "You must also use --use_weight_only for that argument to have an impact.",
    )
    args = parser.parse_args()
    return args


def get_encoder_config(model_metadata: dict, dtype: str, quant_algo: QuantAlgo) -> dict:
    model_is_multilingual = model_metadata["n_vocab"] >= 51865
    num_languages = model_metadata["n_vocab"] - 51765 - int(model_is_multilingual)
    return {
        "architecture": "WhisperEncoder",
        "dtype": dtype,
        "num_hidden_layers": model_metadata["n_audio_layer"],
        "num_attention_heads": model_metadata["n_audio_head"],
        "hidden_size": model_metadata["n_audio_state"],
        "max_position_embeddings": model_metadata["n_audio_ctx"],
        "has_position_embedding": True,
        "n_mels": model_metadata["n_mels"],
        "vocab_size": model_metadata["n_vocab"],
        "hidden_act": "gelu",
        "num_languages": num_languages,
        "quantization": {"quant_algo": quant_algo},
    }


def get_decoder_config(
    model_metadata: dict,
    dtype: str,
    logits_dtype: str,
    quant_algo: QuantAlgo,
) -> dict:
    # Ensure max_position_embeddings has a valid value
    max_pos_embeddings = model_metadata.get("n_text_ctx", 448)
    if max_pos_embeddings is None:
        max_pos_embeddings = 448  # Default value if not specified

    return {
        "architecture": "DecoderModel",
        "dtype": dtype,
        "logits_dtype": logits_dtype,
        "num_hidden_layers": model_metadata["n_text_layer"],
        "num_attention_heads": model_metadata["n_text_head"],
        "hidden_size": model_metadata["n_text_state"],
        "norm_epsilon": 1e-5,
        "vocab_size": model_metadata["n_vocab"],
        "hidden_act": "gelu",
        "use_parallel_embedding": False,
        "embedding_sharding_dim": 0,
        "max_position_embeddings": max_pos_embeddings,
        "use_prompt_tuning": False,
        "head_size": model_metadata["n_text_state"] // model_metadata["n_text_head"],
        "has_position_embedding": True,
        "layernorm_type": LayerNormType.LayerNorm,
        "has_attention_qkvo_bias": True,
        "has_mlp_bias": True,
        "has_model_final_layernorm": True,
        "has_embedding_layernorm": False,
        "has_embedding_scale": False,
        "ffn_hidden_size": 4 * model_metadata["n_text_state"],
        "q_scaling": 1.0,
        "layernorm_position": LayerNormPositionType.pre_layernorm,
        "relative_attention": False,
        "max_distance": 0,
        "num_buckets": 0,
        "model_type": "whisper",
        "rescale_before_lm_head": False,
        "encoder_hidden_size": model_metadata["n_text_state"],
        "encoder_num_heads": model_metadata["n_text_head"],
        "encoder_head_size": None,
        "skip_cross_qkv": False,
        "quantization": {"quant_algo": quant_algo},
    }


def convert_openai_whisper_encoder(
    model_metadata: dict,
    model_params: dict[str, torch.Tensor],
    quant_algo: QuantAlgo | None = None,
) -> dict[str, torch.Tensor]:
    weights = {}

    weights["position_embedding.weight"] = sinusoids(
        model_metadata["n_audio_ctx"], model_metadata["n_audio_state"]
    ).contiguous()

    weights["conv1.weight"] = torch.unsqueeze(
        model_params["encoder.conv1.weight"], -1
    ).contiguous()
    weights["conv1.bias"] = model_params["encoder.conv1.bias"].contiguous()
    weights["conv2.weight"] = torch.unsqueeze(
        model_params["encoder.conv2.weight"], -1
    ).contiguous()
    weights["conv2.bias"] = model_params["encoder.conv2.bias"].contiguous()

    # Encoder conv needs to run in fp32 on Volta/Turing
    major, _minor = torch.cuda.get_device_capability()
    if not major >= 8:
        weights["conv1.weight"] = weights["conv1.weight"].float()
        weights["conv1.bias"] = weights["conv1.bias"].float()
        weights["conv2.weight"] = weights["conv2.weight"].float()
        weights["conv2.bias"] = weights["conv2.bias"].float()

    for i in range(model_metadata["n_audio_layer"]):
        trtllm_layer_name_prefix = f"encoder_layers.{i}"

        weights[f"{trtllm_layer_name_prefix}.attention_layernorm.weight"] = (
            model_params["encoder.blocks." + str(i) + ".attn_ln.weight"].contiguous()
        )
        weights[f"{trtllm_layer_name_prefix}.attention_layernorm.bias"] = model_params[
            "encoder.blocks." + str(i) + ".attn_ln.bias"
        ].contiguous()

        t = torch.cat(
            [
                model_params["encoder.blocks." + str(i) + ".attn.query.weight"],
                model_params["encoder.blocks." + str(i) + ".attn.key.weight"],
                model_params["encoder.blocks." + str(i) + ".attn.value.weight"],
            ],
            dim=0,
        ).contiguous()

        weights[f"{trtllm_layer_name_prefix}.attention.qkv.weight"] = t

        bias_shape = model_params["encoder.blocks." + str(i) + ".attn.query.bias"].shape
        dtype = model_params["encoder.blocks." + str(i) + ".attn.query.bias"].dtype
        fused_bias = torch.cat(
            [
                model_params["encoder.blocks." + str(i) + ".attn.query.bias"],
                torch.zeros([*bias_shape], dtype=dtype),
                model_params["encoder.blocks." + str(i) + ".attn.value.bias"],
            ],
            dim=0,
        ).contiguous()

        weights[f"{trtllm_layer_name_prefix}.attention.qkv.bias"] = fused_bias

        t = model_params["encoder.blocks." + str(i) + ".attn.out.weight"].contiguous()

        weights[f"{trtllm_layer_name_prefix}.attention.dense.weight"] = t
        weights[f"{trtllm_layer_name_prefix}.attention.dense.bias"] = model_params[
            "encoder.blocks." + str(i) + ".attn.out.bias"
        ].contiguous()

        weights[f"{trtllm_layer_name_prefix}.mlp_layernorm.weight"] = model_params[
            "encoder.blocks." + str(i) + ".mlp_ln.weight"
        ].contiguous()
        weights[f"{trtllm_layer_name_prefix}.mlp_layernorm.bias"] = model_params[
            "encoder.blocks." + str(i) + ".mlp_ln.bias"
        ].contiguous()

        t = model_params["encoder.blocks." + str(i) + ".mlp.0.weight"].contiguous()
        weights[f"{trtllm_layer_name_prefix}.mlp.fc.weight"] = t

        weights[f"{trtllm_layer_name_prefix}.mlp.fc.bias"] = model_params[
            "encoder.blocks." + str(i) + ".mlp.0.bias"
        ].contiguous()

        t = model_params["encoder.blocks." + str(i) + ".mlp.2.weight"].contiguous()
        weights[f"{trtllm_layer_name_prefix}.mlp.proj.weight"] = t

        weights[f"{trtllm_layer_name_prefix}.mlp.proj.bias"] = model_params[
            "encoder.blocks." + str(i) + ".mlp.2.bias"
        ].contiguous()

    weights["ln_post.weight"] = model_params["encoder.ln_post.weight"].contiguous()
    weights["ln_post.bias"] = model_params["encoder.ln_post.bias"].contiguous()

    return weight_only_quantize_dict(weights, quant_algo=quant_algo, plugin=True)  # type: ignore


def convert_openai_whisper_decoder(
    model_metadata: dict,
    model_params: dict[str, torch.Tensor],
    quant_algo: QuantAlgo | None = None,
) -> dict[str, torch.Tensor]:
    weights = {}

    weights["embedding.vocab_embedding.weight"] = model_params[
        "decoder.token_embedding.weight"
    ]
    weights["embedding.position_embedding.weight"] = model_params[
        "decoder.positional_embedding"
    ]
    weights["lm_head.weight"] = model_params["decoder.token_embedding.weight"].clone()

    for i in range(model_metadata["n_text_layer"]):
        trtllm_layer_name_prefix = f"decoder_layers.{i}"

        t = torch.cat(
            [
                model_params["decoder.blocks." + str(i) + ".attn.query.weight"],
                model_params["decoder.blocks." + str(i) + ".attn.key.weight"],
                model_params["decoder.blocks." + str(i) + ".attn.value.weight"],
            ],
            dim=0,
        )
        weights[f"{trtllm_layer_name_prefix}.self_attention.qkv.weight"] = t

        t = model_params["decoder.blocks." + str(i) + ".attn.out.weight"].contiguous()
        weights[f"{trtllm_layer_name_prefix}.self_attention.dense.weight"] = t

        bias_shape = model_params["decoder.blocks." + str(i) + ".attn.query.bias"].shape
        dtype = model_params["decoder.blocks." + str(i) + ".attn.query.bias"].dtype
        weights[f"{trtllm_layer_name_prefix}.self_attention.qkv.bias"] = torch.cat(
            [
                model_params["decoder.blocks." + str(i) + ".attn.query.bias"],
                torch.zeros([*bias_shape], dtype=dtype),
                model_params["decoder.blocks." + str(i) + ".attn.value.bias"],
            ],
            dim=0,
        )
        weights[f"{trtllm_layer_name_prefix}.self_attention.dense.bias"] = model_params[
            "decoder.blocks." + str(i) + ".attn.out.bias"
        ]

        weights[f"{trtllm_layer_name_prefix}.self_attention_layernorm.weight"] = (
            model_params["decoder.blocks." + str(i) + ".attn_ln.weight"]
        )
        weights[f"{trtllm_layer_name_prefix}.self_attention_layernorm.bias"] = (
            model_params["decoder.blocks." + str(i) + ".attn_ln.bias"]
        )

        t = torch.cat(
            [
                model_params["decoder.blocks." + str(i) + ".cross_attn.query.weight"],
                model_params["decoder.blocks." + str(i) + ".cross_attn.key.weight"],
                model_params["decoder.blocks." + str(i) + ".cross_attn.value.weight"],
            ],
            dim=0,
        )
        weights[f"{trtllm_layer_name_prefix}.cross_attention.qkv.weight"] = t

        t = model_params[
            "decoder.blocks." + str(i) + ".cross_attn.out.weight"
        ].contiguous()
        weights[f"{trtllm_layer_name_prefix}.cross_attention.dense.weight"] = t

        bias_shape = model_params[
            "decoder.blocks." + str(i) + ".cross_attn.query.bias"
        ].shape
        dtype = model_params[
            "decoder.blocks." + str(i) + ".cross_attn.query.bias"
        ].dtype
        cross_attn_qkv_bias = torch.cat(
            [
                model_params["decoder.blocks." + str(i) + ".cross_attn.query.bias"],
                torch.zeros([*bias_shape], dtype=dtype),
                model_params["decoder.blocks." + str(i) + ".cross_attn.value.bias"],
            ],
            dim=0,
        )

        weights[f"{trtllm_layer_name_prefix}.cross_attention.qkv.bias"] = (
            cross_attn_qkv_bias
        )

        weights[f"{trtllm_layer_name_prefix}.cross_attention.dense.bias"] = (
            model_params["decoder.blocks." + str(i) + ".cross_attn.out.bias"]
        )

        weights[f"{trtllm_layer_name_prefix}.cross_attention_layernorm.weight"] = (
            model_params["decoder.blocks." + str(i) + ".cross_attn_ln.weight"]
        )
        weights[f"{trtllm_layer_name_prefix}.cross_attention_layernorm.bias"] = (
            model_params["decoder.blocks." + str(i) + ".cross_attn_ln.bias"]
        )

        t = model_params["decoder.blocks." + str(i) + ".mlp.0.weight"].contiguous()
        weights[f"{trtllm_layer_name_prefix}.mlp.fc.weight"] = t

        t = model_params["decoder.blocks." + str(i) + ".mlp.2.weight"].contiguous()
        weights[f"{trtllm_layer_name_prefix}.mlp.proj.weight"] = t

        weights[f"{trtllm_layer_name_prefix}.mlp.fc.bias"] = model_params[
            "decoder.blocks." + str(i) + ".mlp.0.bias"
        ]
        weights[f"{trtllm_layer_name_prefix}.mlp.proj.bias"] = model_params[
            "decoder.blocks." + str(i) + ".mlp.2.bias"
        ]

        weights[f"{trtllm_layer_name_prefix}.mlp_layernorm.weight"] = model_params[
            "decoder.blocks." + str(i) + ".mlp_ln.weight"
        ]
        weights[f"{trtllm_layer_name_prefix}.mlp_layernorm.bias"] = model_params[
            "decoder.blocks." + str(i) + ".mlp_ln.bias"
        ]

    weights["final_layernorm.weight"] = model_params["decoder.ln.weight"]
    weights["final_layernorm.bias"] = model_params["decoder.ln.bias"]

    return weight_only_quantize_dict(weights, quant_algo=quant_algo, plugin=True)  # type: ignore


def convert_checkpoints(args: TRTBuilderConfig) -> None:
    tik = time.time()

    assert args.model_dir is not None, "Model directory must be specified."
    assert args.checkpoint_dir is not None, "Checkpoint directory must be specified."
    assert args.model_name is not None, "Model name must be specified."

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    quant_algo = None
    plugin_weight_only_quant_type = None
    if args.use_weight_only and args.weight_only_precision == "int8":
        plugin_weight_only_quant_type = torch.int8
        quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == "int4":
        plugin_weight_only_quant_type = torch.quint4x2
        quant_algo = QuantAlgo.W4A16
    elif args.use_weight_only and args.weight_only_precision == "int4_gptq":
        quant_algo = QuantAlgo.W4A16_GPTQ

    model_path = os.path.join(args.model_dir, args.model_name + ".pt")
    assert os.path.exists(model_path), f"Model {model_path} does not exist."

    model = torch.load(model_path, map_location="cpu")
    logger.info(f"Loaded model from {model_path}")
    model_metadata = model["dims"]
    model_state_dict = model["model_state_dict"]
    for param_tensor in model_state_dict:
        model_state_dict[param_tensor] = model_state_dict[param_tensor].half()

    def convert_and_save(component: str = "encoder"):
        # call get_encoder_config or get_decoder_config according to component
        if component == "encoder":
            config = get_encoder_config(model_metadata, args.dtype, quant_algo)  # type: ignore
        else:
            config = get_decoder_config(
                model_metadata, args.dtype, args.logits_dtype, quant_algo  # type: ignore
            )

        if args.use_weight_only and args.weight_only_precision == "int4_gptq":
            config["quantization"].update(
                {
                    "has_zero_point": True,
                }
            )

        assert (
            args.checkpoint_dir is not None
        ), "Checkpoint directory must be specified."

        component_save_dir = os.path.join(args.checkpoint_dir, component)
        if not os.path.exists(component_save_dir):
            os.makedirs(component_save_dir)

        with open(os.path.join(component_save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        if component == "encoder":
            weights = convert_openai_whisper_encoder(
                model_metadata, model_state_dict, quant_algo=quant_algo
            )
        else:
            assert component == "decoder"
            weights = convert_openai_whisper_decoder(
                model_metadata, model_state_dict, quant_algo=quant_algo
            )

        save_file(weights, os.path.join(component_save_dir, f"rank0.safetensors"))

    logger.info("Converting encoder checkpoints...")
    convert_and_save("encoder")
    logger.info("Converting decoder checkpoints...")
    convert_and_save("decoder")

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Total time of converting checkpoints: {t}")


def run_build(args: TRTBuilderConfig) -> None:
    assert args.output_dir is not None, "Output directory must be specified."

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    encoder_build_cmd = f"trtllm-build  \
        --checkpoint_dir {args.checkpoint_dir}/encoder \
        --output_dir {args.output_dir}/encoder \
        --moe_plugin disable \
        --max_batch_size {args.max_batch_size} \
        --max_input_len {args.max_input_len_enc} \
        --max_seq_len {args.max_input_len_enc} \
        --gemm_plugin disable "

    if args.use_bert_attention_plugin:
        encoder_build_cmd += (
            f"--bert_attention_plugin {args.use_bert_attention_plugin} "
        )
    if args.use_context_fmha_enc:
        encoder_build_cmd += "--use_fp8_context_fmha enable "
    if not args.remove_input_padding:
        encoder_build_cmd += "--remove_input_padding disable "
    else:
        encoder_build_cmd += "--remove_input_padding enable "

    if args.int8_kv_cache:
        encoder_build_cmd += "--int8_kv_cache enable "

    logger.info("Running encoder build command...")
    logger.debug(encoder_build_cmd)

    process = os.popen(encoder_build_cmd)
    out_logs = process.read().split("\n")
    return_code = process.close()
    if return_code is not None:
        # Non-zero return code indicates an error
        logger.error(f"Error running encoder build command. Exit code: {return_code}")
        for line in out_logs:
            logger.error(line)
        sys.exit(1)
    logger.info("Encoder build completed successfully.")

    decoder_build_cmd = f"trtllm-build  \
        --checkpoint_dir {args.checkpoint_dir}/decoder \
        --output_dir {args.output_dir}/decoder \
        --moe_plugin disable \
        --max_beam_width {args.max_beam_width} \
        --max_batch_size {args.max_batch_size} \
        --max_seq_len {args.max_output_len} \
        --max_input_len {args.max_input_len_dec} \
        --max_encoder_input_len {args.max_input_len_enc} "

    if args.use_gemm_plugin:
        decoder_build_cmd += f"--gemm_plugin {args.use_gemm_plugin} "
    if args.use_bert_attention_plugin:
        decoder_build_cmd += (
            f"--bert_attention_plugin {args.use_bert_attention_plugin} "
        )
    if args.use_gpt_attention_plugin:
        decoder_build_cmd += f"--gpt_attention_plugin {args.use_gpt_attention_plugin} "
    if args.use_context_fmha_dec:
        decoder_build_cmd += "--use_fp8_context_fmha enable "
    if not args.remove_input_padding:
        decoder_build_cmd += "--remove_input_padding disable "
    else:
        decoder_build_cmd += "--remove_input_padding enable "

    if args.int8_kv_cache:
        decoder_build_cmd += "--int8_kv_cache enable "

    if args.remove_input_padding:  # If using C++ runtime configuration
        decoder_build_cmd += "--paged_kv_cache enable "

    logger.info("Running decoder build command...")
    logger.debug(decoder_build_cmd)
    process = os.popen(decoder_build_cmd)
    out_logs = process.read().split("\n")
    return_code = process.close()
    if return_code is not None:
        # Non-zero return code indicates an error
        logger.error(f"Error running decoder build command. Exit code: {return_code}")
        for line in out_logs:
            logger.error(line)
        sys.exit(1)
    logger.info("Decoder build completed successfully.")


def run(args: TRTBuilderConfig) -> None:
    convert_checkpoints(args)
    run_build(args)


if __name__ == "__main__":
    args = parse_arguments()

    trt_build_args = load_trt_build_config(args.output_dir)

    logger.info("[TRTBuilderConfig]:")
    logger.info(json.dumps(vars(trt_build_args), indent=4, default=str))

    run(args=trt_build_args)
