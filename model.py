from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import torch
from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

try:
    # Ignore type errors, because unsloth is optional and not as easily installed.
    from unsloth import FastLanguageModel  # type: ignore[reportMissingImports]

    HAS_UNSLOTH = True
except ImportError:
    FastLanguageModel = None
    HAS_UNSLOTH = False

LoraKind = Literal["lora", "qlora"]


def create_model_and_tokenizer(
    name_or_path: str,
    num_labels: int = 16,
    rank: int = 4,
    alpha: int = 8,
    lora_dropout: float = 0.1,
    lora_modules: str | list[str] = "all-linear",
    lora_kind: Optional[LoraKind] = "qlora",
    set_pad_token: bool | str | None = False,
    device_map: Optional[dict | str] = None,
    generative: bool = False,
    unsloth: bool = False,
) -> tuple[PreTrainedModel | PeftModel | PeftMixedModel, PreTrainedTokenizerBase]:
    if unsloth:
        assert FastLanguageModel, (
            "unsloth is not installed, see the instructions on: "
            "https://github.com/unslothai/unsloth"
        )
        assert generative, "unsloth is only supported for generative models"
        model, tokenizer = FastLanguageModel.from_pretrained(
            name_or_path,
            load_in_4bit=lora_kind == "qlora",
            device_map=device_map,  # type: ignore
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0,  # For now 0, because it's optimised.
            # unsloth doesn't support all-linear, but for now just hardcode all
            # possible linear layers for the model they support.
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            if lora_modules == "all-linear"
            else lora_modules,
        )
        if isinstance(set_pad_token, str):
            tokenizer.pad_token = set_pad_token
        return model, tokenizer

    tokenizer = create_tokenizer(
        name_or_path, set_pad_token=set_pad_token, generative=generative
    )

    model_kwargs = dict(
        pretrained_model_name_or_path=name_or_path,
        device_map=device_map,
        offload_folder="offload",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if lora_kind == "qlora"
        else None,
        trust_remote_code=True,
        use_cache=False,
    )
    if generative:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            num_labels=num_labels,
            **model_kwargs,
        )
    if set_pad_token:
        model.config.pad_token_id = tokenizer.pad_token_id

    if lora_kind is not None:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM if generative else TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_modules,
        )
        model = get_peft_model(model, peft_config)
        print(f"Model with {lora_kind}:")
        model.print_trainable_parameters()

    if lora_kind == "qlora":
        # Put the model completely onto bfloat16, even if it's not supported by the GPU,
        # otherwise the dequantisation might not work correctly, resulting in an
        # overflow. It's easier to just keep it entirely in bfloat16 as a direct
        # replacement of float32 (so no mixed-precision). It might be a bit slower but
        # uses a lot less memory, and a bigger batch size means the dequantisation needs
        # to be calculated fewer times.
        model = model.to(torch.bfloat16)

    if model.device.type != "cuda":
        model = model.to("cuda")

    return model, tokenizer


# Load the model for evaluation, this is different from creating the model for training,
# since the LoRA adapters don't need to be added, as they are already included.
# Additionally, here it can't be converted to bfloat16, as it will give an error to
# leave them in their original format.
def load_model_and_tokenizer_inference(
    path: str,
    num_labels: int = 16,
    qlora: bool = True,
    set_pad_token: bool | str | None = False,
    device_map: Optional[dict | str] = None,
    generative: bool = False,
    unsloth: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if unsloth:
        assert FastLanguageModel, (
            "unsloth is not installed, see the instructions on: "
            "https://github.com/unslothai/unsloth"
        )
        assert generative, "unsloth is only supported for generative models"
        model, tokenizer = FastLanguageModel.from_pretrained(
            path,
            device_map=device_map,  # type: ignore
        )
        model = model.eval()
        # Not sure if that is really needed, but might as well keep it.
        FastLanguageModel.for_inference(model)
        # Left side padding for inference
        tokenizer.padding_side = "left"
        if isinstance(set_pad_token, str):
            tokenizer.pad_token = set_pad_token
        return model, tokenizer
    model_kwargs = dict(
        pretrained_model_name_or_path=path,
        device_map=device_map,
        offload_folder="offload",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if qlora
        else None,
        trust_remote_code=True,
        use_cache=False,
    )
    if generative:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            num_labels=num_labels,
            **model_kwargs,
        )
    if model.device.type != "cuda":
        model = model.to("cuda")
    model = model.eval()

    tokenizer_name_or_path = model.name_or_path
    # If there is a config.json file in the model checkpoint, it is not a LoRA model,
    # and because the tokenizer is not saved alongside the model during training, the
    # base model's tokenizer needs to be used. But apparatenly that is not stored
    # anywhere in the loaded model. At least it can be found in the config, so read it
    # from there.
    config_path = Path(tokenizer_name_or_path) / "config.json"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as fd:
            config = json.load(fd)
        tokenizer_name_or_path = config["_name_or_path"]

    tokenizer = create_tokenizer(tokenizer_name_or_path, set_pad_token=set_pad_token)
    if set_pad_token:
        model.config.pad_token_id = tokenizer.pad_token_id
    if generative:
        # Left side padding for inference
        tokenizer.padding_side = "left"
    return model, tokenizer


def create_tokenizer(
    name_or_path: str,
    set_pad_token: bool | str | None = False,
    generative: bool = False,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        add_prefix_space=True,
    )
    if set_pad_token:
        if isinstance(set_pad_token, str):
            tokenizer.pad_token = set_pad_token
        else:
            tokenizer.pad_token = (
                tokenizer.unk_token if generative else tokenizer.eos_token
            )
    return tokenizer
