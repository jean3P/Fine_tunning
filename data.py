from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from trl import DataCollatorForCompletionOnlyLM

# Should probably be configurable, also could be a list, but why not dict.
CLASSES = {
    0: "letter",
    1: "form",
    2: "email",
    3: "handwritten",
    4: "advertisement",
    5: "scientific report",
    6: "scientific publication",
    7: "specification",
    8: "file folder",
    9: "news article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo",
}


@dataclass
class Sample:
    data: dict[str, torch.Tensor]
    info: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class Batch:
    data: dict[str, torch.Tensor]
    info: dict[str, list[Any]] = field(default_factory=lambda: {})


class Collator:
    """
    A simple collator that handles additional meta information besides the regular
    inputs. Depending on whether it is a generative model, the appropriate HuggingFace
    collator is used for all model inputs.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        response_template: str = "### Classification:",
        generative: bool = False,
    ):
        self.is_generative = generative
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.data_collator = (
            DataCollatorForCompletionOnlyLM(
                response_template,
                tokenizer=tokenizer,
            )
            if self.is_generative
            else DataCollatorWithPadding(tokenizer=tokenizer)
        )

    def __call__(self, samples: list[Sample]) -> Batch:
        info = {}
        for sample in samples:
            for key, value in sample.info.items():
                if key not in info:
                    info[key] = []
                info[key].append(value)
        return Batch(
            data=self.data_collator([sample.data for sample in samples]),
            info=info,
        )


# TODO: Get rid of this, because HuggingFace only allows dicts of tensors, but I need
# additional information.
class HfCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        response_template: str = "### Classification:",
        generative: bool = False,
    ):
        self.is_generative = generative
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.data_collator = (
            DataCollatorForCompletionOnlyLM(
                response_template,
                tokenizer=tokenizer,
            )
            if self.is_generative
            else DataCollatorWithPadding(tokenizer=tokenizer)
        )

    def __call__(self, samples: list[Sample]) -> dict[str, torch.Tensor]:
        return self.data_collator([sample.data for sample in samples])


class OcrDataset(Dataset):
    def __init__(
        self,
        dir: str | os.PathLike,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        truncation: bool = True,
        prefix: str = "",
        extension: str = "_aws.json",
        inference: bool = False,
        response_template: str = "### Classification:",
        generative: bool = False,
    ):
        self.dir = Path(dir)
        self.response_template = response_template
        self.inference = inference
        self.extension = extension
        self.tokenizer = tokenizer
        self.truncation = truncation
        self.max_length = max_length
        self.is_generative = generative
        self.files = list(glob.glob(str(self.dir / f"{prefix}*{extension}")))

    def encode_instruction(
        self, ocr_dict: dict, class_name: str | None = None
    ) -> torch.Tensor:
        """
        Encode the instructions such that only the text of OCR is truncated and the
        response is always provided fully. This is necessary to be done by hand.
        """
        # Constructing the input by hand, because the truncating should only happen
        # to the document, not the classification part.
        response = (
            f"\n\n {self.response_template} {class_name}"
            if class_name and not self.inference
            else f"\n\n {self.response_template}"
        )
        response_ids = torch.tensor(
            self.tokenizer.encode(response, add_special_tokens=False), dtype=torch.long
        )
        input_ids = torch.tensor(
            self.tokenizer.encode(
                ocr_dict["ocr"],
                truncation=self.truncation,
                max_length=self.max_length - len(response_ids) - 2,
                add_special_tokens=False,
            ),
            dtype=torch.long,
        )
        to_cat = [
            torch.tensor([self.tokenizer.bos_token_id]),
            input_ids,
            response_ids,
        ]
        # Only add the EOS token during the training, as in the inference, it should
        # generate the rest of the sequence until it produces an EOS.
        if not self.inference:
            to_cat.append(torch.tensor([self.tokenizer.eos_token_id]))
        input_ids = torch.cat(to_cat)
        return input_ids

    def __getitem__(self, i: int) -> Sample:
        path = self.files[i]
        with open(path, "r", encoding="utf-8") as fd:
            ocr_dict = json.load(fd)
        # Label is the last number in the file name
        class_label = int(Path(path).name.removesuffix(self.extension).split("_")[-1])
        class_name = CLASSES[class_label]

        if self.is_generative:
            input_ids = self.encode_instruction(ocr_dict, class_name=class_name)
        else:
            input_ids = torch.tensor(
                self.tokenizer.encode(
                    ocr_dict["ocr"],
                    truncation=self.truncation,
                    max_length=self.max_length,
                ),
                dtype=torch.long,
            )

        inputs = dict(
            input_ids=input_ids,
            # Attention always applies to every token.
            attention_mask=torch.ones_like(input_ids),
        )
        if not self.is_generative:
            inputs["label"] = torch.tensor(class_label)

        return Sample(
            data=inputs,
            info=dict(
                id=ocr_dict.get("id") or ocr_dict["path"],
                class_name=class_name,
                class_label=class_label,
            ),
        )

    def __len__(self) -> int:
        return len(self.files)
