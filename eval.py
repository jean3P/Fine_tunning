import argparse
import csv
import json
import os
from pathlib import Path

import evaluate
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Collator, OcrDataset
from model import load_model_and_tokenizer_inference


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    return {
        "precision": 0.0 if precision is None else precision["precision"],
        "recall": 0.0 if recall is None else recall["recall"],
        "f1-score": 0.0 if f1 is None else f1["f1"],
        "accuracy": 0.0 if accuracy is None else accuracy["accuracy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an LLM model with PEFT")
    parser.add_argument(
        "-d",
        "--data",
        dest="data",
        type=str,
        required=True,
        help="Path to dataset to be evaluated",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        type=str,
        required=True,
        help="Path or name of the model checkpoint",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        type=Path,
        help="Output directory to save the results",
    )
    parser.add_argument(
        "--max-len",
        dest="max_len",
        type=int,
        default=512,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--set-pad-token",
        dest="set_pad_token",
        nargs="?",
        const=True,
        help=(
            "Set the padding token, needed by models such as Mistral-7B. "
            "Optionaly, accepts a custom padding token, "
            "otherwise the unk/eos token is used."
        ),
    )
    parser.add_argument(
        "--split-model",
        dest="split_model",
        action="store_true",
        help=(
            "Split the model across available GPUs. Not allowed when launching "
            "the script distributed, as that would be conflicting."
        ),
    )
    parser.add_argument(
        "--no-qlora",
        dest="no_qlora",
        action="store_true",
        help="Disable QLoRA (quantisation of the model)",
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "-p",
        "--prefix",
        dest="prefix",
        type=str,
        default="test_",
        help="Prefix for the files to be used in the data directory",
    )
    parser.add_argument(
        "-e",
        "--extension",
        dest="extension",
        type=str,
        default="_aws.json",
        help="Extension for the files to be used in the data directory",
    )
    parser.add_argument(
        "--response-template",
        dest="response_template",
        type=str,
        default="### Classification:",
        help="Template to use for the response. [Default: ### Classification:]",
    )
    parser.add_argument(
        "-g",
        "--generative",
        dest="generative",
        action="store_true",
        help="Use a generative model rather than a classifier on top",
    )
    parser.add_argument(
        "--unsloth",
        dest="unsloth",
        action="store_true",
        help="Use unsloth for training (only supports generative)",
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=1234, help="Random seed"
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model, tokenizer = load_model_and_tokenizer_inference(
        args.checkpoint,
        qlora=not args.no_qlora,
        set_pad_token=args.set_pad_token,
        device_map="auto" if args.split_model else None,
        generative=args.generative,
        unsloth=args.unsloth,
    )

    dataset = OcrDataset(
        args.data,
        tokenizer=tokenizer,
        max_length=args.max_len,
        truncation=True,
        prefix=args.prefix,
        extension=args.extension,
        inference=True,
        response_template=args.response_template,
        generative=args.generative,
    )
    print(f"Dataset with {len(dataset)} samples")

    collator = Collator(
        tokenizer=tokenizer,
        response_template=dataset.response_template,
        generative=dataset.is_generative,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=mp.cpu_count(),
        pin_memory=True,
    )

    if args.out_dir:
        cp_name = Path(args.checkpoint).parent.name
        out_dir = args.out_dir / f"{args.max_len}-max-len" / cp_name
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv_fd = open(out_dir / "predictions.tsv", "w", encoding="utf-8")
        writer = csv.writer(tsv_fd, delimiter="\t")
    else:
        tsv_fd = None
        writer = None
        out_dir = None

    predictions = []
    labels = []
    pbar = tqdm(desc="Evaluating", total=len(dataset), leave=False)

    for batch in data_loader:
        batch_labels = batch.data.pop("labels")
        batch_size = batch_labels.size(0)
        if args.generative:
            output = model.generate(
                input_ids=batch.data["input_ids"].to(model.device),
                attention_mask=batch.data["attention_mask"].to(model.device),
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id,
            )
            pred = []
            for i, (out, input_ids) in enumerate(zip(output, batch.data["input_ids"])):
                pred_str = tokenizer.decode(
                    out[input_ids.size(0) :], skip_special_tokens=True
                )
                # Remove leading and trailing whitespace.
                pred_str = pred_str.strip()
                if writer is not None:
                    writer.writerow(
                        [batch.info["id"][i], pred_str, batch.info["class_name"][i]]
                    )
                pred.append(pred_str.lower() == batch.info["class_name"][i].lower())
            # Fake labels, but just whether it's correct or not.
            pred_t = torch.tensor(pred, dtype=torch.long)
            predictions.append(pred_t)
            labels.append(torch.ones_like(pred_t))
        else:
            output = model(**batch)
            _, pred = torch.max(output["logits"], dim=-1)
            predictions.append(pred.cpu())
            labels.append(batch_labels.cpu())
        pbar.update(batch_size)
    pbar.close()

    # print("### Preds")
    # print(torch.cat(predictions))
    # print("### Labels")
    # print(torch.cat(labels))
    metrics = compute_metrics(torch.cat(predictions), torch.cat(labels))
    print(metrics)

    if out_dir:
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as fd:
            json.dump(metrics, fd)

    if tsv_fd is not None:
        tsv_fd.close()


if __name__ == "__main__":
    main()
