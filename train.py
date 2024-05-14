import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import accelerate
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from data import HfCollator, OcrDataset
from eval import compute_metrics
from model import create_model_and_tokenizer

# Which models to save at what point.
# end is used if there should be no checkpoints saved after each epoch, but only at the
# very end. Useful when saving the model takes unnecessary time as it is guaranteed to
# finish the training very quickly.
SaveFrequency = Literal["never", "end", "epoch"]


def classification_from_generation(
    preds: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a "classification" for each prediction from the generation output.

    It doesn't actually classify it by their class, but simply creates
    prediction / label pairs which are either correct or not.
    Just that it can be used to calculate the other metrics.
    """
    # This needs to be done per sample in the batch, because they may have different
    # output lengths (tokens).
    # Just checks whether the output is excatly the same.
    correct = []
    for pred, label in zip(preds, labels):
        active_tokens = label != ignore_index
        pred = pred[active_tokens]
        label = label[active_tokens]
        correct.append(torch.equal(pred, label))
    out = torch.tensor(correct, dtype=torch.long)
    out_label = torch.ones_like(out)
    return out, out_label


class ClassificationTrainer(Trainer):
    def __init__(self, *args, generative: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_generative = generative
        self.train_stats = dict(loss=[], predictions=[], labels=[])

    def compute_loss(
        self, model, inputs: dict[str, torch.Tensor], return_outputs: bool = False
    ):
        if self._is_generative:
            # TODO: should probably to this for the classifier as well,
            # since there is label smoothing for the loss.
            outputs = model(**inputs)
            loss = torch.mean(outputs["loss"])
            labels = inputs["labels"]
            _, pred = torch.max(outputs["logits"], dim=-1)
            # TODO: Real class predictions
            pred, labels = classification_from_generation(pred, labels)
        else:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = F.cross_entropy(logits, labels)
            _, pred = torch.max(logits, dim=-1)
        self.train_stats["loss"].append(loss.item())
        self.train_stats["predictions"].append(pred.cpu())
        self.train_stats["labels"].append(labels.cpu())
        return (loss, outputs) if return_outputs else loss


class CustomCallback(TrainerCallback):
    """
    Custom callback to improve certain behaviours of the training loop.


    Compute the metrics on the training set at the end of the epoch, whose results are
    accumulated over the iterations. This is done because it avoids having to rerun the
    evaluation on the training set as there is no difference between inference and
    training for the classification. Furthermore, it allows to see the average across
    the iterations rather than just the snapshot at the end of the epoch. Much faster.

    Save the model checkpoints based on the frequency given:
        - never: Don't save any checkpoint.
        - end: Only save the model at the very end of training.
        - epoch: Save the latest and best model after each epoch
    """

    def __init__(
        self,
        trainer: ClassificationTrainer,
        save_freq: SaveFrequency = "epoch",
        generative: bool = False,
    ) -> None:
        super().__init__()
        self._trainer = trainer
        self._save_freq = save_freq
        self._is_generative = generative

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        train_stats = self._trainer.train_stats
        self._trainer.train_stats = dict(loss=[], predictions=[], labels=[])
        loss = torch.mean(torch.tensor(train_stats["loss"], dtype=torch.float)).item()
        preds = torch.cat(train_stats["predictions"])
        labels = torch.cat(train_stats["labels"])
        train_metrics = compute_metrics(preds, labels)
        train_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
        self._trainer.log({"train_loss": loss, **train_metrics})

        save_end = self._save_freq == "end" and state.global_step == state.max_steps
        control.should_save = self._save_freq == "epoch" or save_end


# This is just for the trainer which gives a tuple of the evaluation,
# being logits and labels as NumPy arrays.
def compute_metrics_for_trainer(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return compute_metrics(preds, labels)


def compute_metrics_for_trainer_generative(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    preds, labels = classification_from_generation(
        torch.from_numpy(labels), torch.from_numpy(labels)
    )
    return compute_metrics(preds, labels)


def parse_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with PEFT")
    parser.add_argument(
        "-d",
        "--data",
        dest="data",
        type=str,
        required=True,
        help="Path to OCR dataset",
    )
    parser.add_argument(
        "--name",
        dest="name",
        type=str,
        required=True,
        help="Name of the experiment for the logging and saved model",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        default=None,
        required=True,
        help="Name of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "-g",
        "--generative",
        dest="generative",
        action="store_true",
        help="Use generative training rather than a classifier on top",
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
            "the training script distributed, as that would be conflicting."
        ),
    )
    parser.add_argument(
        "-l",
        "--lr",
        dest="lr",
        type=float,
        default=1e-4,
        help="Learning rate for training",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        dest="num_epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=0.1,
        help="Weight decay",
    )
    parser.add_argument(
        "-r", "--lora-rank", dest="lora_rank", type=int, default=4, help="Lora rank"
    )
    parser.add_argument(
        "-a",
        "--lora-alpha",
        dest="lora_alpha",
        type=float,
        default=8,
        help="Lora alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        dest="lora_dropout",
        type=float,
        default=0.2,
        help="Lora dropout",
    )
    parser.add_argument(
        "--trainable",
        dest="trainable",
        type=str,
        default="qlora",
        choices=["classifier", "lora", "qlora", "full"],
        help=(
            "What parts of the model are trained. classifier just trains "
            "the classifier that is added on top of the base model, "
            "lora/qlora additionaly adds the LoRA adapters to each layer "
            "of the base model, and full fine-tunes the every parameter of the model"
        ),
    )
    parser.add_argument(
        "--mixed-precision",
        dest="mixed_precision",
        type=str,
        nargs="?",
        const="auto",
        choices=["fp16", "bf16", "auto"],
        help=(
            "Run in mixed-precision. If the flag is given without an explicit argument "
            "it will use auto, which will use bf16 if the GPU supports it, "
            "and otherwise fallback to fp16."
        ),
    )
    parser.add_argument(
        "--save",
        dest="save_freq",
        default="epoch",
        choices=["never", "end", "epoch"],
        help=(
            "When to save the model checkpoints. Either never, only at the end of "
            "training or at each epoch the latest and the best models are saved. "
        ),
    )
    parser.add_argument(
        "--unsloth",
        dest="unsloth",
        action="store_true",
        help="Use unsloth for training (only supports generative)",
    )
    parser.add_argument(
        "--prefix-train",
        dest="prefix_train",
        type=str,
        default="train_",
        help="Prefix for the files to be used for training in the data directory",
    )
    parser.add_argument(
        "--prefix-validation",
        dest="prefix_validation",
        type=str,
        default="validation_",
        help="Prefix for the files to be used for validation in the data directory",
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
        "--seed", dest="seed", type=int, default=1234, help="Random seed"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    os.environ["WANDB_PROJECT"] = "rvl_cdip_paper"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create the accelerator here, because it initialises the distributed process group,
    # if that is not done before the model is created, it will end up on the same
    # GPU for all processes.
    accelerator = accelerate.Accelerator()
    num_processes = getattr(accelerator, "num_processes", 1)
    model, tokenizer = create_model_and_tokenizer(
        args.model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # The kind corresponds to the trainable option if LoRA is used.
        lora_kind=args.trainable if args.trainable in ["lora", "qlora"] else None,
        set_pad_token=args.set_pad_token,
        device_map="auto" if args.split_model else None,
        generative=args.generative,
        unsloth=args.unsloth,
    )
    if args.trainable == "classifier":
        for name, param in model.named_parameters():
            # Freeze all parameters
            param.requires_grad_(False)
        # Make the classifier trainable.
        # For LLMs this is the score module, but for others it's usually classifier
        if hasattr(model, "score"):
            model.score.requires_grad_(True)
        if hasattr(model, "classifier"):
            model.classifier.requires_grad_(True)

    train_dataset = OcrDataset(
        args.data,
        tokenizer=tokenizer,
        max_length=args.max_len,
        truncation=True,
        prefix=args.prefix_train,
        extension=args.extension,
        response_template=args.response_template,
        generative=args.generative,
    )
    validation_dataset = OcrDataset(
        args.data,
        tokenizer=tokenizer,
        max_length=args.max_len,
        truncation=True,
        prefix=args.prefix_validation,
        extension=args.extension,
        response_template=args.response_template,
        generative=args.generative,
    )

    collator = HfCollator(
        tokenizer=tokenizer,
        response_template=train_dataset.response_template,
        generative=train_dataset.is_generative,
    )

    # Use BF16 mixed-precision training (not bfloat16 as such)
    use_bf16 = args.mixed_precision == "bf16" or (
        args.mixed_precision == "auto" and torch.cuda.is_bf16_supported()
    )
    training_args = TrainingArguments(
        output_dir=Path("log") / args.name,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        # Halve the batch size for eval during generative training, because it somehow
        # uses quite a bit more and can easily OOM, even though it would usually be the
        # opposite.
        per_device_eval_batch_size=args.batch_size // 2
        if args.generative
        else args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # Only keep 2 checkpoints (last and best)
        save_total_limit=2,
        load_best_model_at_end=True,
        # Gradient checkpoint can reduce memory usage drastically, but increases runtime
        # by a lot. Roughly 2-4x slower, but at least you could fit in larger context
        # sizes or use bigger batches. But by default it's disabled.
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs=dict(use_reentrant=False),
        # These are disabled on purpos, because we want to avoid mixed-precision
        # training and just use bfloat16 as a drop-in replacement for float32.
        fp16=args.mixed_precision and not use_bf16,
        bf16=args.mixed_precision and use_bf16,
        report_to=["wandb"],
        run_name=args.name,
        max_grad_norm=0.3,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=mp.cpu_count() // num_processes,
        dataloader_persistent_workers=True,
        # Move the evaluation results of the generative model immediately to CPU, rather
        # than waiting. This is slower, but otherwise it will run out of memory.
        eval_accumulation_steps=1 if args.generative else None,
    )
    # HACK: Restrict the trainer to a single GPU per process unless it's split across
    # GPU. For some reason it just occupies the others when they are unused, but then
    # the combined batch size is used on a single one, making it crash.
    if not args.split_model and num_processes == 1:
        training_args._n_gpu = num_processes

    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_for_trainer_generative
        if args.generative
        else compute_metrics_for_trainer,
        generative=args.generative,
    )
    trainer.add_callback(CustomCallback(trainer, save_freq=args.save_freq))
    trainer.train()


if __name__ == "__main__":
    main()
