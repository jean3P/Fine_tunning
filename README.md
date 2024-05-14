# Fine-tuning LLMs for the Classification of RVL-CDIP


Using LLMs to classify the documents of [RVL-CDIP][rvl-cdip]. Training an LLM with a classifier head on top. Supports
training with [LoRA][lora] and [QLoRA][qlora].

Besides the classifier on top, there is also a generative training, where the model is conditioned to complete the class
name after a given classification directive, such as `### Classification: <class-name>`. This should be the preferred
method, as it does not destroy the capabilities of the LLM to produce any output, which may be unrelated to the trained
task.


## Table of Contents

<!-- vim-markdown-toc GitLab -->

* [Data](#data)
* [Requirements](#requirements)
    * [Unsloth](#unsloth)
* [Training](#training)
    * [Generative](#generative)
        * [Mistral (Generative)](#mistral-generative)
        * [Llama 3 (Generative)](#llama-3-generative)
    * [Multi-GPU (Data Parallelism)](#multi-gpu-data-parallelism)
    * [Split the Model across multiple GPUs (Model Parallelism)](#split-the-model-across-multiple-gpus-model-parallelism)
* [Evaluation](#evaluation)
    * [Evaluating Generative Models](#evaluating-generative-models)

<!-- vim-markdown-toc -->

## Data

A dedicated subset is used, for which the images were processed by [Amazon Textract][aws-ocr] OCR system to get a higher quality
text output compared to the original dataset. Each file is given as JSON where the extracted text is given as `"ocr"`.
Additionally, all files should have a prefix to which dataset split they belong, i.e. `train_`, `validation_` and
`test_`. Each file name contains the class (index) as the last number before the suffix. The suffix by default is
`_aws` and if it is changed, the data loading needs to be adapted to account for it.

The file structure is expected to be as follows:

```
rvl-cdip-ocr/
├── test_0000_10_aws.json
...
├── train_0000_13_aws.json
...
└── validation_0159_7_aws.json
```

## Requirements

All dependencies, with the exception of [`unsloth`][unsloth], can be installed with pip. While `unsloth` is optional, it
is indispensable for the training of the generative models (see below for more details).

```sh
pip install -r requirements.txt
```

On *Windows* the PyTorch packages may not be available on PyPi, hence you need to point to the official PyTorch
registry:

```sh
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Unsloth

**2-10x faster with 80% less memory (only supports generative training for a select few models).**

[`unsloth`][unsloth] is a library for efficient training of generative LLMs. Even though it only supports a handful of
models (Mistral, Llama 3, Phi 3 and Gemma at the moment), which are at least the very commonly used ones, it should
always be used whenever possible, since the memory reduction and training speed is a massive improvement. However, the
installation can cause some trouble, because you need a compatible PyTorch version as well as [`xformers`][xformers], so
here is the recommended setup:

- Python 3.11 (3.12 had issues installing `xformers`, but might now be okay for `torch>=2.3.0`)
- PyTorch 2.2.x
- CUDA 12.1 (default for PyTorch 2.2.x)

Installing the dependencies with `pip`:

```sh
# xformers for CUDA 12.1
pip install xformers --index-url https://download.pytorch.org/whl/cu121

# Current unsloth version from GitHub
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

If you'd prefer to install it through `conda`, follow the instructions from the README on [`xformers`][xformers] and
[`unsloth`][unsloth], respectively.

## Training

Training is done with the `train.py` script:

```sh
python train.py -d path/to/rvl-cdip-ocr/ --name name-of-experiment -m "mistralai/Mistral-7B-v0.1" -b 3 --set-pad-token --lora-rank 8 --lora-alpha 16 --max-len 768
```

- `--name` is used to give the experiment a name, which is used to store the checkpoints in `log/<name>` as well as the name of the experiment in [Weights & Biases][wandb].
- `-d` / `--data` is the path to the directory containing the OCR files, as described above.
- `-m` / `--model` specifies which model to use. Any HuggingFace compatible model can be loaded. Some models might need
  like Mistral-7B need the `--set-pad-token` option, as they don't have a dedicated padding token.
- `-b` / `--batch-size` determines the number of documents to use per batch.
- `--lora-rank` and `--lora-alpha` define the configurations for (Q)LoRA
- `--max-len` sets the maximum number of tokens that are used per document. Some documents may be larger than this, in
  which case they will be truncated.

You can choose between LoRA and QLoRA by specifying either `--trainable qlora` or `--trainable lora`, by default QLoRA
is used.

### Generative

To train the model in a generative manner, you simply need to add the `--generative` flag. In this mode,
a classification directive is added at the end of the text, by default `### Classification:` (can be
configured with `--response-template`), which is then followed by the class name, which the model needs to be predict.

**Important: This is only a *generative* training, not an instruction fine-tuning, and it does not learn to follow
instructions, but only *complete* the class name after the classification directive.**

Only tokens *after* the response template will contribute to the loss. This is achieved with
`DataCollatorForCompletionOnlyLM`, which needs the response template encoded in tokens to match exactly a part of the
input. That can be slightly tricky, because some tokenisers are affected differently by spaces before/after some other
tokens. For example, Mistral uses `--response-template "### Classification:"` (the default), because adding a space
before it, would ruin the tokenisation, where the full text would be different from the response template alone (since
the response follows after a newline). On the other hand, Llama 3 requires a space before it `--response-template " ### Classification:"` (*space after the opening quote*), otherwise it won't find the substring (as tokens) correctly.

*If you see a warning that there are no tokens contributing to the loss (and a long output with the full training text),
it is most likely due to the tokenisation issue with the used response template.*

It is strongly recommended to use `unsloth` for the generative training (see [the section about `unsloth`](#unsloth)),
which can be enabled with the `--unsloth` flag. Keep in mind, that only a handful models are supported, but if you are
using a supported one, make sure to use `unsloth`.

Finally, for `unsloth` you have to use `--mixed-precision`, which did not work correctly for the version without `unsloth`
because of some precision issues when using a GPU that only supports `fp16` instead of `bf16`.

#### Mistral (Generative)

Training Mistral can be done exactly as for the classifier model, by just adding `--generative` (and `--unsloth` with
`--mixed-precision`), everything else works by default. By using `unsloth` you can expect to increase the batch size on
a GPU with 24GB from 3 up to 40 (over 10x more).

```sh
python train.py -d path/to/rvl-cdip-ocr/ --name name-of-experiment -m "mistralai/Mistral-7B-v0.1" -b 40 --set-pad-token --lora-rank 8 --lora-alpha 16 --max-len 768 --generative --unsloth --mixed-precision
```

Note: You can use `unsloth/mistral-7b-bnb-4bit` in place of `mistralai/Mistral-7B-v0.1` when using `unsloth`, because
there is no need to load the full model when the 4bit quantised model is used for QLoRA. In fact, `unsloth` will
automatically switch to this to save disk space, so leaving it as is, makes it easier to switch to the classifier
version, as `unsloth` does not support the classifier models.

#### Llama 3 (Generative)

Llama 3 can be used in the same way as Mistral, but its tokeniser has some additional issues that need to be addressed.
Specifically, there is neither a padding token (same as with Mistral), but also no `<unk>` (unknown) token, so that one
cannot be used as a padding. It is not an option to use the `<eos>` (end of sequence) token as a padding token, because
then the model never learns to produce an `<eos>` token (since it counts as padding and does not contributed to the
loss), and the model during inference would continue generating tokens until the maximum output tokens are reached.
never learns

As a workaround, the `<|eot_id|>` (end of turn) can be repurposed as a padding token, since that is unused in the base
model and only useful for the instruct version. You can specify this token to the `--set-pad-token` option.
Additionally, as mentioned above, the response template needs to be changed to include a space at the beginning.


```sh
python train.py -d path/to/rvl-cdip-ocr/ --name name-of-experiment -m "unsloth/llama-3-8b" -b 16 --lora-rank 8 --lora-alpha 16 --max-len 768 --generative --unsloth --mixed-precision --set-pad-token "<|eot_id|>" --response-template " ### Classification:"
```

Note: Same here for the model, but `unsloth/llama-3-8b` is specified here, just because it does not require to have
a HuggingFace account for the authentication to access the Llama model (again 4bit is done automatically).

Llama 3 requires more memory than Mistral and therefore the batch size needs to be reduced. This is partly due to the
increased vocabulary size of the tokeniser, but also just seems to be generally the case that Mistral is more optimised
compared to all other models (at least in terms of memory).


### Multi-GPU (Data Parallelism)

*Prefer this whenever the model can be trained on a single GPU, but want to use multiple GPUs in parallel.*

Training with multiple GPUs can be achieved by using [`accelerate`][accelerate] when launching the script.

```sh
accelerate launch train.py -d path/to/rvl-cdip-ocr/ --name name-of-experiment -m "mistralai/Mistral-7B-v0.1" -b 3 --set-pad-token --lora-rank 8 --lora-alpha 16 --max-len 768
```

*Important: This loads the model on every GPU and runs them in parallel (Data parallelism), so the model needs to fit
into a single GPU.*

### Split the Model across multiple GPUs (Model Parallelism)

*Use for models that are too large for a single GPU.*

If the model does not fit into a single GPU (including memory needed for the gradients and optimiser), it can be split
across multiple GPU. For this the `--split-model` flag must be added.

```sh
python train.py -d path/to/rvl-cdip-ocr/ --name name-of-experiment -m "mistralai/Mistral-7B-v0.1" -b 3 --set-pad-token --lora-rank 8 --lora-alpha 16 --max-len 768 --split-model
```

*Important: Do **not** use this with `accelerate launch`, otherwise the model is split multiple times across multiple
GPUs, making it much slower while needing the same memory as if it were not split across GPUs*.

This uses `accelerate` as well with `device_map="auto"` to distribute the layers across GPUs (sequentially in chunks).
Because it is done per layer, one GPU will run through its chunk of layers and then the next continues from there,
meaning that only one GPU runs any computations at a given time and the other GPUs are idle during that time.
Unfortunately, this is much less efficient and therefore slower than data parallelism, but at least allows to train
models that wouldn't fit into memory otherwise.


## Evaluation

The evaluation of a model can be done with the `eval.py` script.

```sh
python eval.py -d path/to/rvl-cdip-ocr/ -c path/to/model/checkpoint/ -b 20 --max-len 768 -o results/
```

It is possible to set the maximum number of tokens used with `--max-len`, which allows to evaluate the model when giving
different sizes of the document.

Optionally,`-o`/`--out-dir` can be given to save the results in the specific directory. This requires the checkpoint to
be in a subdirectory of the model, e.g. `log/qlora-r4/checkpoint-dir`, because the name of the parent directory is used
to store the results in the output directory alongside with the maximum length specified for the evaluation, so that it
is easier to distinguish the results from multiple models and the maximum length that was used.

### Evaluating Generative Models

For generative models, the same options need to be specified as for the training, because that information is never
stored in the model, but would also allow to change them to different values, if desired.

For example, Llama 3 using `unsloth`:

```sh
python eval.py -d path/to/rvl-cdip-ocr/ -c path/to/llama3-8b/checkpoint/ -b 20 --max-len 768 -o results-generative/ --generative --unsloth --set-pad-token "<|eot_id|>" --response-template " ### Classification:"
```

*`unsloth` is also strongly recommended for the evaluation, which ended up being 10x faster on the large test set (more
than what they claimed).*

[accelerate]: https://github.com/huggingface/accelerate
[aws-ocr]: https://aws.amazon.com/textract/
[lora]: https://arxiv.org/abs/2106.09685
[qlora]: https://arxiv.org/abs/2305.14314
[rvl-cdip]: https://adamharley.com/rvl-cdip/
[unsloth]: https://github.com/unslothai/unsloth
[wandb]: https://wandb.ai/
[xformers]: https://github.com/facebookresearch/xformers
