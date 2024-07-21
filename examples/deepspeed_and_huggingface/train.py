# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Example on how to use AIHWKIT-Lightning in single or multi-node training setting."""


import os
import glob
import shutil

from datetime import timedelta
import transformers
import accelerate.logging

import torch
from torch.optim import AdamW

from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import datasets
from datasets import load_dataset
from transformers import (
    Trainer,
    BertForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.utils.logging import enable_default_handler, enable_explicit_format

from utils import CustomTrainer, get_args, create_rpu_config
from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.optim import AnalogOptimizer


IS_CUDA = torch.cuda.is_available()


def main():
    # pylint: disable=missing-function-docstring, too-many-locals

    log_level = datasets.logging.INFO
    datasets.utils.logging.set_verbosity(log_level)
    transformers.logging.set_verbosity_info()
    accelerate.logging.get_logger(__name__, log_level="INFO")
    enable_default_handler()
    enable_explicit_format()

    # get args and set seed
    args = get_args()
    args.example_directory = os.path.expanduser(args.example_directory)
    args.ds_config_path = os.path.expanduser(args.ds_config_path)

    set_seed(args.seed)

    # Tell the Accelerator object to log with wandb
    # we specify 1801 because when 1800 is used it is overriden.
    # This was a bug in accelerate
    use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1801))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[kwargs, ddp_kwargs])
    accelerator.init_trackers(
        project_name="AIHWKIT Lightning", init_kwargs={"wandb": {"name": args.run_name}}
    )

    with accelerator.main_process_first():
        # load the model
        example_dir = os.path.expanduser(args.example_directory)
        model = BertForMaskedLM.from_pretrained(
            "google-bert/bert-base-uncased", cache_dir=example_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-uncased", cache_dir=example_dir
        )
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=example_dir)
        dataset["train"] = dataset["train"].select(list(range(1000)))

        def preprocess_function(examples):
            result = tokenizer(
                examples["text"], padding="max_length", max_length=512, truncation=True
            )
            return result

        # tokenize the pages in the dataset
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

        if not args.fp:
            # Convert the model into an anlog model
            rpu_config = create_rpu_config(args)
            model = convert_to_analog(model, rpu_config=rpu_config)

    # count the parameters
    param_count = 0
    for param in model.parameters():
        param_count += param.numel()
    print(f"Number of trainable parameters: {param_count:,}")

    metric_name = "loss"
    # we scale the learning rate linearly with the total number
    # of GPUs we train on
    lr = accelerator.num_processes * args.lr

    if not IS_CUDA:
        print("WARNING: Not using CUDA.")

    training_args = TrainingArguments(
        output_dir=args.example_directory,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        eval_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_safetensors=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type=args.lr_scheduler_type,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        report_to=args.report_to,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_grad_norm=args.max_grad_norm,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        data_seed=args.seed,
        metric_for_best_model=metric_name,
        load_best_model_at_end=args.load_best_model_at_end,
        warmup_ratio=args.warmup_ratio,
        bf16=False,
        bf16_full_eval=False,
        greater_is_better=args.greater_is_better,
        fp16=IS_CUDA,
        fp16_full_eval=IS_CUDA,
        torch_compile=False,
        deepspeed=args.ds_config_path if use_deepspeed else None,
    )

    # define trainer
    lm_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer_cls = CustomTrainer if use_deepspeed else Trainer
    # DeepSpeed uses it's own optimized optimizers
    optimizers = (
        None
        if use_deepspeed
        else (
            (
                AdamW(model.parameters(), lr=lr)
                if args.fp
                else AnalogOptimizer(AdamW, model.analog_layers(), model.parameters(), lr=lr)
            ),
            None,
        )
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=lm_data_collator,
        optimizers=optimizers,
    )

    # train the model and save it
    trainer.train()
    trainer.save_model(args.example_directory)

    eval_result = trainer.evaluate()
    if accelerator.is_main_process:
        print(f"Saving state...\nBest results on validation set are: \n{eval_result}")
    trainer.save_state()

    if accelerator.is_main_process:
        # Cleanup: remove checkpoint folders
        pattern = os.path.join(args.example_directory, "checkpoint-*")
        folders = glob.glob(pattern)
        for folder in folders:
            if os.path.isdir(folder):
                shutil.rmtree(folder)


if __name__ == "__main__":
    main()
