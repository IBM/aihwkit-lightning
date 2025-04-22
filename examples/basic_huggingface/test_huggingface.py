# -*- coding: utf-8 -*-

# (C) Copyright 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-locals, too-many-public-methods, no-member
"""Example to show huggingface compatibility."""

import tempfile
import numpy as np
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from evaluate import load

from aihwkit_lightning.optim import AnalogOptimizer
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.nn.conversion import convert_to_analog


def training_huggingface(use_normal_torch: bool, use_fp16: bool):
    """Train a huggingface model."""

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli_matched": ("premise", "hypothesis"),
        "mnli_mismatched": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    task_name = "cola"
    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
    num_labels = 3 if task_name.startswith("mnli") else 1 if task_name == "stsb" else 2

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:

        model = RobertaForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-base", num_labels=num_labels, cache_dir=temp_dir
        )

        if not use_normal_torch:
            model = convert_to_analog(model, rpu_config=TorchInferenceRPUConfig())

        dataset = load_dataset("glue", task_name)
        sentence1_key, sentence2_key = task_to_keys[task_name]
        metric = load("glue", task_name)
        metric_name = (
            "pearson"
            if task_name == "stsb"
            else "matthews_correlation" if task_name == "cola" else "accuracy"
        )

        def preprocess_function(examples):
            if sentence2_key is None:
                return tokenizer(examples[sentence1_key], truncation=True)
            return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

        training_args = TrainingArguments(
            output_dir=temp_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            eval_strategy="steps",
            save_strategy="steps",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=0.00001,
            lr_scheduler_type="linear",
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-6,
            weight_decay=0.1,
            num_train_epochs=10,
            max_grad_norm=1.0,
            eval_steps=10,
            save_steps=100,
            save_total_limit=1,
            seed=0,
            metric_for_best_model=metric_name,
            warmup_ratio=0.06,
            bf16=False,
            report_to=None,
            greater_is_better=True,
            fp16=use_fp16,
            fp16_full_eval=use_fp16,
            half_precision_backend="apex",
            torch_compile=False,
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if task_name != "stsb":
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions[:, 0]
            return metric.compute(predictions=predictions, references=labels)

        validation_key = (
            "validation_mismatched"
            if task_name == "mnli_mismatched"
            else "validation_matched" if task_name == "mnli_matched" else "validation"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset[validation_key],
            tokenizer=tokenizer,
            optimizers=(
                (
                    AdamW(model.parameters(), lr=0.00001)
                    if use_normal_torch
                    else AnalogOptimizer(
                        AdamW, model.analog_layers(), model.parameters(), lr=0.00001
                    )
                ),
                None,
            ),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(temp_dir)


if __name__ == "__main__":
    training_huggingface(use_normal_torch=False, use_fp16=False)
