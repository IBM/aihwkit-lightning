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

"""Utils for training example."""


from typing import Union, Dict, Any
import argparse
import yaml
import torch
from transformers import Trainer
from aihwkit_lightning.simulator.configs.configs import TorchInferenceRPUConfig
from aihwkit_lightning.simulator.parameters.enums import WeightModifierType, WeightClipType
from aihwkit_lightning.exceptions import ArgumentError


class CustomTrainer(Trainer):
    """Custom trainer handling weight clipping."""

    # overwriting for clipping the weights
    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model: The model to train.
            inputs: The inputs for for the training step.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        del inputs
        torch.cuda.empty_cache()
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.accelerator.deepspeed_engine_wrapped is not None:
            made_step = (
                self.accelerator.deepspeed_engine_wrapped.engine.is_gradient_accumulation_boundary()
            )
        else:
            made_step = True
        self.accelerator.backward(loss)
        # this is what we changed. we clip the weights when we updated the parameters
        with torch.no_grad():
            if made_step and hasattr(model, "analog_layers"):
                for analog_layer in model.analog_layers():
                    analog_layer.clip_weights()
        return loss.detach() / self.args.gradient_accumulation_steps


def create_rpu_config(args):
    """
    Create RPU config based on namespace.
    Args:
        args (Namespace): The namespace populated with fields.
    Returns:
        Union[InferenceRPUConfig,TorchInferenceRPUConfig]: The RPUConifg.
    Raises:
        ArgumentError: When wrong modes are passed.
    """

    rpu_config = TorchInferenceRPUConfig()

    rpu_config.forward.inp_res = args.forward_inp_res
    rpu_config.forward.out_noise = args.forward_out_noise
    rpu_config.forward.out_noise_per_channel = args.forward_out_noise_per_channel

    rpu_config.clip.sigma = args.clip_sigma
    if args.clip_type == "gaussian":
        clip_type = WeightClipType.LAYER_GAUSSIAN
    elif args.clip_type == "gaussian_channel":
        clip_type = WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL
    elif args.clip_type == "none":
        clip_type = WeightClipType.NONE
    else:
        raise ArgumentError("Clip type not supported")
    rpu_config.clip.type = clip_type

    rpu_config.modifier.std_dev = args.modifier_std_dev
    rpu_config.modifier.res = args.modifier_res
    rpu_config.modifier.enable_during_test = args.modifier_enable_during_test
    if args.modifier_type == "add_gauss":
        modifier_type = WeightModifierType.ADD_NORMAL
    elif args.modifier_type == "add_gauss_channel":
        modifier_type = WeightModifierType.ADD_NORMAL_PER_CHANNEL
    elif args.modifier_type == "none":
        modifier_type = WeightModifierType.NONE
    else:
        raise ArgumentError("Unknown modifier type")
    rpu_config.modifier.type = modifier_type

    rpu_config.mapping.max_input_size = args.mapping_max_input_size

    rpu_config.pre_post.input_range.enable = args.input_range_enable
    rpu_config.pre_post.input_range.learn_input_range = args.input_range_learn_input_range
    rpu_config.pre_post.input_range.init_value = args.input_range_init_value
    rpu_config.pre_post.input_range.fast_mode = args.input_range_fast_mode
    rpu_config.pre_post.input_range.init_with_max = args.input_range_init_with_max
    rpu_config.pre_post.input_range.init_from_data = args.input_range_init_from_data
    rpu_config.pre_post.input_range.init_std_alpha = args.input_range_init_std_alpha
    rpu_config.pre_post.input_range.decay = args.input_range_decay
    rpu_config.pre_post.input_range.input_min_percentage = args.input_range_input_min_percentage
    return rpu_config


class PrettySafeLoader(yaml.SafeLoader):
    """Allows specifying tuples in yaml config."""

    def construct_python_tuple(self, node):
        """Create tuple."""
        return tuple(self.construct_sequence(node))


def check_and_eval_args(args):
    """
    Goes through the args and checks if they are
    fulfilling some condiditons. Also, turns
    strings of lambda functions into lambda functions.

    Args:
        args (object): Parsed command line arguments.

    Returns:
        object: The modified command line arguments.
    """
    for key, value in args.__dict__.items():
        if key == "lr" and not isinstance(value, list):
            args.lr = float(args.lr)
        if isinstance(value, str) and "lambda" in value:
            setattr(args, key, eval(value))  # pylint: disable=eval-used
        elif isinstance(value, list):
            new_l = []
            for element in value:
                if isinstance(element, str) and "lambda" in element:
                    new_l.append(eval(element))  # pylint: disable=eval-used
                elif key == "lr":
                    new_l.append(float(element))
                else:
                    new_l.append(element)
            setattr(args, key, new_l)
    return args


def get_args():
    """Get the arguments and parse them."""
    args = create_parser()
    args = parse_args(args)
    args = check_and_eval_args(args)
    return args


def create_parser():
    """Create the parser that expects the config yaml."""
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Training Args")
    group.add_argument("--config", dest="config", type=argparse.FileType(mode="r"))
    return parser


def maybe_inf2float(val):
    """If it is a string with inf,-inf, return float"""
    if isinstance(val, str) and (val in ["inf", "-inf"]):
        print(f"WARNING: String {val} casted to float")
        return float(val)
    return val


def parse_args(parser: argparse.ArgumentParser):
    """
    Use all the parameters from default.yaml and then overrides some of them
    using the parameters found in the specified --config config.yaml
    Input:
        parser: a parser created with create_parser()
    Returns:
        args: parsed args from the .yaml file, adding the specified ones to the default
    """
    args = parser.parse_args()
    if hasattr(args, "config") and args.config:
        data = yaml.load(args.config, Loader=PrettySafeLoader)
        args.config = args.config.name
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    arg_dict[subkey] = maybe_inf2float(subvalue)
            else:
                arg_dict[key] = maybe_inf2float(value)
    return args
