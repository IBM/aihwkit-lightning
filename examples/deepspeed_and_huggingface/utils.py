from typing import Union, Dict, Any
import yaml
import argparse
import torch
from transformers import Trainer
from aihwkit_lightning.simulator.configs.configs import TorchInferenceRPUConfig
from aihwkit_lightning.simulator.parameters.enums import (
    WeightModifierType,
    WeightClipType,
)


class CustomTrainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    # overwriting for clipping the weights
    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
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
        raise Exception("Clip type not supported")
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
        raise Exception("Unknown modifier type")
    rpu_config.modifier.type = modifier_type

    rpu_config.mapping.max_input_size = args.mapping_max_input_size

    rpu_config.pre_post.input_range.enable = args.input_range_enable
    rpu_config.pre_post.input_range.learn_input_range = (
        args.input_range_learn_input_range
    )
    rpu_config.pre_post.input_range.init_value = args.input_range_init_value
    rpu_config.pre_post.input_range.fast_mode = args.input_range_fast_mode
    rpu_config.pre_post.input_range.init_with_max = args.input_range_init_with_max
    rpu_config.pre_post.input_range.init_from_data = args.input_range_init_from_data
    rpu_config.pre_post.input_range.init_std_alpha = args.input_range_init_std_alpha
    rpu_config.pre_post.input_range.decay = args.input_range_decay
    rpu_config.pre_post.input_range.input_min_percentage = (
        args.input_range_input_min_percentage
    )
    return rpu_config


class PrettySafeLoader(yaml.SafeLoader):
    """Allows specifying tuples in yaml config."""

    def construct_python_tuple(self, node):
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
    for k, v in args.__dict__.items():
        if k == "lr" and not isinstance(v, list):
            args.lr = float(args.lr)
        if isinstance(v, str) and "lambda" in v:
            setattr(args, k, eval(v))
        elif isinstance(v, list):
            new_l = []
            for el in v:
                if isinstance(el, str) and "lambda" in el:
                    new_l.append(eval(el))
                elif k == "lr":
                    new_l.append(float(el))
                else:
                    new_l.append(el)
            setattr(args, k, new_l)
    return args


def get_args():
    args = create_parser()
    args = parse_args(args)
    args = check_and_eval_args(args)
    return args


def create_parser():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Training Args")
    g.add_argument("--config", dest="config", type=argparse.FileType(mode="r"))
    return parser


def maybe_inf2float(val):
    """If it is a string with inf,-inf, return float"""
    if isinstance(val, str) and (val == "inf" or val == "-inf"):
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
