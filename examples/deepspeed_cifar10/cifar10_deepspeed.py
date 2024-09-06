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
# NOTE: This is a modified version of the DeepSpeed example from here
# https://github.com/microsoft/DeepSpeedExamples/tree/master/training/cifar

"""Example on how to use AIHWKIT-Lightning with just DeepSpeed."""


import os
import argparse
from argparse import Namespace

from torch import nn, distributed, utils, no_grad, manual_seed, bfloat16, float16
from torch import max as torch_max
import torch.nn.functional as F
from torchvision import transforms, datasets
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger

from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.simulator.configs import (
    TorchInferenceRPUConfig,
    WeightClipType,
    WeightModifierType,
)

logger.setLevel("WARNING")


def add_argument() -> Namespace:
    """Add the argument parser for the DeepSpeed example."""
    parser = argparse.ArgumentParser(description="CIFAR")

    # For train.
    parser.add_argument(
        "--use-triton",
        default=False,
        action="store_true",
        help="use triton or not in AIHWKIT-Lightning",
    )
    parser.add_argument(
        "-e", "--epochs", default=30, type=int, help="number of total epochs (default: 30)"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local rank passed from distributed launcher"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )

    # For mixed precision training.
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="Datatype used for training",
    )

    # For ZeRO Optimization.
    parser.add_argument(
        "--stage", default=1, type=int, choices=[0, 1, 2, 3], help="Datatype used for training"
    )

    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    parsed_args = parser.parse_args()

    return parsed_args


def get_ds_config(command_args: Namespace) -> dict:
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": 16,
        "steps_per_print": command_args.log_interval,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 0.001, "betas": [0.8, 0.999], "eps": 1e-8, "weight_decay": 3e-7},
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 0.001, "warmup_num_steps": 1000},
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": command_args.dtype == "bf16"},
        "fp16": {
            "enabled": command_args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "auto_cast": True,
            "loss_scale_window": 1000,
            "hysteresis": 1,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": command_args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }
    return ds_config


class Net(nn.Module):
    """Define the Convolution Neural Network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass of the network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(model_engine, testset, local_device, target_dtype, test_batch_size=4):
    """Test the network on the test data.

    Args:
        model_engine (deepspeed.runtime.engine.DeepSpeedEngine): the DeepSpeed engine.
        testset (torch.utils.data.Dataset): the test dataset.
        local_device (str): the local device name.
        target_dtype (torch.dtype): the target datatype for the test data.
        test_batch_size (int): the test batch size.
    """
    # pylint: disable=too-many-locals

    # The 10 classes for CIFAR10.
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    # Define the test dataloader.
    testloader = utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=0
    )

    # For total accuracy.
    correct, total = 0, 0
    # For accuracy per class.
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    # Start testing.
    model_engine.eval()
    with no_grad():
        for data in testloader:
            images, labels = data
            if target_dtype is not None:
                images = images.to(target_dtype)
            outputs = model_engine(images.to(local_device))
            _, predicted = torch_max(outputs.data, 1)
            # Count the total accuracy.
            total += labels.size(0)
            correct += (predicted == labels.to(local_device)).sum().item()

            # Count the accuracy per class.
            batch_correct = (predicted == labels.to(local_device)).squeeze()
            for i in range(test_batch_size):
                label = labels[i]
                class_correct[label] += batch_correct[i].item()
                class_total[label] += 1

    if model_engine.local_rank == 0:
        print(f"Accuracy of the network on the {total} test images: {100*correct/total:.0f}%")

        # For all classes, print the accuracy.
        for i in range(10):
            print(f"Accuracy of {classes[i]:>5s}: {100*class_correct[i]/class_total[i]:2.0f}%")


def main(args):
    """Main function for the DeepSpeed example."""
    # pylint: disable=too-many-locals, too-many-statements

    manual_seed(0)
    # Initialize DeepSpeed distributed backend.
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

    ########################################################################
    # Step1. Data Preparation.
    #
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    #
    # Note:
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.
    ########################################################################
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    if distributed.get_rank() != 0:
        # Might be downloading cifar data, let rank 0 download first.
        distributed.barrier()

    # Load or download cifar data.
    data_path = os.path.expanduser("~/scratch/aihwkit-lightning-example/")
    trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    if distributed.get_rank() == 0:
        # Cifar data is downloaded, indicate other ranks can proceed.
        distributed.barrier()

    ########################################################################
    # Step 2. Define the network with DeepSpeed.
    #
    # First, we define a Convolution Neural Network.
    # Then, we define the DeepSpeed configuration dictionary and use it to
    # initialize the DeepSpeed engine.
    ########################################################################
    net = Net()

    rpu_config = TorchInferenceRPUConfig()
    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.init_from_data = 100
    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 2.5
    rpu_config.forward.inp_res = 254
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    rpu_config.modifier.std_dev = 0.023

    net = convert_to_analog(net, rpu_config=rpu_config)

    # Get list of parameters that require gradients.
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    # Initialize DeepSpeed to use the following features.
    #   1) Distributed model.
    #   2) Distributed data loader.
    #   3) DeepSpeed optimizer.
    ds_config = get_ds_config(args)
    model_engine, _, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=trainset, config=ds_config
    )

    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # For float32, target_dtype will be None so no datatype conversion needed.
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = float16

    # Define the Classification Cross-Entropy loss function.
    criterion = nn.CrossEntropyLoss()

    ########################################################################
    # Step 3. Train the network.
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize. (DeepSpeed handles the distributed details for us!)
    ########################################################################

    if args.use_triton:
        print("Using triton for AIHWKIT Lightning")
        os.environ["AIHWKIT_USE_TRITON"] = "1"

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # Get the inputs. ``data`` is a list of [inputs, labels].
            inputs, labels = data[0].to(local_device), data[1].to(local_device)

            # Try to convert to target_dtype if needed.
            if target_dtype is not None:
                inputs = inputs.to(target_dtype)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)

            # For DeepSpeed, we need to implement the clipping manually
            # this is what we changed. we clip the weights when we updated the parameters
            with no_grad():
                if hasattr(net, "analog_layers"):
                    for analog_layer in net.analog_layers():
                        analog_layer.clip_weights()

            model_engine.step()

            # Print statistics
            running_loss += loss.item()
            if local_rank == 0 and i % args.log_interval == (
                args.log_interval - 1
            ):  # Print every log_interval mini-batches.
                avg_loss = running_loss / args.log_interval
                print(f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {avg_loss : .3f}")
                running_loss = 0.0
    print("Finished Training")

    ########################################################################
    # Step 4. Test the network on the test data.
    ########################################################################
    test(model_engine, testset, local_device, target_dtype)


if __name__ == "__main__":
    arguments = add_argument()
    main(arguments)
