#!/bin/bash

export MAX_JOBS=8
deepspeed --master_port=29501 --bind_cores_to_rank cifar10_deepspeed.py --deepspeed $@