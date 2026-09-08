# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Custom Exceptions for aihwkit."""


class AihwkitException(Exception):
    """Base class for exceptions related to aihwkit."""


class ModuleError(AihwkitException):
    """Exceptions related to analog neural network modules."""


class ArgumentError(AihwkitException):
    """Exceptions related to wrong arguments."""


class ConfigError(AihwkitException):
    """Exceptions related to tile configuration."""


class TorchTileConfigError(ConfigError):
    """Exceptions related to torch tile configuration."""
