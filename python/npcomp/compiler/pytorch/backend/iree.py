#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import torch

from mlir.ir import *
from mlir.passmanager import *
from npcomp.compiler.utils import logging
import iree.runtime as ireert
import iree.compiler as ireec

__all__ = [
    "CompilerBackend",
]

PREPARE_FOR_IREE_PASSES = (
  "npcomp-iree-backend-lower-linkage",
)

class IreeModuleInvoker:
  """Wrapper around a native IREE module for calling functions."""

  def __init__(self, iree_module):
    super().__init__()
    self._iree_module = iree_module

  def __getattr__(self, function_name):
    return self.__getitem__(function_name)

  def __getitem__(self, function_name):

    def invoke(*args):
      results = self._iree_module[function_name](*args)
      if len(results) == 1:
        # De-tuple.
        return results[0]
      else:
        return tuple(results)

    invoke.__isnpcomp__ = True
    return invoke


class TorchIreeModuleInvoker(IreeModuleInvoker):
  """Allows torch.Tensor inputs to be passed to module invocations."""

  def __getitem__(self, function_name: str):
    numpy_invoke = super().__getitem__(function_name)

    def invoke(*args):
      args = tuple(
          arg.numpy() if isinstance(arg, torch.Tensor) else arg for arg in args)
      return numpy_invoke(*args)

    return invoke


class CompilerBackend:
  """Main entry-point for the backend."""

  def __init__(self):
    super().__init__()
    self._debug = logging.debug_enabled()

  def compile(self, imported_module: Module):
    """Compiles an imported module, with a flat list of functions.
    The module is expected to be in "TCP + scalar code" form.
    TODO: More clearly define the backend contract. Generally this will
    extend to support globals, lists, and other stuff.

    Args:
      imported_module: The MLIR module consisting of funcs in the torch
        dialect.
    Returns:
      An opaque, backend specific module object that can be passed to load.
      The object may actually be something more specific to the backend (i.e.
      for IREE, it is a serialized VM flatbuffer) but the contract is that
      it is operated on by methods on this class.
    """
    with imported_module.context as context:
      if self._debug:
        logging.debug("IR passed to IREE compiler backend:\n{}",
                      imported_module)
      pipeline_str = ",".join(PREPARE_FOR_IREE_PASSES)
      if self._debug:
        logging.debug("Running Prepare For IREE pipeline '{}'", pipeline_str)
      pm = PassManager.parse(pipeline_str)
      pm.run(imported_module)
      if self._debug:
        logging.debug(
          "IREE Input IR (this is what IREE's compiler will see):\n{}",
          imported_module)

      # Backend.
      binary = ireec.compile_str(str(imported_module),
                                 target_backends=["dylib-llvm-aot"])
      iree_config = ireert.Config(driver_name="dylib")

      iree_module = ireert.load_module(ireert.VmModule.from_flatbuffer(binary),
                                       config=iree_config)
    return iree_module

  def load(self, iree_module) -> TorchIreeModuleInvoker:
    """Loads a compiled artifact into the runtime."""
    return TorchIreeModuleInvoker(iree_module)
