//===-------------------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Operation summaries and descriptions were systematically derived from public
// API docstrings and are licensed accordingly:
//   https://github.com/pytorch/pytorch/blob/master/LICENSE
//===----------------------------------------------------------------------===//
//
// This file is automatically generated.  Please do not edit.
// Generated via:
//   python -m torch_mlir_utils.codegen.torch_ods_gen
//
//===----------------------------------------------------------------------===//

def Torch_QuantizedLinearOp : Torch_Op<"quantized.linear", [
    HasValueSemantics,
    AllowsTypeRefinement,
    HasValueSemantics
  ]> {
  let summary = "Generated op for `quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$X,
    Torch_LinearParamsType:$W_prepack,
    AnyFloat:$Y_scale_i,
    AnyTorchIntType:$Y_zero_point_i
  );
  let results = (outs
    AnyTorchTensorType:$Y
  );
  let assemblyFormat = "$X `,` $W_prepack `,` $Y_scale_i `,` $Y_zero_point_i attr-dict `:` type($X) `,` type($W_prepack) `,` type($Y_scale_i) `,` type($Y_zero_point_i) `->` type($Y)";
}

