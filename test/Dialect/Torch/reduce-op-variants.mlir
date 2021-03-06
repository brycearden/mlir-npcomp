// RUN: npcomp-opt -torch-reduce-op-variants  %s | FileCheck %s

// CHECK-LABEL:   func @convert_to_value_semantic_tensors(
// CHECK-SAME:                                       %[[ARG:.*]]: !torch.tensor<[],f32>) -> !torch.tensor<[],f32> {
// CHECK:           %[[OPERAND_TENSOR:.*]] = torch.copy.tensor %[[ARG]] : !torch.tensor<[],f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[RESULT_TENSOR:.*]] = torch.aten.tanh %[[OPERAND_TENSOR]] : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[RET:.*]] = torch.copy.tensor %[[RESULT_TENSOR]] : !torch.vtensor<[],f32> -> !torch.tensor<[],f32>
// CHECK:           return %[[RET]] : !torch.tensor<[],f32>
func @convert_to_value_semantic_tensors(%arg0: !torch.tensor<[],f32>) -> !torch.tensor<[],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.tensor<[],f32> -> !torch.tensor<[],f32>
  return %0 : !torch.tensor<[],f32>
}


// CHECK-LABEL:   func @reduce_trailing_underscore_inplace_variant(
// CHECK-SAME:                          %[[ARG0:.*]]: !torch.tensor<[2,2],f32>,
// CHECK-SAME:                          %[[ARG1:.*]]: !torch.tensor<[2,2],f32>) -> (!torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>) {
// CHECK:           %[[VAL_2:.*]] = constant 1 : i64
// CHECK:           %[[TENSOR0:.*]] = torch.copy.tensor %[[ARG0]] : !torch.tensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           %[[TENSOR1:.*]] = torch.copy.tensor %[[ARG1]] : !torch.tensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           %[[TENSOR_RESULT:.*]] = torch.aten.add.Tensor %[[TENSOR0]], %[[TENSOR1]], %[[VAL_2]] : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>, i64 -> !torch.vtensor<[2,2],f32>
// Note: This somewhat redundant conversion back and forth
// (which is cleaned up by canonicalization) is an artifact of two patterns
// being applied in sequence.
// CHECK:           %[[ARRAY_RESULT:.*]] = torch.copy.tensor %[[TENSOR_RESULT]] : !torch.vtensor<[2,2],f32> -> !torch.tensor<[2,2],f32>
// CHECK:           %[[TENSOR_AGAIN:.*]] = torch.copy.tensor %[[ARRAY_RESULT]] : !torch.tensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           torch.overwrite.tensor %[[TENSOR_AGAIN]] overwrites %[[ARG0]] : !torch.vtensor<[2,2],f32>, !torch.tensor<[2,2],f32>
// CHECK:           return %[[ARG0]], %[[ARG0]] : !torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>
func @reduce_trailing_underscore_inplace_variant(%arg0: !torch.tensor<[2,2],f32>, %arg1: !torch.tensor<[2,2],f32>) -> (!torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>) {
  %c1 = constant 1 : i64
  %0 = torch.aten.add_.Tensor %arg0, %arg1, %c1 : !torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>, i64 -> !torch.tensor<[2,2],f32>
  return %0, %arg0 : !torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>
}
