// RUN: npcomp-opt -torch-adjust-calling-conventions -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           %[[NONVAL_TENSOR:.*]] = torch.copy.tensor %[[ERASED]] : !torch.vtensor -> !torch.tensor
// CHECK:           return %[[NONVAL_TENSOR]] : !torch.tensor
func @basic(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[2,3,?],f32>}) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// CHECK-LABEL:   func @no_type_bound(
// CHECK-SAME:                        %[[ARG:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           return %[[ARG]] : !torch.tensor
func @no_type_bound(%arg0: !torch.tensor) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// CHECK-LABEL:   func @call(
// CHECK-SAME:               %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[ARG_ERASED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           %[[ARG_NONVAL:.*]] = torch.copy.tensor %[[ARG_ERASED]] : !torch.vtensor -> !torch.tensor
// CHECK:           %[[INFO_ADDED:.*]] = torch.tensor_static_info_cast %[[ARG_NONVAL]] : !torch.tensor to !torch.tensor<[2,3,?],f32>
// CHECK:           %[[CALL_ARG:.*]] = torch.copy.tensor %[[INFO_ADDED]] : !torch.tensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
// CHECK:           %[[CALL_RES:.*]] = call @call(%[[CALL_ARG]]) : (!torch.vtensor<[2,3,?],f32>) -> !torch.tensor
// CHECK:           return %[[ARG_NONVAL]] : !torch.tensor
func @call(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[2,3,?],f32>}) -> !torch.tensor {
  %0 = call @call(%arg0) : (!torch.tensor) -> !torch.tensor
  return %arg0 : !torch.tensor
}

// CHECK-LABEL:   func @none_return() {
// CHECK:           %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
// CHECK:           return
func @none_return() -> !basicpy.NoneType {
  %1 = basicpy.singleton : !basicpy.NoneType
  return %1 : !basicpy.NoneType
}

// CHECK-LABEL:   func @none_call_return() {
// CHECK:           call @none_return() : () -> ()
// CHECK:           %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
// CHECK:           "test.use"(%[[NONE]]) : (!basicpy.NoneType) -> ()
// CHECK:           return
func @none_call_return() {
  %0 = call @none_return() : () -> !basicpy.NoneType
  "test.use"(%0) : (!basicpy.NoneType) -> ()
  return
}
