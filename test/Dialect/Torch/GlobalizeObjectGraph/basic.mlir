// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

// Basic case.

// CHECK-LABEL:   torch.global_slot @b : !basicpy.BoolType  {
// CHECK:           %[[INIT:.*]] = basicpy.bool_constant true
// CHECK:           torch.global_slot.init %[[INIT]] : !basicpy.BoolType
// CHECK:         }

// CHECK-LABEL:   torch.global_slot @i : i64  {
// CHECK:           %[[INIT:.*]] = basicpy.numeric_constant 3 : i64
// CHECK:           torch.global_slot.init %[[INIT]] : i64
// CHECK:         }

// CHECK-LABEL:   torch.global_slot @f : f64  {
// CHECK:           %[[INIT:.*]] = basicpy.numeric_constant 4.250000e+01 : f64
// CHECK:           torch.global_slot.init %[[INIT]] : f64
// CHECK:         }

// CHECK-LABEL:   torch.global_slot @t : !torch.tensor  {
// CHECK:           %[[T:.*]] = torch.tensor(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor
// CHECK:           torch.global_slot.init %[[T]] : !torch.tensor
// CHECK:         }

torch.class_type @c {
  torch.attr "b" : !basicpy.BoolType
  torch.attr "i" : i64
  torch.attr "f" : f64
  torch.attr "t" : !torch.tensor
}

%bool_true = basicpy.bool_constant true
%i = basicpy.numeric_constant 3 : i64
%f = basicpy.numeric_constant 4.250000e+01 : f64
%t = torch.tensor(dense<1.0> : tensor<1xf32>) : !torch.tensor
torch.nn_module {
  torch.slot "b", %bool_true : !basicpy.BoolType
  torch.slot "i", %i : i64
  torch.slot "f", %f : f64
  torch.slot "t", %t : !torch.tensor
} : !torch.nn.Module<"c">
