// RUN: npcomp-opt -lower-to-refbackrt-abi -split-input-file -verify-diagnostics <%s | FileCheck %s --dump-input=fail

// Test refbackrt support for scalar arguments

// CHECK-LABEL: func @scalar_args(%arg0: i32) -> i32 {
func @scalar_args(%arg0: i32) -> i32 {
  return %arg0 : i32
}