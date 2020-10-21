// RUN: npcomp-opt --tcf-insert-identity <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @binary_elementwise
func @binary_elementwise(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tcf.identity %{{.*}} : tensor<?xf32>
  // CHECK: tcf.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tcf.exp %arg0 : tensor<?xf32>
  %0 = tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = tcf.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = tcf.exp %arg0 : tensor<?xf32>
  return
}