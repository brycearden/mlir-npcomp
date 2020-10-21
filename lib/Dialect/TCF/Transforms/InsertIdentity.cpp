#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "npcomp/Dialect/TCF/IR/TCFDialect.h"
#include "npcomp/Dialect/TCF/IR/TCFOps.h"
#include "npcomp/Dialect/TCF/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::tcf;

namespace {

class InsertIdentityPass : public TCFInsertIdentityBase<InsertIdentityPass> {
  void runOnOperation() override {
    auto func = getOperation();
    // TODO: Implement for real.
    func.walk([](tcf::AddOp addOp) {
      llvm::dbgs() << "found an AddOp!" << "\n";
      OpBuilder builder(addOp);
      builder.setInsertionPointAfter(addOp);

      // The output type of the identity matches the input type
      auto addResultType = addOp.getResult().getType();

      // TODO(brycearden): Write a RewritePattern so that all Op's that were consuming the
      // tcf::AddOp now consume the IdentityOp
      auto identityOp = builder.create<tcf::IdentityOp>(
        addOp.getLoc(), ArrayRef<Type>{addResultType}, ArrayRef<Value>{addOp.getResult()});
    });

    // // If the change cascaded to any returns, need to update the function
    // // signature.
    // Optional<ReturnOp> firstReturnOp;
    // func.walk([&](tcf::AddOp addOp) {
    //   if (!firstReturnOp) {
    //     firstReturnOp = returnOp;
    //   } else {
    //     if (returnOp.getOperandTypes() != firstReturnOp->getOperandTypes()) {
    //       returnOp.emitError() << "after refining shapes, different "
    //                               "terminators have different types";
    //       signalPassFailure();
    //     }
    //   }
    // });

    // assert(firstReturnOp && "function lacks a terminator");
    // auto funcType = func.getType();
    // SmallVector<Type, 4> resultTypes(firstReturnOp->getOperandTypes().begin(),
    //                                  firstReturnOp->getOperandTypes().end());
    // func.setType(FunctionType::get(funcType.getInputs(), resultTypes,
    //                                funcType.getContext()));
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::tcf::createInsertIdentityPass() {
  return std::make_unique<InsertIdentityPass>();
}