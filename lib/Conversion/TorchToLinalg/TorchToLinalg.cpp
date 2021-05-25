//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // TODO: For `memref.dim`.
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

// -----------------------------------------------------------------------------
// Patterns (as this grows, it should be organized into multiple files)
// -----------------------------------------------------------------------------
// This is going to eventually be O(#aten ops), which is in the 100s.
//
// Most of these patterns consist of:
// 1. Checking that the operand/result types and other static properties are
//    good-enough to create a valid linalg op (such as operands being of
//    ranks/dtypes acceptable to the linalg op).
// 2. Creating dynamic error guards, usually checking a predicate on the
//    compatibility of operand shapes.
// 3. Creating init tensors for the computation op. Usually this involves
//    reifying IR for a shape transfer function based on the operand shapes.
// 4. Creating a named linalg op to replace the original op.
//
// TODO: Use linalg OpDSL to autogenerate at least 1)/2)/3) such
// that these patterns become mostly mechanical associations of
// "aten.foo -> linalg.foo".

static LogicalResult verifyLinalgCompatibleTypes(Operation *op, PatternRewriter &rewriter) {
  // For now, use a small allowlist of types we don't reject.
  // The main culprit in practice is that !numpy.any_dtype might be present
  // if shape/dtype inference wasn't good enough.
  auto isValidLinalgType = [](Type type) {
    if (auto rankedTensor = type.dyn_cast<RankedTensorType>()) {
      if (BaseMemRefType::isValidElementType(rankedTensor.getElementType()))
        return true;
    }
    if (type.isa<FloatType, IntegerType, IndexType>())
      return true;
    return false;
  };
  bool valid = llvm::all_of(op->getOperandTypes(), isValidLinalgType) &&
               llvm::all_of(op->getResultTypes(), isValidLinalgType);
  if (!valid)
    return rewriter.notifyMatchFailure(op, "type cannot be lowered to linalg");
  return success();
}

LogicalResult convertMmOp(AtenMmOp op, PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);

  // A user can write an errorneous program where `aten.mm` is in fact called
  // with operands of invalid rank or dtype. We cannot convert to linalg in this
  // case or we will get a verifier error, which corresponds to breaking of
  // *internal* compiler invariants, and for a user manifests as a compiler
  // crash in the worst case (such as we try to canonicalize/fold/print the
  // invalid op before the verifier gets to see it -- also release builds of a
  // mature copmiler usually have the verifier turned off for compile time
  // reasons).
  //
  // The compiler cannot crash even if the user wrote an erroneous program!
  if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
    return failure();
  if (lhs.getType().cast<RankedTensorType>().getRank() != 2 ||
      rhs.getType().cast<RankedTensorType>().getRank() != 2) {
    return rewriter.notifyMatchFailure(
        op, "expected both operands to aten.mm to be rank 2");
  }

  Value lhsDim0 = rewriter.create<memref::DimOp>(loc, lhs, 0);
  Value lhsDim1 = rewriter.create<memref::DimOp>(loc, lhs, 1);
  Value rhsDim0 = rewriter.create<memref::DimOp>(loc, rhs, 0);
  Value rhsDim1 = rewriter.create<memref::DimOp>(loc, rhs, 1);
  Value contractingDimEqual =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, lhsDim1, rhsDim0);
  rewriter.create<AssertOp>(
      loc, contractingDimEqual,
      rewriter.getStringAttr(
          "mismatching contracting dimension for torch.aten.mm"));

  Type elementType = op.getType().cast<TensorType>().getElementType();
  Value initTensor = rewriter.create<linalg::InitTensorOp>(
      loc, ValueRange{lhsDim0, rhsDim1}, elementType);
  Value c0 = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
  Value zeroFill =
      rewriter.create<linalg::FillOp>(loc, initTensor, c0).getResult(0);
  Value matmul = rewriter
                     .create<linalg::MatmulOp>(loc, zeroFill.getType(),
                                               ValueRange{lhs, rhs}, zeroFill)
                     .getResult(0);
  // When constructed with just dynamic sizes, InitTensorOp will have a result
  // type which has all `?`'s for dimensions, which might not be the result
  // type of `op`. The constraints on later linalg ops means that the result of
  // the MatmulOp will have this type too. So cast it to the desired type so
  // that in the end we have the original result type.
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), matmul);

  return success();
}

// See comments at in convertMmOp and the heading for this section for general
// considerations. This function needs to be auto-generated.
LogicalResult convertLinearOp(AtenLinearOp op, PatternRewriter &rewriter) {
  MLIRContext *context = op->getContext();
  Location loc = op->getLoc();
  Value input = op.input();
  Value weight = op.weight();
  Value bias = op.bias();
  // TODO: Handle the case of bias being None (bias is optional).
  if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
    return failure();
  auto inputType = input.getType().cast<RankedTensorType>();
  auto weightType = weight.getType().cast<RankedTensorType>();
  auto biasType = bias.getType().cast<RankedTensorType>();
  // Only handle the case of rank 2 `input` for now.
  // TODO: Insert the appropriate reshape to collapse any leading dimensions.
  if (inputType.getRank() != 2 || weightType.getRank() != 2 ||
      biasType.getRank() != 1) {
    return rewriter.notifyMatchFailure(
        op,
        "expected both input and weight to be rank 2 and bias to be rank 1");
  }
  // TODO: Handle type promotion. What are ATen's promotion rules?
  if (inputType.getElementType() != weightType.getElementType() ||
      inputType.getElementType() != biasType.getElementType()) {
    return rewriter.notifyMatchFailure(op, "unimplemented: type promotion");
  }

  // TODO: We can handle a static size 1 here at some complexity cost, but the
  // dynamic case is not representable in linalg. We don't handle either for
  // now. Biases are generally statically shaped for most models (since for
  // inference they are constants, and for training they don't change shape
  // typically), so this is not too constraining.
  auto biasSize = bias.getType().cast<RankedTensorType>().getShape()[0];
  if (biasSize == 1 || biasSize == ShapedType::kDynamicSize)
    return rewriter.notifyMatchFailure(
        op, "unimplemented: size-1 broadcasting for aten::LinearOp");

  auto getDimOp = [&](Value v, int dimension) {
    return rewriter.create<memref::DimOp>(loc, v, dimension);
  };
  Value inputDim0 = getDimOp(input, 0);
  Value inputDim1 = getDimOp(input, 1);
  Value weightDim0 = getDimOp(weight, 0);
  Value weightDim1 = getDimOp(weight, 1);
  Value biasDim0 = getDimOp(bias, 0);
  Value contractingDimEqual =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, inputDim1, weightDim1);
  rewriter.create<AssertOp>(
      loc, contractingDimEqual,
      rewriter.getStringAttr(
          "mismatching contracting dimension for aten.linear"));
  // Here we take advantage of ruling out the size-1 case above.
  // In the static-size-1 case, we will not emit this check at all.
  Value biasSizeCorrect =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, weightDim0, biasDim0);
  rewriter.create<AssertOp>(
      loc, biasSizeCorrect,
      rewriter.getStringAttr("mismatching bias size for aten.linear"));

  Value initTensor = rewriter.create<linalg::InitTensorOp>(
      loc, ValueRange{inputDim0, weightDim0}, inputType.getElementType());
  SmallVector<AffineMap> broadcastIndexingMaps = {
      AffineMap::get(
          /*dimCount=*/2, /*symbolCount=*/0, rewriter.getAffineDimExpr(1)),
      rewriter.getMultiDimIdentityMap(2)};
  SmallVector<StringRef> iteratorTypes(2, "parallel");
  Value broadcasted = rewriter
                          .create<linalg::GenericOp>(
                              loc, initTensor.getType(), bias, initTensor,
                              /*indexingMaps=*/broadcastIndexingMaps,
                              /*iteratorTypes=*/iteratorTypes,
                              [](OpBuilder &b, Location loc, ValueRange args) {
                                b.create<linalg::YieldOp>(loc, args[0]);
                              })
                          .getResult(0);
  // We need a matmul with dimension ordering (N, K) * (M, K), so transpose
  // the weights to fit into linalg::MatmulOp which is (N, K) * (K, M).
  // TODO: This whole aten.linear lowering should eventually be generated from a
  // single linalg ODS generator statement. Both the bias and matmul part.
  SmallVector<AffineMap> transposeIndexingMaps = {
      AffineMap::get(
          /*dimCount=*/2, /*symbolCount=*/0,
          {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(0)},
          context),
      rewriter.getMultiDimIdentityMap(2)};
  Value transposedWeightInitTensor = rewriter.create<linalg::InitTensorOp>(
      loc, ValueRange{weightDim1, weightDim0}, weightType.getElementType());
  Value transposedWeights =
      rewriter
          .create<linalg::GenericOp>(
              loc, transposedWeightInitTensor.getType(), weight,
              transposedWeightInitTensor,
              /*indexingMaps=*/transposeIndexingMaps,
              /*iteratorTypes=*/iteratorTypes,
              [](OpBuilder &b, Location loc, ValueRange args) {
                b.create<linalg::YieldOp>(loc, args[0]);
              })
          .getResult(0);
  Value matmul = rewriter.create<linalg::MatmulOp>(
      loc, broadcasted.getType(), ValueRange{input, transposedWeights},
      broadcasted).getResult(0);
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), matmul);
  return success();
}

// See comments at in convertMmOp and the heading for this section for general
// considerations. This function needs to be auto-generated.
LogicalResult convertConv2dOp(AtenConv2dOp op, PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  Value input    = op.input();
  Value weight   = op.weight();
  Value bias     = op.bias();
  Value stride   = op.stride();
  Value padding  = op.padding();
  Value dilation = op.dilation();
  Value groups   = op.groups();
  // TODO: Handle the case of bias being None (bias is optional).
  if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
    return failure();
  auto inputType = input.getType().cast<RankedTensorType>();
  auto weightType = weight.getType().cast<RankedTensorType>();
  auto biasType = bias.getType().cast<RankedTensorType>();
  // Only handle the case of rank 4 `input`, NCHW format, for now.
  // TODO: Insert the appropriate reshape to collapse any leading dimensions.
  if (inputType.getRank() != 4 || weightType.getRank() != 4 ||
      biasType.getRank() != 1) {
    return rewriter.notifyMatchFailure(
        op,
        "expected both input and weight to be rank 4 and bias to be rank 1");
  }
  // TODO: Handle type promotion. What are ATen's promotion rules?
  if (inputType.getElementType() != weightType.getElementType() ||
      inputType.getElementType() != biasType.getElementType()) {
    return rewriter.notifyMatchFailure(op, "unimplemented: type promotion");
  }

  // TODO: We can handle a static size 4 here at some complexity cost, but the
  // dynamic case is not representable in linalg. We don't handle either for
  // now. Biases are generally statically shaped for most models (since for
  // inference they are constants, and for training they don't change shape
  // typically), so this is not too constraining.
  auto biasSize = bias.getType().cast<RankedTensorType>().getShape()[0];
  if (biasSize == 4 || biasSize == ShapedType::kDynamicSize)
    return rewriter.notifyMatchFailure(
        op, "unimplemented: size-4 broadcasting for aten::Conv2dOp");

  auto getDimOp = [&](Value v, int dimension) {
    return rewriter.create<memref::DimOp>(loc, v, dimension);
  };
  Value inputDim0 = getDimOp(input, 0); // B
  Value inputDim1 = getDimOp(input, 1); // Cin
  Value inputDim2 = getDimOp(input, 2); // H
  Value inputDim3 = getDimOp(input, 3); // W
  Value weightDim0 = getDimOp(weight, 0); // Cout
  Value weightDim1 = getDimOp(weight, 1); // Cin
  Value weightDim2 = getDimOp(weight, 2); // KH
  Value weightDim3 = getDimOp(weight, 3); // KW
  Value biasDim0 = getDimOp(bias, 0);
  Value contractingDimEqual =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, inputDim1, weightDim1);
  rewriter.create<AssertOp>(
      loc, contractingDimEqual,
      rewriter.getStringAttr(
          "mismatching contracting dimension for aten.conv2d"));
  Value validFilterH =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::uge, inputDim2, weightDim2);
  rewriter.create<AssertOp>(
      loc, validFilterH,
      rewriter.getStringAttr(
          "input height must be greater than or equal to filter KH-dimension"));
  Value validFilterW =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::uge, inputDim3, weightDim3);
  rewriter.create<AssertOp>(
      loc, validFilterW,
      rewriter.getStringAttr(
          "input width must be greater than or equal to filter KW-dimension"));
  // Here we take advantage of ruling out the size-4 case above.
  // In the static-size-4 case, we will not emit this check at all.
  Value biasSizeCorrect =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, weightDim0, biasDim0);
  rewriter.create<AssertOp>(
      loc, biasSizeCorrect,
      rewriter.getStringAttr("mismatching bias size for aten.conv2d"));

  // Determine output shape.
  // TODO: This only supports the NCHW data format. Consider other formats and lower ranks.
  // TODO: Replace hard-coded stride/dilation/padding constant-ops.
  Value cI1 = rewriter.create<ConstantOp>(op->getLoc(), rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  Value cI2 = rewriter.create<ConstantOp>(op->getLoc(), rewriter.getIntegerAttr(rewriter.getIndexType(), 2));
  //Value stride = cI1;
  //Value dilation = cI1;
  //Value padding = cI0;
  Value strideHeight = stride;
  Value strideWidth = stride;
  Value dilationHeight = dilation;
  Value dilationWidth = dilation;
  Value paddingHeight = padding;
  Value paddingWidth = padding;
  // Output height
  Value twicePaddingHeight = rewriter.create<MulIOp>(loc, paddingHeight, cI2);
  Value heightPlusTwicePadding = rewriter.create<SubIOp>(loc, inputDim2, twicePaddingHeight);
  Value filterHeightMinusOne = rewriter.create<SubIOp>(loc, weightDim2, cI1);
  Value dilationFilterHeight = rewriter.create<MulIOp>(loc, dilationHeight, filterHeightMinusOne);
  Value outHeightUnstridedPlusOne = rewriter.create<SubIOp>(loc, heightPlusTwicePadding, dilationFilterHeight);
  Value outHeightUnstrided = rewriter.create<SubIOp>(loc, outHeightUnstridedPlusOne, cI1);
  Value outHeightMinusOne = rewriter.create<UnsignedDivIOp>(loc, outHeightUnstrided, strideHeight);
  Value outHeight = rewriter.create<AddIOp>(loc, outHeightMinusOne, cI1);
  // Output width
  Value twicePaddingWidth = rewriter.create<MulIOp>(loc, paddingWidth, cI2);
  Value widthPlusTwicePadding = rewriter.create<SubIOp>(loc, inputDim3, twicePaddingWidth);
  Value filterWidthMinusOne = rewriter.create<SubIOp>(loc, weightDim3, cI1);
  Value dilationFilterWidth = rewriter.create<MulIOp>(loc, dilationWidth, filterWidthMinusOne);
  Value outWidthUnstridedPlusOne = rewriter.create<SubIOp>(loc, widthPlusTwicePadding, dilationFilterWidth);
  Value outWidthUnstrided = rewriter.create<SubIOp>(loc, outWidthUnstridedPlusOne, cI1);
  Value outWidthMinusOne = rewriter.create<UnsignedDivIOp>(loc, outWidthUnstrided, strideWidth);
  Value outWidth = rewriter.create<AddIOp>(loc, outWidthMinusOne, cI1);
  // Output shape
  ValueRange outputShape = ValueRange({inputDim0, weightDim0, outHeight, outWidth});
  //Value outputShape = rewriter.create<tensor::FromElementsOp>(
  //    loc, ValueRange({inputDim0, weightDim0, outHeight, outWidth}));

  Value initTensor = rewriter.create<linalg::InitTensorOp>(
      loc, outputShape, inputType.getElementType());
  SmallVector<AffineMap> broadcastIndexingMaps = {
      AffineMap::get(
          /*dimCount=*/4, /*symbolCount=*/0, rewriter.getAffineDimExpr(1)),
      rewriter.getMultiDimIdentityMap(4)};
  SmallVector<StringRef> iteratorTypes(2, "parallel");
  Value broadcasted = rewriter
                          .create<linalg::GenericOp>(
                              loc, initTensor.getType(), bias, initTensor,
                              /*indexingMaps=*/broadcastIndexingMaps,
                              /*iteratorTypes=*/iteratorTypes,
                              [](OpBuilder &b, Location loc, ValueRange args) {
                                b.create<linalg::YieldOp>(loc, args[0]);
                              })
                          .getResult(0);

  Value conv2dNCHW = rewriter.create<linalg::ConvNCHWOp>(
      op.getLoc(), TypeRange(op.getType()),
      ValueRange({input, weight}), ValueRange(broadcasted)).getResult(0);
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), conv2dNCHW);
  return success();
}

namespace {
// Converts a unary op. There is no implicit broadcasting behavior, so these can
// be trivially lowered to linalg.
// TODO: For binary ops, we will need a "linalg.generic-like" op that models
// N-ary broadcasting and allows us to do multiversioning techniques for
// lowering to linalg. We can trivially handle this as through that
// abstraction instead.
struct ConvertUnaryOp : RewritePattern {
  ConvertUnaryOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<AtenTanhOp>(op))
      return rewriter.notifyMatchFailure(op, "not a unary op");

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Value operand = op->getOperand(0);
    auto type = op->getResult(0).getType().cast<RankedTensorType>();
    auto rank = type.getRank();

    SmallVector<StringRef> iteratorTypes(rank, "parallel");
    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(rank),
        rewriter.getMultiDimIdentityMap(rank)};

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, type, operand, operand,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result;
          if (isa<AtenTanhOp>(op)) {
            result = b.create<math::TanhOp>(loc, args[0]);
          }
          b.create<linalg::YieldOp>(loc, result);
        });

    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToLinalg
    : public ConvertTorchToLinalgBase<ConvertTorchToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<math::MathDialect>();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), getPatterns());
  }

  FrozenRewritePatternSet getPatterns() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add(convertMmOp);
    patterns.add(convertLinearOp);
    patterns.add(convertConv2dOp);
    patterns.add<ConvertUnaryOp>(context);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertTorchToLinalgPass() {
  return std::make_unique<ConvertTorchToLinalg>();
}
