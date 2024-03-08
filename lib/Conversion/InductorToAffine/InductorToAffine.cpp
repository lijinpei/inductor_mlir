#include "inductor/Conversion/InductorToAffine/Passes.h"
#include "inductor/Dialect/Inductor/IR/InductorDialect.h"
#include "inductor/Dialect/Inductor/IR/InductorOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"

#include <functional>
#include <iterator>
#include <optional>
#include <utility>

namespace mlir {
namespace inductor {
#define GEN_PASS_DEF_CONVERTINDUCTORTOAFFINE
#include "inductor/Conversion/InductorToAffine/Passes.h.inc"

namespace {
class ToAffineConversionTarget : public ConversionTarget {
public:
  ToAffineConversionTarget(MLIRContext &context) : ConversionTarget(context) {
    addIllegalDialect<inductor::InductorDialect>();
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<affine::AffineDialect>();
  }
};

struct PointwiseLoopPattern : OpConversionPattern<inductor::PointwiseLoopOp> {
  using OpTy = inductor::PointwiseLoopOp;
  using BaseTy = OpConversionPattern<inductor::PointwiseLoopOp>;
  PointwiseLoopPattern(MLIRContext &context) : BaseTy(&context) {}
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    Value c0 = builder.create<index::ConstantOp>(loc, 0);
    auto ubs = adaptor.getRanges();
    auto rank = ubs.size();
    SmallVector<Value> lbs(rank, c0);
    SmallVector<int64_t> steps(rank, 1);
    auto idenMap = builder.getMultiDimIdentityMap(rank);
    auto parOp = builder.create<affine::AffineParallelOp>(
        loc, TypeRange{}, ArrayRef<arith::AtomicRMWKind>{}, idenMap, lbs,
        idenMap, ubs, steps);
    auto *parBody = parOp.getBody();
    auto &fnRegion = op.getInnerFn();
    Block *fnBlock;
    if (fnRegion.hasOneBlock()) {
      fnBlock = &fnRegion.front();
    } else {
      // TODO:
      fnBlock = nullptr;
      assert(false);
    }
    auto oldYield = cast<inductor::LoopYieldOp>(fnBlock->getTerminator());
    auto yieldVal = oldYield.getValue();
    builder.setInsertionPoint(oldYield);
    auto ivs = parOp.getIVs();
    builder.create<affine::AffineStoreOp>(loc, yieldVal, adaptor.getBuffer(),
                                          ivs);
    builder.eraseOp(oldYield);
    builder.inlineBlockBefore(fnBlock, parBody, std::prev(parBody->end()), ivs);
    builder.replaceOp(op, parOp);
    return success();
  }
};

arith::AtomicRMWKind reductionTypeToRMWKind(ReductionType redType,
                                            Type elemType) {
  switch (redType) {
  default:
    break;
  case ReductionType::ANY:
    assert(false);
    break;
  case ReductionType::MAX:
    if (elemType.isUnsignedInteger() || elemType.isIndex()) {
      return arith::AtomicRMWKind::maxu;
    } else if (elemType.isSignedInteger()) {
      return arith::AtomicRMWKind::maxs;
    } else {
      return arith::AtomicRMWKind::maximumf;
    }
  case ReductionType::MIN:
    if (elemType.isUnsignedInteger() || elemType.isIndex()) {
      return arith::AtomicRMWKind::minu;
    } else if (elemType.isSignedInteger()) {
      return arith::AtomicRMWKind::mins;
    } else {
      return arith::AtomicRMWKind::minimumf;
    }
  case ReductionType::PROD:
    if (elemType.isIntOrIndex()) {
      return arith::AtomicRMWKind::muli;
    } else {
      return arith::AtomicRMWKind::mulf;
    }
  case ReductionType::SUM:
    if (elemType.isIntOrIndex()) {
      return arith::AtomicRMWKind::addi;
    } else {
      return arith::AtomicRMWKind::addf;
    }
  case ReductionType::XOR_SUM:
    return arith::AtomicRMWKind::addi;
  case ReductionType::ARGMIN:
    assert(false);
    break;
  case ReductionType::ARGMAX:
    assert(false);
    break;
  case ReductionType::WELFORD_COMBINE:
    assert(false);
    break;
  }
  llvm_unreachable("invalid inductor reduction type");
}

struct ReductionLoopPattern : OpConversionPattern<inductor::ReductionLoopOp> {
  using OpTy = inductor::ReductionLoopOp;
  using BaseTy = OpConversionPattern<OpTy>;
  ReductionLoopPattern(MLIRContext &context) : BaseTy(&context) {}
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    Value c0 = builder.create<index::ConstantOp>(loc, 0);

    auto outerUbs = adaptor.getRanges();
    auto outerRank = outerUbs.size();
    SmallVector<Value> outerLbs(outerRank, c0);
    SmallVector<int64_t> outerSteps(outerRank, 1);
    auto outerIdenMap = builder.getMultiDimIdentityMap(outerRank);
    auto outerParOp = builder.create<affine::AffineParallelOp>(
        loc, TypeRange{}, ArrayRef<arith::AtomicRMWKind>{}, outerIdenMap,
        outerLbs, outerIdenMap, outerUbs, outerSteps);
    auto *outerParBody = outerParOp.getBody();
    auto outerParTerm = std::prev(outerParBody->end());

    auto &fnRegion = op.getInnerFn();
    Block *fnBlock;
    if (fnRegion.hasOneBlock()) {
      fnBlock = &fnRegion.front();
    } else {
      // TODO:
      fnBlock = nullptr;
      assert(false);
    }
    auto oldYield = cast<inductor::LoopYieldOp>(fnBlock->getTerminator());
    auto yieldVal = oldYield.getValue();
    auto innerElemType = yieldVal.getType();
    auto rmwKind = reductionTypeToRMWKind(op.getReductionType(), innerElemType);

    builder.setInsertionPoint(outerParBody, outerParTerm);
    auto innerUbs = adaptor.getReductionRanges();
    auto innerRank = innerUbs.size();
    SmallVector<Value> innerLbs(innerRank, c0);
    SmallVector<int64_t> innerSteps(innerRank, 1);
    auto innerIdenMap = builder.getMultiDimIdentityMap(innerRank);
    auto innerParOp = builder.create<affine::AffineParallelOp>(
        loc, innerElemType, rmwKind, innerIdenMap, innerLbs, innerIdenMap,
        innerUbs, innerSteps);
    auto *innerParBody = innerParOp.getBody();

    auto outerIVs = outerParOp.getIVs();
    assert(innerParOp.getResults().size() == 1);
    builder.create<affine::AffineStoreOp>(loc, innerParOp.getResults()[0],
                                          adaptor.getBuffer(), outerIVs);
    auto innerIVs = innerParOp.getIVs();
    SmallVector<Value> ivs;
    ivs.append(outerIVs.begin(), outerIVs.end());
    ivs.append(innerIVs.begin(), innerIVs.end());
    builder.inlineBlockBefore(fnBlock, innerParBody,
                              std::prev(innerParBody->end()), ivs);
    builder.replaceOp(op, outerParOp);
    return success();
  }
};

struct LoopYieldPattern : OpConversionPattern<inductor::LoopYieldOp> {
  using OpTy = inductor::LoopYieldOp;
  using BaseTy = OpConversionPattern<OpTy>;
  LoopYieldPattern(MLIRContext &context) : BaseTy(&context) {}
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    auto newYield =
        builder.create<affine::AffineYieldOp>(loc, adaptor.getValue());
    builder.replaceOp(op, newYield);
    return success();
  }
};

void populateToAffineConversionPatterns(MLIRContext &context,
                                        RewritePatternSet &patterns) {
  patterns.add<PointwiseLoopPattern>(context);
  patterns.add<ReductionLoopPattern>(context);
  patterns.add<LoopYieldPattern>(context);
}

class ConvertInductorToAffine
    : public impl::ConvertInductorToAffineBase<ConvertInductorToAffine> {
  void runOnOperation() override {
    auto mod = getOperation();
    auto *context = mod->getContext();
    ToAffineConversionTarget target(*context);
    RewritePatternSet patterns(context);
    populateToAffineConversionPatterns(*context, patterns);
    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
std::unique_ptr<OperationPass<ModuleOp>> createConvertInductorToAffinePass() {
  return std::make_unique<ConvertInductorToAffine>();
}
} // namespace inductor
} // namespace mlir
