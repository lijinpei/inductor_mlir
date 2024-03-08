#include "inductor/Conversion/InductorToTriton/Passes.h"
#include "inductor/Dialect/Inductor/IR/InductorDialect.h"
#include "inductor/Dialect/Inductor/IR/InductorOps.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <functional>
#include <optional>
#include <utility>

namespace mlir {
namespace inductor {
#define GEN_PASS_DEF_CONVERTINDUCTORTOTRITON
#include "inductor/Conversion/InductorToTriton/Passes.h.inc"

namespace {
constexpr unsigned GlobalAddressSpace = 1;
constexpr size_t ParallelTilingSize = 1024;
constexpr size_t ReductionTilingSize = 1024;

constexpr size_t ParallelAxis = 0;
constexpr size_t ReductionAxis = 1;

// constexpr StringRef InductorToBeConvert = "inductor.to_be_convert";

struct TilingParam {
  size_t parallelTilingSize = ParallelTilingSize;
  size_t reductionTilingSize = ReductionTilingSize;
};

class LoopTilingTypeConverter : public TypeConverter {
  const TilingParam &tilingParam;

public:
  LoopTilingTypeConverter(const TilingParam &tilingParam)
      : tilingParam(tilingParam) {
    addConversion([&](Type type) -> std::optional<Type> {
      if (type.isIntOrIndexOrFloat()) {
        return getTiledType(type);
      }
      return std::nullopt;
    });
  }
  Type getTiledType(Type type) {
    return RankedTensorType::get(
        ArrayRef<int64_t>({(int64_t)tilingParam.parallelTilingSize,
                           (int64_t)tilingParam.reductionTilingSize}),
        type);
  }
};

class LoopTilingConversionTarget : public ConversionTarget {
  DenseSet<Operation *> illegalOpSet;

public:
  LoopTilingConversionTarget(Operation *op)
      : ConversionTarget(*op->getContext()) {
    op->walk([&](Operation *subOp) { illegalOpSet.insert(subOp); });
    addIllegalDialect<inductor::InductorDialect>();
    addDynamicallyLegalDialect<arith::ArithDialect>(
        [this](Operation *op) -> std::optional<bool> {
          return !illegalOpSet.contains(op);
        });
    addDynamicallyLegalDialect<math::MathDialect>(
        [this](Operation *op) -> std::optional<bool> {
          return !illegalOpSet.contains(op);
        });
    addLegalDialect<triton::TritonDialect>();
    addLegalDialect<scf::SCFDialect>();
  }
};

template <typename OpTy>
struct LoopsConverionBase : public OpConversionPattern<OpTy> {
  const TilingParam &tilingParam;
  using BaseTy = OpConversionPattern<OpTy>;
  LoopsConverionBase(const TilingParam &tilingParam, MLIRContext *context)
      : BaseTy(context), tilingParam(tilingParam) {}
};

struct ArithConstOpConversion : OpConversionPattern<arith::ConstantOp> {
  LoopTilingTypeConverter &typeConverter;
  using BaseTy = OpConversionPattern<arith::ConstantOp>;
  ArithConstOpConversion(LoopTilingTypeConverter &typeConverter,
                         MLIRContext *context)
      : BaseTy(context), typeConverter(typeConverter) {}
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    auto newConst = cast<arith::ConstantOp>(builder.clone(*op.getOperation()));
    auto resType = typeConverter.getTiledType(op.getResult().getType());
    auto splat = builder.create<triton::SplatOp>(loc, resType, newConst);
    builder.replaceOp(op, splat);
    return success();
  }
};

struct InductorCalcPlainOffsetConversion
    : OpConversionPattern<inductor::CalcPlainOffsetOp> {
  LoopTilingTypeConverter &typeConverter;
  using BaseTy = OpConversionPattern<inductor::CalcPlainOffsetOp>;
  InductorCalcPlainOffsetConversion(LoopTilingTypeConverter &typeConverter,
                                    MLIRContext *context)
      : BaseTy(context), typeConverter(typeConverter) {}
  LogicalResult
  matchAndRewrite(inductor::CalcPlainOffsetOp op,
                  inductor::CalcPlainOffsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    Value res = adaptor.getOffset();
    for (auto [coord, stride] :
         llvm::zip(adaptor.getCoord(), adaptor.getStride())) {
      auto prod = builder.create<arith::MulIOp>(loc, coord, stride);
      res = builder.create<arith::AddIOp>(loc, res, prod);
    }
    builder.replaceOp(op, res);
    return success();
  }
};

struct InductorLoadOpConversion : OpConversionPattern<inductor::LoadOp> {
  using BaseTy = OpConversionPattern<inductor::LoadOp>;
  LoopTilingTypeConverter &typeConverter;
  const llvm::StringMap<Value> &bufferMap;
  InductorLoadOpConversion(LoopTilingTypeConverter &typeConverter,
                           const llvm::StringMap<Value> &bufferMap,
                           MLIRContext *context)
      : BaseTy(context), typeConverter(typeConverter), bufferMap(bufferMap) {}
  LogicalResult
  matchAndRewrite(inductor::LoadOp op, inductor::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    auto bufferAttr = op.getBuffer();
    auto buffer = bufferMap.at(bufferAttr.strref());
    auto tensorPtrType = typeConverter.getTiledType(buffer.getType());
    auto bufferSplat =
        builder.create<triton::SplatOp>(loc, tensorPtrType, buffer);
    auto ptr = builder.create<triton::AddPtrOp>(loc, tensorPtrType, bufferSplat,
                                                adaptor.getOffset());
    auto res = builder.create<triton::LoadOp>(
        loc, ptr, triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
        /* isVolatile */ false);
    builder.replaceOp(op, res);
    return success();
  }
};

template <typename OpTy>
struct MathFloatUnaryOpConversion : OpConversionPattern<OpTy> {
  using BaseTy = OpConversionPattern<OpTy>;
  MathFloatUnaryOpConversion(MLIRContext *context) : BaseTy(context) {}
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    auto res =
        builder.create<OpTy>(loc, adaptor.getOperand(), op.getFastmath());
    builder.replaceOp(op, res);
    return success();
  }
};

template <typename OpTy>
struct ArithFloatBinaryOpConversionBase : OpConversionPattern<OpTy> {
  using BaseTy = OpConversionPattern<OpTy>;
  ArithFloatBinaryOpConversionBase(MLIRContext *context) : BaseTy(context) {}
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    auto res = builder.create<OpTy>(loc, adaptor.getLhs(), adaptor.getRhs(),
                                    op.getFastmath());
    builder.replaceOp(op, res);
    return success();
  }
};

struct YieldConversion : OpConversionPattern<YieldOp> {
  using BaseTy = OpConversionPattern<YieldOp>;
  YieldConversion(MLIRContext *context) : BaseTy(context) {}
  LogicalResult
  matchAndRewrite(YieldOp op, YieldOp::Adaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    // FIXME: reduction combine kind should be an attribute of YieldOp
    // FIXME: reduction arg should be an operand of YieldOp
    auto forOp = op->getParentOfType<scf::ForOp>();
    Value redOpRes;
    {
      OpBuilder::InsertionGuard guard(builder);
      auto redOp = builder.create<triton::ReduceOp>(loc, adaptor.getValue(),
                                                    (uint32_t)ReductionAxis);
      redOpRes = redOp.getResult()[0];

      auto &combineBB = *builder.createBlock(&redOp.getCombineOp());
      auto combineArgType = builder.getF32Type();
      combineBB.addArgument(combineArgType, loc);
      combineBB.addArgument(combineArgType, loc);
      builder.setInsertionPointToStart(&combineBB);
      Value combinedVal = builder.create<arith::AddFOp>(
          loc, combineBB.getArgument(0), combineBB.getArgument(1));
      builder.create<triton::ReduceReturnOp>(loc, combinedVal);
    }

    auto redArgVal = forOp.getRegionIterArg(0);
    Value redVal = builder.create<arith::AddFOp>(loc, redArgVal, redOpRes);
    builder.create<scf::YieldOp>(loc, redVal);
    builder.eraseOp(op);
    return success();
  }
};

struct ReductionLoopConversion : LoopsConverionBase<ReductionOp> {
  using BaseTy = LoopsConverionBase<ReductionOp>;
  llvm::StringMap<Value> bufferMap;
  ReductionLoopConversion(const TilingParam &tilingParam,
                          llvm::StringMap<Value> &bufferMap,
                          MLIRContext *context)
      : BaseTy(tilingParam, context), bufferMap(bufferMap) {}
  LogicalResult
  matchAndRewrite(ReductionOp op, ReductionOp::Adaptor adaptor,
                  ConversionPatternRewriter &builder) const final {
    auto loc = op.getLoc();
    auto ranges = op.getRanges();
    auto redRanges = op.getReductionRanges();
    assert(ranges.size() == 1 && redRanges.size() == 1);
    auto redRange = redRanges[0];

    auto indexTensorType = RankedTensorType::get(
        ArrayRef<int64_t>({(int64_t)tilingParam.parallelTilingSize,
                           (int64_t)tilingParam.reductionTilingSize}),
        builder.getI32Type());

    auto pgId =
        builder.create<triton::GetProgramIdOp>(loc, triton::ProgramIDDim::X);
    auto parTileSize = builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(tilingParam.parallelTilingSize));
    Value parTileStart = builder.create<arith::MulIOp>(loc, pgId, parTileSize);
    auto parIndexTensorTy =
        RankedTensorType::get(ArrayRef<int64_t>(tilingParam.parallelTilingSize),
                              builder.getI32Type());
    parTileStart =
        builder.create<triton::SplatOp>(loc, parIndexTensorTy, parTileStart);
    auto parTileOffset = builder.create<triton::MakeRangeOp>(
        loc, parIndexTensorTy, builder.getI32IntegerAttr(0),
        builder.getI32IntegerAttr(tilingParam.parallelTilingSize));
    auto parIndex1D =
        builder.create<arith::AddIOp>(loc, parTileStart, parTileOffset);
    auto parIndexSlice =
        builder.create<triton::ExpandDimsOp>(loc, parIndex1D, ReductionAxis);
    Value parIndex = builder.create<triton::BroadcastOp>(loc, indexTensorType,
                                                         parIndexSlice);
    Value lower =
        builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(0));
    Value step = builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(tilingParam.reductionTilingSize));
    // FIXME: support other types
    auto elemTy = builder.getF32Type();
    auto accTensorTy =
        RankedTensorType::get((int64_t)tilingParam.parallelTilingSize, elemTy);
    // FIXME: this init is only for sum
    auto accInitScalar =
        builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(.0));
    Value accInit =
        builder.create<triton::SplatOp>(loc, accTensorTy, accInitScalar);
    auto redForLoop =
        builder.create<scf::ForOp>(loc, lower, redRange, step, accInit);
    auto &innerFnBody = op.getInnerFn().front();
    auto &loopBody = redForLoop.getRegion().front();
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&loopBody);
      auto redIndexTensorTy = RankedTensorType::get(
          ArrayRef<int64_t>(tilingParam.reductionTilingSize),
          builder.getI32Type());
      auto redTileStart = builder.create<triton::SplatOp>(
          loc, redIndexTensorTy, redForLoop.getInductionVar());
      auto redTileOffset = builder.create<triton::MakeRangeOp>(
          loc, redIndexTensorTy, builder.getI32IntegerAttr(0),
          builder.getI32IntegerAttr(tilingParam.reductionTilingSize));
      auto redIndex1D =
          builder.create<arith::AddIOp>(loc, redTileStart, redTileOffset);
      auto redIndexSlice = builder.create<triton::ExpandDimsOp>(
          loc, redIndex1D, builder.getI32IntegerAttr(ParallelAxis));
      Value redIndex = builder.create<triton::BroadcastOp>(loc, indexTensorType,
                                                           redIndexSlice);
      SmallVector<Value, 2> argReplacements{parIndex, redIndex};
      builder.inlineBlockBefore(&innerFnBody, &loopBody, loopBody.end(),
                                argReplacements);
    }
    // FIXME: conversion of reduction-loop should not depends on parent op.
    auto computedBufferOp = op->getParentOfType<inductor::ComputedBufferOp>();
    auto buffer = bufferMap.at(computedBufferOp.getName()->strref());
    auto outTensorPtrType = RankedTensorType::get(
        (int64_t)tilingParam.parallelTilingSize, buffer.getType());
    auto bufferSplat =
        builder.create<triton::SplatOp>(loc, outTensorPtrType, buffer);
    auto outPtr = builder.create<triton::AddPtrOp>(loc, outTensorPtrType,
                                                   bufferSplat, parTileOffset);
    builder.create<triton::StoreOp>(loc, outPtr, redForLoop.getResults()[0],
                                    triton::CacheModifier::NONE,
                                    triton::EvictionPolicy::NORMAL);
    builder.eraseOp(op);
    return success();
  }
};

class ConvertInductorToTriton
    : public impl::ConvertInductorToTritonBase<ConvertInductorToTriton> {
  llvm::StringMap<Value> buffers;

  void populateLoopsConversionPatterns(MLIRContext *context,
                                       const TilingParam &tilingParam,
                                       LoopTilingTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
    patterns.add<YieldConversion>(context);
    patterns.add<ReductionLoopConversion>(tilingParam, buffers, context);
    patterns.add<ArithConstOpConversion>(typeConverter, context);
    patterns.add<InductorCalcPlainOffsetConversion>(typeConverter, context);
    patterns.add<InductorLoadOpConversion>(typeConverter, buffers, context);
    patterns.add<MathFloatUnaryOpConversion<math::SinOp>>(context);
    patterns.add<MathFloatUnaryOpConversion<math::CosOp>>(context);
    patterns.add<ArithFloatBinaryOpConversionBase<arith::AddFOp>>(context);
  }
  void convertReductionLoop(inductor::ReductionOp redLoop) {
    auto *context = redLoop.getContext();
    // TODO: we should be able to have per-loop tiling param.
    TilingParam tilingParam;
    LoopTilingTypeConverter typeConverter(tilingParam);
    LoopTilingConversionTarget convTarget(redLoop);
    RewritePatternSet convPatterns(context);
    populateLoopsConversionPatterns(context, tilingParam, typeConverter,
                                    convPatterns);
    if (failed(applyFullConversion(redLoop, convTarget,
                                   std::move(convPatterns)))) {
      // TODO:
      assert(false);
    }
  }
  void convertComputedBuffer(OpBuilder &builder,
                             inductor::StorageOpInterface storageOp,
                             inductor::ComputedBufferOp bufOp) {
    auto &bufferBody = bufOp.getDataRegion().front();
    for (auto &op : llvm::make_early_inc_range(bufferBody)) {
      auto loopOp = dyn_cast<inductor::LoopsOpInterface>(&op);
      if (!loopOp) {
        continue;
      }
      if (auto redLoop = dyn_cast<inductor::ReductionOp>(&op)) {
        convertReductionLoop(redLoop);
        continue;
      }
      assert(false);
    }
    auto &parentBB = *builder.getInsertionBlock();
    parentBB.getOperations().splice(parentBB.end(), bufferBody.getOperations());
    storageOp->erase();
  }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto *context = mod.getContext();
    UnknownLoc loc = UnknownLoc::get(context);
    auto &bodyBB = mod.getBodyRegion().front();
    OpBuilder builder(mod.getBodyRegion());
    SmallVector<Type> funcArgTypes;
    SmallVector<std::pair<StringRef, Type>> inputArgs, outputArgs;
    for (auto &op : llvm::make_early_inc_range(bodyBB)) {
      auto storageOp = dyn_cast<inductor::StorageOpInterface>(&op);
      if (!storageOp) {
        continue;
      }
      auto *dataOp = storageOp.getDataOp();
      assert(dataOp);
      if (isa<inductor::InputBufferOp>(dataOp)) {
        inputArgs.emplace_back(storageOp.getName(), storageOp.getDtype());
      } else {
        outputArgs.emplace_back(storageOp.getName(), storageOp.getDtype());
      }
    }
    llvm::StringMap<size_t> bufferArgIndex;
    for (const auto &kv :
         llvm::concat<std::pair<StringRef, Type>>(inputArgs, outputArgs)) {
      bufferArgIndex.try_emplace(kv.first, funcArgTypes.size());
      funcArgTypes.push_back(
          triton::PointerType::get(kv.second, GlobalAddressSpace));
    }
    auto funcTy = FunctionType::get(context, funcArgTypes, {});
    auto funcOp = builder.create<triton::FuncOp>(loc, "kernel", funcTy);
    SmallVector<Location> argLocs(funcArgTypes.size(), loc);
    auto *entryBB = builder.createBlock(
        &funcOp.getBody(), funcOp.getBody().end(), funcArgTypes, argLocs);
    for (const auto &kv : bufferArgIndex) {
      buffers[kv.getKey()] = entryBB->getArgument(kv.getValue());
    }
    builder.setInsertionPointToStart(entryBB);
    for (auto &op : llvm::make_early_inc_range(bodyBB)) {
      if (isa<triton::FuncOp>(&op)) {
        continue;
      }
      auto storageOp = dyn_cast<inductor::StorageOpInterface>(&op);
      if (!storageOp) {
        op.remove();
        builder.insert(&op);
      } else {
        auto *dataOp = storageOp.getDataOp();
        assert(dataOp);
        if (isa<inductor::InputBufferOp>(dataOp)) {
          op.erase();
          continue;
        }
        if (auto bufOp = dyn_cast<inductor::ComputedBufferOp>(dataOp)) {
          convertComputedBuffer(builder, storageOp, bufOp);
          continue;
        }
        op.remove();
        builder.insert(&op);
      }
    }
    builder.create<triton::ReturnOp>(loc, ValueRange{});
  }
};
} // namespace
std::unique_ptr<OperationPass<ModuleOp>> createConvertInductorToTritonPass() {
  return std::make_unique<ConvertInductorToTriton>();
}
} // namespace inductor
} // namespace mlir
