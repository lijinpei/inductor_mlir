#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include <string>
#include <vector>

class InductorIRBuilder {
  mlir::MLIRContext &context;
  std::vector<mlir::ModuleOp> top_level_modules;
  mlir::OpBuilder builder{&context};

public:
  InductorIRBuilder(mlir::MLIRContext &context);
  mlir::OpBuilder &getBuilder() { return builder; }
  mlir::MLIRContext *getContext() { return &context; }

  int64_t getKDynamic();

  /// Insertion Point
  mlir::OpBuilder::InsertPoint getInsertPoint();
  void setInsertPoint(mlir::OpBuilder::InsertPoint);

  /// Location

  mlir::Location getFileLineColLoc(const std::string &file, unsigned line,
                                   unsigned column);
  mlir::Location getUnknownLoc();

  /// Attribute
  mlir::Attribute getReductionHintAttr(int64_t redHint);

  /// Type
  mlir::Type getF32Type();
  mlir::Type getMemRefType(const std::vector<int64_t> &shape,
                           mlir::Type elemType);

  /// Operation
  mlir::OpState createModuleOp(mlir::Location loc,
                               std::optional<std::string> name);

  /// Debug Dump
  void dump();
};
