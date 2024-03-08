#pragma once
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace inductor {

#define GEN_PASS_DECL
#include "inductor/Dialect/Inductor/Transforms/Passes.h.inc"

namespace wrapper {
enum ArgKind { AK_Index, AK_Integer, AK_MemRef };
struct ArgDesc {
  ArgKind kind;
  union {
    struct {
      int64_t rank;
    } memref;
    struct {
      int64_t size_and_signed;
    } integer;
  };
};
class WrapperSignature {
  std::vector<ArgDesc> args;

public:
  WrapperSignature(std::size_t reserveNum = 0) { args.reserve(reserveNum); }
  std::size_t getNumArgs() const { return args.size(); }
  const ArgDesc &getArg(std::size_t idx) const { return args[idx]; }
  void addArg(const ArgDesc &arg) { args.push_back(arg); }
  const std::vector<ArgDesc> &getArgs() const { return args; }
  std::string toStr() const;
  static WrapperSignature fromStr();
};
std::string generateWrapperDeclaration(const std::string &name,
                                       const WrapperSignature &signature);
struct WrapperInfo {
  std::string name;
  WrapperSignature signature;
  std::string generateDeclaration() const {
    return generateWrapperDeclaration(name, signature);
  }
};
} // namespace wrapper

std::unique_ptr<Pass>
createInductorGPUWrapperExtractionPass(std::string *wrapper);

#define GEN_PASS_REGISTRATION
#include "inductor/Dialect/Inductor/Transforms/Passes.h.inc"

} // namespace inductor
} // namespace mlir
