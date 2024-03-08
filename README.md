# Inductor MLIR

## Introduction

Inductor-MLIR is a tool to export inductor's define-by-run IR to MLIR. The principles of this project includes the followings:

1. Don't linearize offset calculation. Currently, this is achieved by monkey-patching inductor, I hope in the future, this part can be contributed back to pytorch.
2. Integrate at an early point. This means to export inductor's IR right after lowering, and before scheduling, loop-fusion, loop-coalescence. Those optimizations can be done at MLIR level, and they make offset calculation complex, and impossible to extract coordinates.
3. Keep a small integration surface for backend. For developer of a new backend of torch.compile, the 2K+ torch operators are too much. And in some situation, even linalg/HLO operators are too much to support. Triton/Inductor-IR has much fewer operators to support.
4. Reuse pytorch/inductor's lowering logic. As a developer for new pytorch backend, I need to depend on pytorch/inductor community to lower its ops to inductor-ir.

## How does it work?

Offset/Coordinate calculation:
Monkey patching.

### Inductor Dialect
