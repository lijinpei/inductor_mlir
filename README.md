# Inductor MLIR

## Introduction

Inductor-MLIR is a tool to export inductor's python define-by-run loop-level IR to MLIR's c++ linalg dialect generic-op.

The goals of this project includes the followings:

0. Complete. This project aims to be an end-to-end dynamo backend, with widespread operator support.
1. MLIR first. Although pytorch is python-frist, many compiler developers are more familiar with MLIR/C++. Inductor-mlir can bridge the gap by converting inductor python ir to linalg c++ ir, and leave out AOTAutoGrad, CUDAGraph etc.
2. For 'X'-PU.
    - Integrate At correct level:
        - After decomposition, lowering.
        - Before too much lowering, optimization.
        - Don't linearize offset calculation. This preserves the possibility to do layout transformations for 'X'-PU backends.
3. Small.
    - Reuse/steal inductor's lowering logic.
    - Small number of IR constructs.

### Inductor Dialect
