import torch
from inductor_mlir.dynamo_backend import inductor_mlir

#enable_debug(True)
#debug_only("dialect-conversion")

# Usage:
# step-1: clone this module
# step-2: git submodule update --init --recursive
# step-3: pip install -e python
# you may need to set env-vars like "export INDUCTOR_MLIR_LLVM_DIR=${HOME}/development/build_llvm/lib/cmake/llvm" to points to a LLVM install with MLIR.
# step-4: python python/example/1.py
# yes, it will coredump, but it will print IRs before coredump.

# Dumped Inductor-IR
#[ComputedBuffer(name='buf0', layout=FixedLayout('cuda', torch.float32, size=[10000], stride=[1]), data=Reduction(
#  'cuda',
#  torch.float32,
#  def inner_fn(index, rindex):
#      i0 = index
#      r0 = rindex
#      tmp0 = ops.load(arg0_1, to_plain_offset(0, index(i0, r0), stride(10000, 1), size(10000, 10000)))
#      tmp1 = ops.sin(tmp0)
#      tmp2 = ops.load(arg1_1, to_plain_offset(0, index(i0, r0), stride(10000, 1), size(10000, 10000)))
#      tmp3 = ops.cos(tmp2)
#      tmp4 = tmp1 + tmp3
#      return tmp4
#  ,
#  ranges=[10000],
#  reduction_ranges=[10000],
#  reduction_type=sum,
#  origin_node=sum_1,
#  origins={sin, sum_1, cos, add}
#))]

#Generated MLIR InductorDialect:
#module {
#  %c10000_i64 = arith.constant 10000 : i64
#  %c10000_i64_0 = arith.constant 10000 : i64
#  %c10000_i64_1 = arith.constant 10000 : i64
#  %c1_i64 = arith.constant 1 : i64
#  %c0_i64 = arith.constant 0 : i64
#  %0 = "inductor.plain_storage"(%c10000_i64, %c10000_i64_0, %c10000_i64_1, %c1_i64, %c0_i64) <{device = #inductor<device "cuda:0">, dtype = f32, is_fixed, name = "arg0_1", operandSegmentSizes = array<i32: 2, 2, 1>}> ({
#    "inductor.input_buffer"() <{name = "arg0_1"}> : () -> ()
#  }) : (i64, i64, i64, i64, i64) -> memref<10000x10000xf32>
#  %c10000_i64_2 = arith.constant 10000 : i64
#  %c10000_i64_3 = arith.constant 10000 : i64
#  %c10000_i64_4 = arith.constant 10000 : i64
#  %c1_i64_5 = arith.constant 1 : i64
#  %c0_i64_6 = arith.constant 0 : i64
#  %1 = "inductor.plain_storage"(%c10000_i64_2, %c10000_i64_3, %c10000_i64_4, %c1_i64_5, %c0_i64_6) <{device = #inductor<device "cuda:0">, dtype = f32, is_fixed, name = "arg1_1", operandSegmentSizes = array<i32: 2, 2, 1>}> ({
#    "inductor.input_buffer"() <{name = "arg1_1"}> : () -> ()
#  }) : (i64, i64, i64, i64, i64) -> memref<10000x10000xf32>
#  %c10000_i64_7 = arith.constant 10000 : i64
#  %c1_i64_8 = arith.constant 1 : i64
#  %c0_i64_9 = arith.constant 0 : i64
#  %2 = "inductor.plain_storage"(%c10000_i64_7, %c1_i64_8, %c0_i64_9) <{device = #inductor<device "cuda:0">, dtype = f32, is_fixed, name = "buf0", operandSegmentSizes = array<i32: 1, 1, 1>}> ({
#    "inductor.computed_buffer"() <{name = "buf0"}> ({
#      %c10000_i64_10 = arith.constant 10000 : i64
#      %c10000_i64_11 = arith.constant 10000 : i64
#      "inductor.reduction"(%c10000_i64_11, %c10000_i64_10) <{device = #inductor<device "cuda:0">, operandSegmentSizes = array<i32: 1, 1>, reduction_hint = #inductor<reduction_hint DEFAULT>, reduction_type = "sum"}> ({
#      ^bb0(%arg0: i64, %arg1: i64):
#        %c10000_i64_12 = arith.constant 10000 : i64
#        %c1_i64_13 = arith.constant 1 : i64
#        %c10000_i64_14 = arith.constant 10000 : i64
#        %c10000_i64_15 = arith.constant 10000 : i64
#        %c0_i64_16 = arith.constant 0 : i64
#        %3 = "inductor.ops_calc_plain_offset"(%arg0, %arg1, %c10000_i64_14, %c10000_i64_15, %c10000_i64_12, %c1_i64_13, %c0_i64_16) : (i64, i64, i64, i64, i64, i64, i64) -> i64
#        %4 = "inductor.ops_load"(%3) <{buffer = "arg0_1"}> : (i64) -> f32
#        %5 = math.sin %4 : f32
#        %c10000_i64_17 = arith.constant 10000 : i64
#        %c1_i64_18 = arith.constant 1 : i64
#        %c10000_i64_19 = arith.constant 10000 : i64
#        %c10000_i64_20 = arith.constant 10000 : i64
#        %c0_i64_21 = arith.constant 0 : i64
#        %6 = "inductor.ops_calc_plain_offset"(%arg0, %arg1, %c10000_i64_19, %c10000_i64_20, %c10000_i64_17, %c1_i64_18, %c0_i64_21) : (i64, i64, i64, i64, i64, i64, i64) -> i64
#        %7 = "inductor.ops_load"(%6) <{buffer = "arg1_1"}> : (i64) -> f32
#        %8 = math.cos %7 : f32
#        %9 = arith.addf %5, %8 : f32
#        "inductor.ops_yield"(%9) : (f32) -> ()
#      }) : (i64, i64) -> ()
#    }) : () -> ()
#  }) : (i64, i64, i64) -> memref<10000xf32>
#}

# Generated ttir
#module {
#  tt.func @kernel(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>) {
#    %c10000_i32 = arith.constant 10000 : i32
#    %c10000_i32_0 = arith.constant 10000 : i32
#    %c10000_i32_1 = arith.constant 10000 : i32
#    %c1_i32 = arith.constant 1 : i32
#    %c0_i32 = arith.constant 0 : i32
#    %c10000_i32_2 = arith.constant 10000 : i32
#    %c10000_i32_3 = arith.constant 10000 : i32
#    %c10000_i32_4 = arith.constant 10000 : i32
#    %c1_i32_5 = arith.constant 1 : i32
#    %c0_i32_6 = arith.constant 0 : i32
#    %c10000_i32_7 = arith.constant 10000 : i32
#    %c1_i32_8 = arith.constant 1 : i32
#    %c0_i32_9 = arith.constant 0 : i32
#    %c10000_i32_10 = arith.constant 10000 : i32
#    %c10000_i32_11 = arith.constant 10000 : i32
#    %0 = tt.get_program_id x : i32
#    %c1024_i32 = arith.constant 1024 : i32
#    %1 = arith.muli %0, %c1024_i32 : i32
#    %2 = tt.splat %1 : i32 -> tensor<1024xi32>
#    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
#    %4 = arith.addi %2, %3 : tensor<1024xi32>
#    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<1024xi32> -> tensor<1024x1xi32>
#    %6 = tt.broadcast %5 : tensor<1024x1xi32> -> tensor<1024x1024xi32>
#    %c0_i32_12 = arith.constant 0 : i32
#    %c1024_i32_13 = arith.constant 1024 : i32
#    %cst = arith.constant 0.000000e+00 : f32
#    %7 = tt.splat %cst : f32 -> tensor<1024xf32>
#    %8 = scf.for %arg3 = %c0_i32_12 to %c10000_i32_10 step %c1024_i32_13 iter_args(%arg4 = %7) -> (tensor<1024xf32>)  : i32 {
#      %11 = tt.splat %arg3 : i32 -> tensor<1024xi32>
#      %12 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
#      %13 = arith.addi %11, %12 : tensor<1024xi32>
#      %14 = tt.expand_dims %13 {axis = 0 : i32} : tensor<1024xi32> -> tensor<1x1024xi32>
#      %15 = tt.broadcast %14 : tensor<1x1024xi32> -> tensor<1024x1024xi32>
#      %c10000_i32_14 = arith.constant 10000 : i32
#      %16 = tt.splat %c10000_i32_14 : i32 -> tensor<1024x1024xi32>
#      %c1_i32_15 = arith.constant 1 : i32
#      %17 = tt.splat %c1_i32_15 : i32 -> tensor<1024x1024xi32>
#      %c10000_i32_16 = arith.constant 10000 : i32
#      %18 = tt.splat %c10000_i32_16 : i32 -> tensor<1024x1024xi32>
#      %c10000_i32_17 = arith.constant 10000 : i32
#      %19 = tt.splat %c10000_i32_17 : i32 -> tensor<1024x1024xi32>
#      %c0_i32_18 = arith.constant 0 : i32
#      %20 = tt.splat %c0_i32_18 : i32 -> tensor<1024x1024xi32>
#      %21 = arith.muli %6, %16 : tensor<1024x1024xi32>
#      %22 = arith.addi %20, %21 : tensor<1024x1024xi32>
#      %23 = arith.muli %15, %17 : tensor<1024x1024xi32>
#      %24 = arith.addi %22, %23 : tensor<1024x1024xi32>
#      %25 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<1024x1024x!tt.ptr<f32, 1>>
#      %26 = tt.addptr %25, %24 : tensor<1024x1024x!tt.ptr<f32, 1>>, tensor<1024x1024xi32>
#      %27 = tt.load %26 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024x1024xf32>
#      %28 = math.sin %27 : tensor<1024x1024xf32>
#      %c10000_i32_19 = arith.constant 10000 : i32
#      %29 = tt.splat %c10000_i32_19 : i32 -> tensor<1024x1024xi32>
#      %c1_i32_20 = arith.constant 1 : i32
#      %30 = tt.splat %c1_i32_20 : i32 -> tensor<1024x1024xi32>
#      %c10000_i32_21 = arith.constant 10000 : i32
#      %31 = tt.splat %c10000_i32_21 : i32 -> tensor<1024x1024xi32>
#      %c10000_i32_22 = arith.constant 10000 : i32
#      %32 = tt.splat %c10000_i32_22 : i32 -> tensor<1024x1024xi32>
#      %c0_i32_23 = arith.constant 0 : i32
#      %33 = tt.splat %c0_i32_23 : i32 -> tensor<1024x1024xi32>
#      %34 = arith.muli %6, %29 : tensor<1024x1024xi32>
#      %35 = arith.addi %33, %34 : tensor<1024x1024xi32>
#      %36 = arith.muli %15, %30 : tensor<1024x1024xi32>
#      %37 = arith.addi %35, %36 : tensor<1024x1024xi32>
#      %38 = tt.splat %arg1 : !tt.ptr<f32, 1> -> tensor<1024x1024x!tt.ptr<f32, 1>>
#      %39 = tt.addptr %38, %37 : tensor<1024x1024x!tt.ptr<f32, 1>>, tensor<1024x1024xi32>
#      %40 = tt.load %39 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024x1024xf32>
#      %41 = math.cos %40 : tensor<1024x1024xf32>
#      %42 = arith.addf %28, %41 : tensor<1024x1024xf32>
#      %43 = "tt.reduce"(%42) <{axis = 1 : i32}> ({
#      ^bb0(%arg5: f32, %arg6: f32):
#        %45 = arith.addf %arg5, %arg6 : f32
#        tt.reduce.return %45 : f32
#      }) : (tensor<1024x1024xf32>) -> tensor<1024xf32>
#      %44 = arith.addf %arg4, %43 : tensor<1024xf32>
#      scf.yield %44 : tensor<1024xf32>
#    }
#    %9 = tt.splat %arg2 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>>
#    %10 = tt.addptr %9, %3 : tensor<1024x!tt.ptr<f32, 1>>, tensor<1024xi32>
#    tt.store %10, %8 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
#    tt.return
#  }
#}


def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return (a + b).sum(1)


foo_compiled = torch.compile(foo, backend=inductor_mlir)

m = 10000
n = 10000

x = torch.randn(m, n, device='cuda')
y = torch.randn(m, n, device='cuda')

a = foo_compiled(x, y)
