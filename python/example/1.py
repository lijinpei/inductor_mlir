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

# Dumped fx graph:
#graph():
#    %arg0_1 : [num_users=0] = placeholder[target=arg0_1]
#    %arg1_1 : [num_users=2] = placeholder[target=arg1_1]
#    %arg2_1 : [num_users=2] = placeholder[target=arg2_1]
#    %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%arg1_1,), kwargs = {})
#    %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%arg2_1,), kwargs = {})
#    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sin, %cos), kwargs = {})
#    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_2, [1]), kwargs = {})
#    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, 1), kwargs = {})
#    %sin_1 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%add,), kwargs = {})
#    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, 1), kwargs = {})
#    %cos_1 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%add_1,), kwargs = {})
#    %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sin_1, %cos_1), kwargs = {})
#    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sub, [0]), kwargs = {})
#    %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %sum_2), kwargs = {})
#    return (add_3,)

# Dumped Inductor-Python-IR
#ComputedBuffer(name='buf0', layout=FixedLayout('cpu', torch.float32, size=[s0], stride=[1]), data=Reduction(
#  'cpu',
#  torch.float32,
#  def inner_fn(index, rindex):
#      i0 = index
#      r0 = rindex
#      tmp0 = ops.load(arg1_1, r0 + i0 * s0)
#      tmp1 = ops.sin(tmp0)
#      tmp2 = ops.load(arg2_1, r0 + i0 * s0)
#      tmp3 = ops.cos(tmp2)
#      tmp4 = tmp1 + tmp3
#      return tmp4
#  ,
#  ranges=[s0],
#  reduction_ranges=[s0],
#  reduction_type=sum,
#  origin_node=sum_1,
#  origins={sum_1, cos, add_2, sin}
#))
#ComputedBuffer(name='buf1', layout=FixedLayout('cpu', torch.float32, size=[s0], stride=[1]), data=Reduction(
#  'cpu',
#  torch.float32,
#  def inner_fn(index, rindex):
#      i0 = index
#      r0 = rindex
#      tmp0 = ops.load(arg1_1, i0 + r0 * s0)
#      tmp1 = ops.constant(1, torch.float32)
#      tmp2 = tmp0 + tmp1
#      tmp3 = ops.sin(tmp2)
#      tmp4 = ops.load(arg2_1, i0 + r0 * s0)
#      tmp5 = ops.constant(1, torch.float32)
#      tmp6 = tmp4 + tmp5
#      tmp7 = ops.cos(tmp6)
#      tmp8 = tmp3 - tmp7
#      return tmp8
#  ,
#  ranges=[s0],
#  reduction_ranges=[s0],
#  reduction_type=sum,
#  origin_node=sum_2,
#  origins={sum_2, sub, cos_1, add_1, sin_1, add}
#))
#ComputedBuffer(name='buf2', layout=FixedLayout('cpu', torch.float32, size=[s0], stride=[1]), data=Pointwise(
#  'cpu',
#  torch.float32,
#  def inner_fn(index):
#      i0 = index
#      tmp0 = ops.load(buf0, i0)
#      tmp1 = ops.load(buf1, i0)
#      tmp2 = tmp0 + tmp1
#      return tmp2
#  ,
#  ranges=[s0],
#  origin_node=add_3,
#  origins={add_3}
#))

# Dumped Inductor-MLIR
##map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
##map1 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#module {
#  func.func public @my_kernel(%arg0: memref<?xf32, strided<[1]>>, %arg1: index, %arg2: memref<?x?xf32, strided<[?, 1]>>, %arg3: memref<?x?xf32, strided<[?, 1]>>) {
#    %idx1 = index.constant 1
#    %idx0 = index.constant 0
#    %alloca = memref.alloca(%arg1)[%idx0, %idx1] : memref<?xf32, #map>
#    "inductor.reduction_loop"(%alloca, %arg1, %arg1) <{operandSegmentSizes = array<i32: 1, 1, 1>, reduction_hint = #inductor<reduction_hint DEFAULT>, reduction_type = #inductor<reduction_type SUM>}> ({
#    ^bb0(%arg4: index, %arg5: index):
#      %0 = affine.apply #map1(%arg4, %arg5)[%idx0, %arg1, %idx1]
#      %1 = memref.load %arg2[%arg4, %arg5] : memref<?x?xf32, strided<[?, 1]>>
#      %2 = math.sin %1 : f32
#      %3 = affine.apply #map1(%arg4, %arg5)[%idx0, %arg1, %idx1]
#      %4 = memref.load %arg3[%arg4, %arg5] : memref<?x?xf32, strided<[?, 1]>>
#      %5 = math.cos %4 : f32
#      %6 = arith.addf %2, %5 : f32
#      "inductor.loopYield"(%6) : (f32) -> ()
#    }) : (memref<?xf32, #map>, index, index) -> ()
#    %alloca_0 = memref.alloca(%arg1)[%idx0, %idx1] : memref<?xf32, #map>
#    "inductor.reduction_loop"(%alloca_0, %arg1, %arg1) <{operandSegmentSizes = array<i32: 1, 1, 1>, reduction_hint = #inductor<reduction_hint DEFAULT>, reduction_type = #inductor<reduction_type SUM>}> ({
#    ^bb0(%arg4: index, %arg5: index):
#      %0 = affine.apply #map1(%arg5, %arg4)[%idx0, %arg1, %idx1]
#      %1 = memref.load %arg2[%arg5, %arg4] : memref<?x?xf32, strided<[?, 1]>>
#      %cst = arith.constant 1.000000e+00 : f32
#      %2 = arith.addf %1, %cst : f32
#      %3 = math.sin %2 : f32
#      %4 = affine.apply #map1(%arg5, %arg4)[%idx0, %arg1, %idx1]
#      %5 = memref.load %arg3[%arg5, %arg4] : memref<?x?xf32, strided<[?, 1]>>
#      %cst_1 = arith.constant 1.000000e+00 : f32
#      %6 = arith.addf %5, %cst_1 : f32
#      %7 = math.cos %6 : f32
#      %8 = arith.subf %3, %7 : f32
#      "inductor.loopYield"(%8) : (f32) -> ()
#    }) : (memref<?xf32, #map>, index, index) -> ()
#    "inductor.pointwise_loop"(%arg0, %arg1) ({
#    ^bb0(%arg4: index):
#      %0 = affine.apply #map(%arg4)[%idx0, %idx1]
#      %1 = memref.load %alloca[%arg4] : memref<?xf32, #map>
#      %2 = affine.apply #map(%arg4)[%idx0, %idx1]
#      %3 = memref.load %alloca_0[%arg4] : memref<?xf32, #map>
#      %4 = arith.addf %1, %3 : f32
#      "inductor.loopYield"(%4) : (f32) -> ()
#    }) : (memref<?xf32, strided<[1]>>, index) -> ()
#    return
#  }
#}

# Dumped C++ launcher
"""
#ifdef ENABLE_PYTHNON_EXT

#define ENABLE_TORCH_CXX_EXT

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "torch/csrc/autograd/python_variable.h"

#endif // ENABLE_PYTHNON_EXT

#ifdef __cplusplus
#include <cstdint>
#include <cassert>
#else
#include <stdint.h>
#include <assert.h>
#endif // __cplusplus

#ifdef ENABLE_TORCH_CXX_EXT
#include "ATen/ATen.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void my_kernel(void*, void*, int64_t, int64_t, int64_t, int64_t, void*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, void*, void*, int64_t, int64_t, int64_t, int64_t, int64_t);


#ifdef ENABLE_TORCH_CXX_EXT

void _torch_my_kernel(const at::Tensor& arg0, int64_t arg1, const at::Tensor& arg2, const at::Tensor& arg3) {
  auto arg0_sizes = arg0.sizes();
  auto arg0_strides = arg0.strides();
  int64_t arg0_dim = arg0.dim();
  assert(arg0_dim == 1);
  void* arg0_storage_ptr = arg0.at::TensorBase::storage().mutable_data();
  int64_t arg0_storage_offset = arg0.at::TensorBase::storage_offset();
  void* arg0_data_ptr = arg0.at::TensorBase::data_ptr();
  int64_t arg0_size0 = arg0_sizes[0];
  int64_t arg0_stride0 = arg0_strides[0];


  auto arg2_sizes = arg2.sizes();
  auto arg2_strides = arg2.strides();
  int64_t arg2_dim = arg2.dim();
  assert(arg2_dim == 2);
  void* arg2_storage_ptr = arg2.at::TensorBase::storage().mutable_data();
  int64_t arg2_storage_offset = arg2.at::TensorBase::storage_offset();
  void* arg2_data_ptr = arg2.at::TensorBase::data_ptr();
  int64_t arg2_size0 = arg2_sizes[0];
  int64_t arg2_stride0 = arg2_strides[0];
  int64_t arg2_size1 = arg2_sizes[1];
  int64_t arg2_stride1 = arg2_strides[1];

  auto arg3_sizes = arg3.sizes();
  auto arg3_strides = arg3.strides();
  int64_t arg3_dim = arg3.dim();
  assert(arg3_dim == 2);
  void* arg3_storage_ptr = arg3.at::TensorBase::storage().mutable_data();
  int64_t arg3_storage_offset = arg3.at::TensorBase::storage_offset();
  void* arg3_data_ptr = arg3.at::TensorBase::data_ptr();
  int64_t arg3_size0 = arg3_sizes[0];
  int64_t arg3_stride0 = arg3_strides[0];
  int64_t arg3_size1 = arg3_sizes[1];
  int64_t arg3_stride1 = arg3_strides[1];

  my_kernel(arg0_storage_ptr, arg0_data_ptr, arg0_storage_offset, arg0_size0, arg0_stride0,
arg1,
arg2_storage_ptr, arg2_data_ptr, arg2_storage_offset, arg2_size0, arg2_size1, arg2_stride0, arg2_stride1,
arg3_storage_ptr, arg3_data_ptr, arg3_storage_offset, arg3_size0, arg3_size1, arg3_stride0, arg3_stride1);
}

#endif // ENABLE_TORCH_CXX_EXT

#ifdef ENABLE_PYTHNON_EXT

static PyObject * _py_my_kernel(PyObject *self,
                         PyObject *const *args,
                         Py_ssize_t nargs) {
  auto& arg0 = THPVariable_Unpack(args[0]);
  auto arg1 = PyLong_AsLong(args[1]);
  auto& arg2 = THPVariable_Unpack(args[2]);
  auto& arg3 = THPVariable_Unpack(args[3]);
  _torch_my_kernel(arg0, arg1, arg2, arg3);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef _my_kernel_methods[] = {
	{ "call", _PyCFunction_CAST(_py_my_kernel), METH_FASTCALL,
	    "Call my_kernel." },
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef _my_kernel_module = {
	PyModuleDef_HEAD_INIT,
	"my_kernel",
	NULL,
	-1,
	_my_kernel_methods
};

PyMODINIT_FUNC
PyInit_my_kernel(void) {
  return PyModule_Create(&_my_kernel_module);
}

#endif // ENABLE_PYTHNON_EXT

#ifdef __cplusplus
} // extern "C"
#endif
"""


def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    c = torch.sin(x + 1)
    d = torch.cos(y + 1)
    return (a + b).sum(1) + torch.sum((c - d), dim=0)


foo_compiled = torch.compile(foo, backend=inductor_mlir, dynamic=True)

m = 10000
n = 10000

x = torch.randn(m, n)
y = torch.randn(m, n)

res_compiled = foo_compiled(x, y)
res = foo(x, y)
torch.testing.assert_close(res_compiled, res, rtol=5e-6, atol=1e-5)
