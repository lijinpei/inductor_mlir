import torch._inductor as _inductor
import torch._inductor.graph
from torch._inductor.utils import output_node
from torch._inductor.virtualized import V
import torch._subclasses
from mlir.ir import Context, Module, Location, InsertionPoint, ShapedType, F32Type, F16Type, BF16Type, IntegerType, StridedLayoutAttr, MemRefType, IndexType, FunctionType, Block, AffineMapAttr
from mlir.dialects import affine, func, inductor, memref, index, arith
import torch.utils.cpp_extension

import contextlib
from typing import List
import sympy
import functools

from .patch import patch_class
from .inductor_ops_handler import BuildMLIROpsHandler, IndexExprWrapper

from .utils import StridedBufferLayout


class MLIRImporter:

    reduction_type_map = {
        "any": inductor.ReductionType.ANY,
        "max": inductor.ReductionType.MAX,
        "min": inductor.ReductionType.MIN,
        "prod": inductor.ReductionType.PROD,
        "sum": inductor.ReductionType.SUM,
        "xor_sum": inductor.ReductionType.XOR_SUM,
        "argmin": inductor.ReductionType.ARGMIN,
        "argmax": inductor.ReductionType.ARGMAX,
        "welford_combine": inductor.ReductionType.WELFORD_COMBINE,
    }

    reduction_hint_map = {
        _inductor.ir.ReductionHint.INNER: inductor.ReductionHint.INNER,
        _inductor.ir.ReductionHint.OUTER: inductor.ReductionHint.OUTER,
        _inductor.ir.ReductionHint.OUTER_TINY:
        inductor.ReductionHint.OUTER_TINY,
        _inductor.ir.ReductionHint.DEFAULT: inductor.ReductionHint.DEFAULT,
    }

    def __init__(self, graph: _inductor.graph.GraphLowering):
        self.graph = graph

    @functools.cache
    def get_kernel_name(self):
        if self.graph.name is not None:
            return self.graph.name
        return 'anonymous_kernel'

    def import_module(self, args):
        print('import_mlir_module')
        print(self.graph.graph)
        print('example_inputs', self.graph.example_inputs)
        self.mlir_ctx = Context()
        with self.mlir_ctx, self.loc_from_graph(self.graph):
            self.mlir_mod = Module.create()
            with InsertionPoint(self.mlir_mod.body):
                fx_out = output_node(self.graph)
                self.create_kernel(fx_out, self.graph.example_inputs)
                with InsertionPoint.at_block_terminator(
                        self.kernel.entry_block):
                    for buffer in self.graph.buffers:
                        self.visit_buffer(buffer)
        print(self.mlir_mod)
        print(self.buffer_map)
        print(self.symbol_map)
        asm_code, wrapper_code = inductor.compile_module(
            self.mlir_ctx, self.mlir_mod)
        print('asm: ', asm_code)
        print('wrapper: ', wrapper_code)
        self.compiled_module = load_inline(
            self.get_kernel_name(), [(asm_code, 's'), (wrapper_code, 'cpp')],
            extra_cflags=['-g', '-DENABLE_PYTHNON_EXT'],
            extra_ldflags=["-L", "/home/lijinpei/", "-lruntime"],
            verbose=True)

    def codegen(self):
        print('codegen')

    def loc_from_graph(self, model: torch.fx.GraphModule):
        # TODO:
        return Location.unknown()

    def create_kernel(self, out_node, example_inputs: List[torch.Tensor]):
        output_types = []
        assert len(out_node.args) == 1
        fx_out_args = out_node.args[0]
        num_outputs = len(fx_out_args)
        assert num_outputs == len(self.graph.graph_outputs)
        for ir_out in self.graph.graph_outputs:
            output_types.append(
                StridedBufferLayout.create_with_expr(
                    ir_out.data.layout).get_memref_type(self))
        assert len(example_inputs) == len(self.graph.graph_inputs)
        input_types = []
        for in_name in self.graph.graph_input_names:
            in_arg = self.graph.graph_inputs[in_name]
            if isinstance(in_arg, sympy.Symbol):
                input_types.append(self.get_index_type())
            else:
                assert isinstance(in_arg, _inductor.ir.TensorBox)
                input_types.append(
                    StridedBufferLayout.create_with_expr(
                        in_arg.layout).get_memref_type(self))
        func_type = FunctionType.get(input_types, output_types, self.mlir_ctx)
        self.kernel = kernel = func.FuncOp(self.get_kernel_name(),
                                           func_type,
                                           visibility='public')
        self.kernel_insertion_point = None
        kernel_entry_bb = kernel.add_entry_block()
        kernel_args = kernel_entry_bb.arguments
        self.buffer_map = {}
        self.symbol_map = {}
        self.index_constant_map = {}
        self.layout_map = {}
        for idx, (in_name, fx_in) in enumerate(
                zip(self.graph.graph_input_names, example_inputs)):
            in_arg = self.graph.graph_inputs[in_name]
            if isinstance(in_arg, sympy.Symbol):
                assert isinstance(fx_in, torch.SymInt)
                fx_expr = fx_in.node.expr
                assert isinstance(fx_expr, sympy.Symbol)
                assert in_arg.name == fx_expr.name
                self.symbol_map[in_arg.name] = kernel_args[idx]
            else:
                assert isinstance(in_arg, _inductor.ir.TensorBox)
                assert isinstance(fx_in, torch.Tensor)
                self.buffer_map[in_arg.get_name()] = kernel_args[idx]
        with InsertionPoint.at_block_begin(kernel_entry_bb):
            out_buffers = []
            for idx, (ir_out, fx_out) in enumerate(
                    zip(self.graph.graph_outputs, fx_out_args)):
                print('type: ', ir_out)
                buffer = self.get_or_alloc_buffer(ir_out.data,
                                                  is_scratch=False)
                self.buffer_map[fx_out.name] = buffer
                #out_buffers.append(memref.cast(output_types[idx], buffer))
                out_buffers.append(buffer)
            func.return_(out_buffers)

        self.ops_handler = None

    def visit_buffer(self, buffer):
        print(buffer)
        if isinstance(buffer, _inductor.ir.ComputedBuffer):
            return self.visit_computed_buffer(buffer)
        assert False

    def visit_computed_buffer(self, buffer):
        buf_val = self.get_or_alloc_buffer(buffer)
        loop = buffer.data
        if isinstance(loop, _inductor.ir.Pointwise):
            self.visit_pointwise_loop(buf_val, loop)
        elif isinstance(loop, _inductor.ir.Reduction):
            self.visit_reduction_loop(buf_val, loop)
        else:
            assert False

    def visit_pointwise_loop(self, buf_val, loop):
        ranges = [self.emit_index_expr(x) for x in loop.ranges]
        pointwise_op = inductor.PointwiseLoopOp(buf_val, ranges)
        arg_types = [x.type for x in ranges]
        with self.save_kernel_ip():
            entry_bb = Block.create_at_start(pointwise_op.inner_fn, arg_types)
            with InsertionPoint.at_block_begin(entry_bb):
                with self.enter_inner_fn():
                    res = loop.inner_fn(list(entry_bb.arguments))
                    inductor.loopYield(res.value)

    @contextlib.contextmanager
    def save_kernel_ip(self):
        self.kernel_insertion_point = InsertionPoint.current
        yield
        self.kernel_insertion_point = None

    def visit_reduction_loop(self, buf_val, loop):
        red_type = self.convert_reduction_type(loop.reduction_type)
        red_hint = self.convert_reduction_hint(loop.reduction_hint)
        ranges = [self.emit_index_expr(x) for x in loop.ranges]
        red_ranges = [self.emit_index_expr(x) for x in loop.reduction_ranges]
        reduction_loop_op = inductor.ReductionLoopOp(buf_val, ranges,
                                                     red_ranges, red_type,
                                                     red_hint)
        arg_types = [x.type for x in ranges + red_ranges]
        with self.save_kernel_ip():
            entry_bb = Block.create_at_start(reduction_loop_op.inner_fn,
                                             arg_types)
            with InsertionPoint.at_block_begin(entry_bb):
                with self.enter_inner_fn():
                    num_par_args = len(ranges)
                    res = loop.inner_fn(
                        list(entry_bb.arguments[:num_par_args]),
                        list(entry_bb.arguments[num_par_args:]))
                    inductor.loopYield(res.value)

    @contextlib.contextmanager
    def enter_inner_fn(self):

        def make_indexer(layout_):
            layout = self.get_layout(layout_)

            def indexer(index):
                assert len(index) == len(layout_.stride) == len(layout_.size)
                val = affine.apply(AffineMapAttr.get(layout.affine_map),
                                   index + layout.symbol_operands)
                return IndexExprWrapper(val, layout_, index)

            return indexer

        with patch_class(_inductor.ir.FixedLayout, 'make_indexer',
                         make_indexer), V.set_ops_handler(
                             self.get_ops_handler()):
            yield

    def get_ops_handler(self):
        if self.ops_handler is None:
            self.ops_handler = BuildMLIROpsHandler(self)
        return self.ops_handler

    def convert_reduction_type(self, reduction_type: str):
        res = self.reduction_type_map.get(reduction_type, None)
        assert res is not None
        return res

    def convert_reduction_hint(self,
                               reduction_hint: _inductor.ir.ReductionHint):
        res = self.reduction_hint_map.get(reduction_hint, None)
        assert res is not None
        return res

    def get_or_alloc_buffer(self, buffer, is_scratch=True):
        name = buffer.get_name()
        buf_val = self.buffer_map.get(name, None)
        if buf_val is not None:
            return buf_val
        buffer_layout = buffer.layout
        layout = self.get_layout(buffer_layout)
        memref_type = layout.get_memref_type(self)
        alloc_op = memref.alloca if is_scratch else memref.alloc
        alloca = alloc_op(memref_type, layout.get_dynamic_size_filtered(),
                          layout.symbol_operands)
        self.buffer_map[name] = alloca
        return alloca

    def get_layout(self, layout):
        res = self.layout_map.get(id(layout), None)
        if res is not None:
            return res
        ctx = self.kernel_insertion_point if self.kernel_insertion_point is not None else contextlib.nullcontext(
        )
        with ctx:
            res = StridedBufferLayout.create_with_value(layout, self)
        self.layout_map[id(layout)] = res
        return res

    def emit_index_expr(self, expr):
        if isinstance(expr, int):
            return self.emit_index_constant(expr)
        if isinstance(expr, sympy.Integer):
            return self.emit_index_constant(int(expr))
        if isinstance(expr, sympy.Symbol):
            return self.symbol_map[expr.name]
        if isinstance(expr, sympy.Mul):
            res = self.emit_index_expr(expr.args[0])
            for arg in expr.args[1:]:
                arg_ = self.emit_index_expr(arg)
                res = arith.muli(res, arg_)
            return res
        assert False

    def emit_index_constant(self, expr):
        res = self.index_constant_map.get(expr, None)
        if res is None:
            res = index.constant(expr)
            self.index_constant_map[expr] = res
        return res

    def get_mlir_memref_type_from_fake_tensor(
            self, tensor: torch._subclasses.FakeTensor):

        def to_shaped_type_size(x):
            if isinstance(x, torch.SymInt):
                return ShapedType.get_dynamic_size()
            else:
                assert isinstance(x, int)
                return x

        elem_type = self.to_mlir_dtype(tensor.dtype)
        size = [to_shaped_type_size(x) for x in tensor.size()]
        stride = [to_shaped_type_size(x) for x in tensor.stride()]
        layout = StridedLayoutAttr.get(offset=0,
                                       strides=stride,
                                       context=self.mlir_ctx)
        return MemRefType.get(size, elem_type, layout)

    def get_index_type(self):
        return IndexType.get(context=self.mlir_ctx)

    def to_mlir_dtype(self, dtype: torch.dtype):
        if dtype == torch.float32:
            return F32Type.get(context=self.mlir_ctx)
        elif dtype == torch.float16:
            return F16Type.get(context=self.mlir_ctx)
        elif dtype == torch.bfloat16:
            return BF16Type.get(context=self.mlir_ctx)
        elif dtype == torch.uint8:
            return IntegerType.get_unsigned(8, context=self.mlir_ctx)
        elif dtype == torch.int8:
            return IntegerType.get_signed(8, context=self.mlir_ctx)
        elif dtype == torch.int16:
            return IntegerType.get_signed(16, context=self.mlir_ctx)
        elif dtype == torch.int32:
            return IntegerType.get_signed(32, context=self.mlir_ctx)
        elif dtype == torch.int64:
            return IntegerType.get_signed(32, context=self.mlir_ctx)
        assert False

    def call_tuple(self, t):
        print('call_tuple args: ', t)
        res = torch.empty((t[0], ), dtype=torch.float32)
        self.compiled_module.call(res, *t)
        return res

    def compile_to_fn(self):
        print('before compile_to_fn')
        print(self.mlir_mod)
        #inductor.run_jit(self.mlir_ctx, self.mlir_mod)
        return self.call_tuple


def load_inline(name,
                sources,
                cuda_sources=None,
                functions=None,
                extra_cflags=None,
                extra_cuda_cflags=None,
                extra_ldflags=None,
                extra_include_paths=None,
                build_directory=None,
                verbose=False,
                with_cuda=None,
                is_python_module=True,
                with_pytorch_error_handling=True,
                keep_intermediates=True):
    import os
    from torch.utils.cpp_extension import _jit_compile, _get_build_directory, _maybe_write
    build_directory = build_directory or _get_build_directory(name, verbose)

    if not isinstance(sources, list):
        sources = [sources]

    sources_paths = []
    for idx, (source, ext) in enumerate(sources):
        cpp_source_path = os.path.join(build_directory,
                                       'source{}.{}'.format(idx, ext))
        _maybe_write(cpp_source_path, source)
        sources_paths.append(cpp_source_path)

    return _jit_compile(name,
                        sources_paths,
                        extra_cflags,
                        extra_cuda_cflags,
                        extra_ldflags,
                        extra_include_paths,
                        build_directory,
                        verbose,
                        with_cuda,
                        is_python_module,
                        is_standalone=False,
                        keep_intermediates=keep_intermediates)
