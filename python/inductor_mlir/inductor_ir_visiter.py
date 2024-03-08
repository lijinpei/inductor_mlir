from torch._inductor import ir as ti_ir
from typing import Mapping, Callable, Tuple
import torch._inductor as _inductor
import torch._inductor.dependencies
import torch._inductor.graph
import torch._inductor.virtualized
import torch._subclasses.fake_tensor
import inspect
from .utility import BoxedMLIRModule
import mlir.ir
from mlir.dialects import func, memref, inductor, _ods_common
import itertools
import contextlib
from .inductor_ops_handler import InnerFnOpsHandler

visiter_map: Mapping[ti_ir.IRNode, Callable[Tuple['InductorIRVisiter', ...],
                                            None]] = {}
visiter_map_cached: Mapping[ti_ir.IRNode, Callable[Tuple['InductorIRVisiter',
                                                         ...], None]] = {}


def get_visiter_for_type(tpe):
    visiter = visiter_map_cached.get(None)
    if visiter is not None:
        return visiter
    visiter = visiter_map.get(tpe, None)
    if visiter is not None:
        visiter_map_cached[tpe] = visiter
        return visiter
    for base in tpe.__bases__:
        visiter = get_visiter_for_type(base)
        if visiter is not None:
            visiter_map_cached[tpe] = visiter
            return visiter
    return None


def get_visiter(node: ti_ir.IRNode):
    return get_visiter_for_type(type(node))


def visit_impl(fn):
    node_type = inspect.signature(fn).parameters['node'].annotation
    visiter_map[node_type] = fn
    return fn


class InductorIRVisiter:
    debug_dump = True

    @classmethod
    def debug_print(cls, *args, **kw_args):
        if not cls.debug_dump:
            return
        print(*args, **kw_args)

    @staticmethod
    def get_buffer_name(buf):
        while True:
            old_buf = buf
            if isinstance(buf, ti_ir.MutableBox):
                buf = buf.data
            if isinstance(buf, ti_ir.Buffer):
                assert buf.name is not None
                return buf.name
            if isinstance(buf, str):
                return buf
            assert old_buf is not buf

    def __init__(self, graph: _inductor.graph.GraphLowering,
                 mlir_mod: BoxedMLIRModule):
        self.graph = graph
        self.mlir_mod = mlir_mod.module
        self.ctx = mlir_mod.ctx
        self.current_kernel = None
        self.buf_name_to_value = {}
        self.ops_handler = InnerFnOpsHandler(self)

    @contextlib.contextmanager
    def enter_kernel(self):
        self.debug_print('start_kernel:')
        self.debug_print(self.graph.graph_input_names)
        self.debug_print(self.graph.graph_inputs)
        self.debug_print(self.graph.graph_outputs)
        arg_types = []
        for node in self.graph.module.graph.nodes:
            if node.op != 'output':
                continue
            with self.ctx, mlir.ir.Location.unknown():
                arg_types.append(
                    self.get_mlir_memref_type_from_fake_tensor(
                        node.args[0][0].meta['val']))
        for in_arg in self.graph.example_inputs:
            if isinstance(in_arg, torch.Tensor):
                with self.ctx, mlir.ir.Location.unknown():
                    arg_types.append(
                        self.get_mlir_memref_type_from_fake_tensor(in_arg))
            elif isinstance(in_arg, torch.SymInt):
                arg_types.append(self.get_index_type())
            else:
                assert False
        func_type = mlir.ir.FunctionType.get(arg_types, [], self.ctx)
        with self.ctx, mlir.ir.Location.unknown(), mlir.ir.InsertionPoint(
                self.mlir_mod.body):
            kernel = self.current_kernel = func.FuncOp(self.graph.name,
                                                       func_type,
                                                       visibility='public')
            kernel_args = kernel.add_entry_block().arguments
        for idx, buf, in enumerate(
                itertools.chain(self.graph.graph_outputs,
                                self.graph.graph_input_names)):
            name = self.get_buffer_name(buf)
            self.buf_name_to_value[name] = kernel_args[idx]
        with self.ctx, mlir.ir.Location.unknown(), mlir.ir.InsertionPoint(
                next(iter(kernel.body))):
            yield
        self.debug_print(self.mlir_mod)

    def visit(self, node: ti_ir.IRNode):
        self.debug_print('before visit:')
        self.debug_print(node)
        visiter = get_visiter(node)
        assert visiter is not None
        visiter(self, node)

    def visit_buffer(self, buf: ti_ir.Buffer):
        assert isinstance(buf, ti_ir.Buffer)
        return self.visit(buf)

    def to_mlir_layout_attr(self, layout: ti_ir.Layout):
        stride = [self.graph.sizevars.size_hint(x) for x in layout.stride]
        assert all(map(lambda x: isinstance(x, int), stride))
        offset = self.graph.sizevars.size_hint(layout.offset)
        assert isinstance(offset, int)
        return mlir.ir.StridedLayoutAttr.get(offset, stride)

    def to_mlir_memref_type(self, layout: ti_ir.Layout):
        elem_type = self.to_mlir_dtype(layout.dtype)
        size = [self.graph.sizevars.size_hint(x) for x in layout.size]
        layout_attr = self.to_mlir_layout_attr(layout)
        return mlir.ir.MemRefType.get(size, elem_type, layout_attr)

    def visit_loops(self, buf: mlir.ir.Value,
                    read_writes: _inductor.dependencies.ReadWrites,
                    node: ti_ir.Loops):
        if isinstance(node, ti_ir.Pointwise):
            return self.visit_pointwise(buf, read_writes, node)
        elif isinstance(node, ti_ir.Reduction):
            return self.visit_reduction(buf, read_writes, node)
        else:
            assert isinstance(node, ti_ir.Scan)
            return self.visit_scan(buf, read_writes, node)

    def visit_pointwise(self, buf: mlir.ir.Value,
                        read_write_bufs: _inductor.dependencies.ReadWrites,
                        node: ti_ir.Pointwise):
        pass

    def visit_reduction(self, buf: mlir.ir.Value,
                        read_write_bufs: _inductor.dependencies.ReadWrites,
                        node: ti_ir.Reduction):
        assert len(read_write_bufs.writes) == 1
        print('node range: ', node.ranges)
        print('node reduction range: ', node.reduction_ranges)
        pointwise_loop = inductor.pointwise_loop([])
        return pointwise_loop

    def visit_scan(self, buf: mlir.ir.Value, node: ti_ir.Scan):
        pass

    @visit_impl
    def visit_input_buffer(self, node: ti_ir.InputBuffer):
        assert False

    @visit_impl
    def visit_computed_buffer(self, node: ti_ir.ComputedBuffer):
        assert node.name is not None
        read_write_bufs = node.get_read_writes()
        assert len(read_write_bufs.writes) == 1
        with self.ctx, mlir.ir.Location.unknown():
            memref_type = self.to_mlir_memref_type(node.layout)
            buf = _ods_common.get_op_result_or_value(
                memref.AllocOp(memref_type, [], []))
        loop = self.visit_loops(buf, read_write_bufs, node.data)
        return loop

    @visit_impl
    def visit_Template_buffer(self, node: ti_ir.TemplateBuffer):
        assert False

    @visit_impl
    def visit_inputs_kernel(self, node: ti_ir.InputsKernel):
        assert False

    def get_index_type(self):
        return mlir.ir.IndexType.get(context=self.ctx)

    def get_mlir_memref_type_from_node(self, node: ti_ir.IRNode):
        while True:
            old_node = node
            if isinstance(node, ti_ir.MutableBox):
                node = node.data
            if isinstance(node, ti_ir.Buffer):
                node = node.layout
            if node is old_node:
                break
        assert isinstance(node, ti_ir.Layout)
        elem_type = self.to_mlir_dtype(node.dtype)
        print('original size: ', node.size)
        print('original size: ', [type(x) for x in node.size])
        size = [self.graph.sizevars.size_hint(x) for x in node.size]
        print('size is: ', size)
        # FIXME why size_hint can get static size?
        # FIXME: how to determine which dimensions should be dynamic
        # FIXME: memref affine map
        # FIXME: address space
        return mlir.ir.MemRefType.get(size, elem_type)

    @contextlib.contextmanager
    def patch_fixed_layout_indexer(self):
        pass

    @contextlib.contextmanager
    def enter_emit_inner_fn(self):
        with self.patch_fixed_layout_indexer(
        ), _inductor.virtualized.V.set_ops_handler(self.ops_handler):
            yield
