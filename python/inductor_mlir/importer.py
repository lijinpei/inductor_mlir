import torch._inductor as _inductor
import torch._inductor.ir as ti_ir
import contextlib
from .patch import patch as patch
from .utility import to_plain_offset, BoxedMLIRModule
from .inductor_ir_visiter import InductorIRVisiter


def fixed_layout_make_indexer_patch(origin_target):

    def fixed_layout_make_indexer(self):

        def indexer(index):
            print('indexer called', index, self.stride, self.size)
            assert len(index) == len(self.stride) == len(self.size)
            return to_plain_offset(self.offset, *index, *self.stride,
                                   *self.size)

        return indexer

    return fixed_layout_make_indexer


class InductorImporter:
    wrapper_registry = {}

    def __init__(self, graph: _inductor.graph.GraphLowering,
                 mlir_mod: BoxedMLIRModule):
        self.graph = graph
        self.mlir_mod = mlir_mod

    @contextlib.contextmanager
    def into_run_graph(self):
        with patch(ti_ir.FixedLayout, 'make_indexer',
                   fixed_layout_make_indexer_patch):
            print('var_to_val', self.graph.sizevars.var_to_val)
            print('start patch fixed layout')
            yield
            print('finish patch fixed layout')
        self.finish_graph_run()

    #def __init__(self, mlir_context, graph):
    #    self.mlir_context = mlir_context
    #    self.graph = graph
    #    self.ir_builder = IRBuilder(mlir_context)
    #    self.unknown_loc = self.ir_builder.create_unknown_loc()
    #    self.mod = self.ir_builder.create_module_op(self.unknown_loc, None)
    #    self.kdynamic = self.ir_builder.get_kdynamic()
    #    self.buffer_to_tensor_map = {}
    #    self.loop_to_buffer_map = {}
    #    self.loop_to_tensor_map = {}
    #    self.graph_input_names = []
    #    self.node_to_op_map = {}

    def finish_graph_run(self):
        ir_visiter = InductorIRVisiter(self.graph, self.mlir_mod)
        with ir_visiter.enter_kernel():
            for buf in self.graph.buffers:
                ir_visiter.visit_buffer(buf)

        #for buf_name in self.graph_input_names:
        #    tensor = self.graph.graph_inputs[buf_name]
        #    assert isinstance(tensor, _inductor.ir.TensorBox) and isinstance(
        #        tensor.data, _inductor.ir.StorageBox) and isinstance(
        #            tensor.data.data, _inductor.ir.InputBuffer)
        #    self.visit_node(tensor)
        #for buffer in self.graph.buffers:
        #    storage = self.buffer_to_tensor_map.get(id(buffer), None)
        #    if storage is None:
        #        assert isinstance(buffer, _inductor.ir.Loops)
        #        storage = self.loop_to_tensor_map[id(buffer)]
        #    self.visit_node(storage)


#    def emit_expr(self, expr) -> inductor_mlir.Value:
#        if isinstance(expr, list):
#            return [self.emit_expr(x) for x in expr]
#        if isinstance(expr, int):
#            return self.ir_builder.create_constant_i32_op(
#                self.unknown_loc, expr).result(0)
#        if isinstance(expr, sympy.Integer):
#            int_val = int(expr)
#            return self.ir_builder.create_constant_i32_op(
#                self.unknown_loc, int_val).result(0)
#        assert False
#
#    def get_constant_int(self, node):
#        if isinstance(node, int):
#            return node
#        if isinstance(node, sympy.Integer):
#            return int(node)
#        return None
#
#    def get_dev_str(self, dev):
#        return str(dev)
#
#    def visit_node(self, node):
#        key = id(node)
#        res = self.node_to_op_map.get(key, None)
#        if res is not None:
#            return res
#        wrapper_cls = self.get_wrapper(node)
#        wrapper = wrapper_cls(self, node)
#        wrapper.verify(self, node)
#        res = wrapper.to_mlir(self, node)
#        self.node_to_op_map[key] = res
#        return res
#
#    def emit_layout_common_args(self, layout_node):
#        dev_str = self.get_dev_str(layout_node.device)
#        dtype = self.to_mlir_type(layout_node.dtype)
#        static_size = []
#        for size in layout_node.size:
#            constant_size = self.get_constant_int(size)
#            if constant_size is not None:
#                static_size.append(constant_size)
#            else:
#                static_size.append(self.kdynamic)
#        size = self.emit_expr(layout_node.size)
#        stride = self.emit_expr(layout_node.stride)
#        offset = self.emit_expr(layout_node.offset)
#        return (dev_str, dtype, static_size, size, stride, offset)
#
#    @contextlib.contextmanager
#    def keep_insertion_point(self):
#        ip = self.ir_builder.get_insertion_point()
#        yield
#        self.ir_builder.set_insertion_point(ip)
#
#
#class WrapperBase:
#
#    def __init__(self, imp, node):
#        self.loc = self.extract_location(imp, node)
#
#    def extract_location(self, imp, node):
#        return imp.unknown_loc
#
#    def verify(self, imp, node):
#        """
#            Verify the invariants of this inductor-ir node.
#        """
#        pass
#
#
#class InnerFnBuilder:
#
#    def __init__(self, imp, entry_bb, num_range_args, num_red_args):
#        assert entry_bb.get_num_arguments() == num_range_args + num_red_args
#        self.entry_bb = entry_bb
#        self.range_args = [entry_bb.arg(i) for i in range(num_range_args)]
#        self.red_args = [
#            entry_bb.arg(i + num_range_args) for i in range(num_red_args)
#        ]
#        self.saved_insertion_point = imp.ir_builder.get_insertion_point()
#        self.imp = imp
#
#    @contextlib.contextmanager
#    def enter_inner_fn(self, imp):
#
#        def patch_fixed_layout_make_indexer(origin_target):
#
#            def fixed_layout_make_indexer(self):
#
#                def indexer(index):
#                    assert len(index) == len(self.stride) == len(self.size)
#                    stride = [imp.emit_expr(x) for x in self.stride]
#                    size = [imp.emit_expr(x) for x in self.size]
#                    offset = imp.emit_expr(self.offset)
#                    # FIXME: really implements this
#                    return imp.ir_builder.create_calc_plain_offset_op(
#                        imp.unknown_loc, index, size, stride, offset).result(0)
#
#                return indexer
#
#            return fixed_layout_make_indexer
#
#        imp.ir_builder.set_insertion_point_to_start(self.entry_bb)
#        ops_handler = InnerFnOpsHandler(self.imp)
#        with V.set_ops_handler(ops_handler), patch(
#                torch._inductor.ir.FixedLayout, 'make_indexer',
#                patch_fixed_layout_make_indexer):
#            yield
#        imp.ir_builder.set_insertion_point(self.saved_insertion_point)
#
#
#@register_wrapper(_inductor.ir.Reduction)
#class Reduction(WrapperBase):
#
#    def __init__(self, imp, node):
#        super().__init__(imp, node)
#
#    def to_mlir(self, imp, node):
#        reduction_ranges = imp.emit_expr(node.reduction_ranges)
#        reduction_type = node.reduction_type
#        src_dtype = imp.to_mlir_type(node.src_dtype)
#        reduction_hint = int(node.reduction_hint.value)
#        device = imp.get_dev_str(node.device)
#        dtype = imp.to_mlir_type(node.dtype)
#        # FIXME: src_dtype
#        print(src_dtype, dtype)
#        ranges = imp.emit_expr(node.ranges)
#        res = imp.ir_builder.create_reduction_op(imp.unknown_loc, device,
#                                                 ranges, reduction_ranges,
#                                                 reduction_type,
#                                                 reduction_hint)
#        inner_fn_region = imp.ir_builder.get_reduction_op_inner_fn(res)
#        inner_fn_entry_bb = inner_fn_region.front()
#        num_range_arg = len(ranges)
#        num_red_arg = len(reduction_ranges)
#        inner_fn_builder = InnerFnBuilder(imp, inner_fn_entry_bb,
#                                          num_range_arg, num_red_arg)
#        with inner_fn_builder.enter_inner_fn(imp):
#            #TODO: create yield-op
#            value = node.inner_fn(inner_fn_builder.range_args,
#                                  inner_fn_builder.red_args)
#            imp.ir_builder.create_yield_op(imp.unknown_loc, value.value)
#        return res
#
#
#@register_wrapper(_inductor.ir.TensorBox)
#class TensorBox(WrapperBase):
#
#    def __init__(self, imp, node):
#        super().__init__(imp, node)
#
#    def to_mlir(self, imp, node):
#        # Ignore TensorBox
#        return imp.visit_node(node.data)
#
#
#@register_wrapper(_inductor.ir.StorageBox)
#class StorageBox(WrapperBase):
#
#    def __init__(self, imp, node):
#        super().__init__(imp, node)
#
#    def to_mlir(self, imp, node):
#        data_node = node.data
#        if isinstance(data_node, _inductor.ir.Buffer):
#            layout_node = data_node.layout
#            name = data_node.name
#            layout_args = imp.emit_layout_common_args(layout_node)
#            layout_kind = type(layout_node).__name__
#            view_or_target = imp.ir_builder.empty_value()
#            if isinstance(layout_node, _inductor.ir.AliasedLayout):
#                view_or_target = imp.visit_node(layout_node.view).result(0)
#            elif isinstance(layout_node, _inductor.ir.MutationLayout):
#                view_or_target = imp.visit_node(layout_node.target).result(0)
#            storage_op = imp.ir_builder.create_storage_op_from_layout(
#                self.loc, layout_kind, *layout_args, view_or_target, name)
#            with imp.keep_insertion_point():
#                body_bb = imp.ir_builder.get_entry_bb_of_region(storage_op, 0)
#                imp.ir_builder.set_insertion_point_to_start(body_bb)
#                imp.visit_node(data_node)
#            return storage_op
#        else:
#            assert False
#
#
#@register_wrapper(_inductor.ir.ComputedBuffer)
#class ComputedBuffer(WrapperBase):
#
#    def __init__(self, imp, node):
#        super().__init__(imp, node)
#
#    def to_mlir(self, imp, node):
#        res = imp.ir_builder.create_computed_buffer_op(imp.unknown_loc,
#                                                       node.name)
#        with imp.keep_insertion_point():
#            entry_bb = imp.ir_builder.get_entry_bb_of_region(res, 0)
#            imp.ir_builder.set_insertion_point_to_start(entry_bb)
#            imp.visit_node(node.data)
#        return res
#
#
#@register_wrapper(_inductor.ir.InputBuffer)
#class InputBuffer(WrapperBase):
#
#    def __init__(self, imp, node):
#        super().__init__(imp, node)
#
#    def to_mlir(self, imp, node):
#        res = imp.ir_builder.create_input_buffer_op(self.loc, node.name)
#        return res
