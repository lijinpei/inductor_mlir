import torch.fx

import torch._inductor.graph
from .mlir_importer import MLIRImporter

InductorGraphLowering = torch._inductor.graph.GraphLowering


class GraphLowering(InductorGraphLowering):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args):
        # args are SymNode and FakeTensor
        res = super().run(*args)
        self.mlir_importer = MLIRImporter(self)
        self.mlir_importer.import_module(args)
        return res

    def compile_to_fn(self):
        #def func(*args, **kwargs):
        #    print('func called', args, kwargs)
        #    return torch.empty((2,3), dtype=torch.int64)
        #return func
        return self.mlir_importer.compile_to_fn()
        return super().compile_to_fn()


#
#    def __init__(self,
#                 model: torch.fx.GraphModule,
#                 example_inputs: List[torch.Tensor],
#                 shape_env,
#                 name=None):
#        super().__init__(model)
#        self.model = model
#        self.example_inputs = example_inputs
#        if shape_env is None:
#            shape_env = ShapeEnv()
#            self.reuse_shape_env = False
#        else:
#            self._shape_env = shape_env
#            self.reuse_shape_env = True
#        self.sizevars = SizeVarAllocator(shape_env)
#        self.name = name
#        self.mlir_ctx = None
#        self.mlir_mod = None
#        self.buffers: List[ir.Buffer] = []
#        self.name_to_buffer: Dict[str, ir.Buffer] = {}
#
#    def qualify_name(self, name: str) -> str:
#        """Prepend the given name with the graph name if any."""
#        if self.name is not None:
#            return f"{self.name}_{name}"
#        return name
#
#    def register_buffer(self, buffer: ir.Buffer):
#        name = self.qualify_name(f"buf{len(self.buffers)}")
#        self.buffers.append(buffer)
#        self.name_to_buffer[name] = buffer
#        return name
#
#
#
#
#    def run(self, *args, **kw_args):
#        return super().run(*args, **kw_args)
#
#    def run_node(self, node: torch.fx.Node):
#        return super().run_node(node)
#
#    def placeholder(self, target: str, args, kwargs):
#        example = super().placeholder(target, args, kwargs)
#        if isinstance(example, SymTypes):
#            return example.node.expr
#        elif isinstance(example, (int, bool, float)):
#            return sympy.sympify(example)
#        if isinstance(example, BackwardState):
#            # Ignored arg, must be unused
#            # Alternately we could filter this out in AotAutograd
#            return None
#        assert isinstance(example, torch.Tensor), example
#
#        def symbolic_sizes(xs):
#            return [
#                x.node.expr
#                if isinstance(x, torch.SymInt) else sympy.Integer(x)
#                for x in xs
#            ]
#
#        sizes = symbolic_sizes(example.size())
#        strides = symbolic_sizes(example.stride())
#        target = self.qualify_name(target)
#        tensor = ir.TensorBox.create(
#            ir.InputBuffer(
#                target,
#                ir.FixedLayout(example.device, example.dtype, sizes, strides),
#            ))
#        return tensor
#
#    def call_function(self, target, args: Tuple, kwargs: Dict):
#        if target is operator.getitem and isinstance(args[0],
#                                                     (list, tuple, dict)):
#            return super().call_function(target, args, kwargs)
#
#        if hasattr(target, "_inductor_lowering_function"):
#            # passthrough lowerings from .pattern_matcher
#            return target(*args, **kwargs)
#
#        if target not in lowerings:
#            assert isinstance(
#                target,
#                torch._ops.OpOverload), f"{target} is not an OpOverload"
#            base_name = target.name().split(".")[0]
#            if base_name in FALLBACK_ALLOW_LIST:
#                make_fallback(target)
#            elif get_decompositions([target]):
#                # There isn't a good way to dynamically patch this in
#                # since AOT Autograd already ran.  The error message tells
#                # the user how to fix it.
#                raise MissingOperatorWithDecomp(target, args, kwargs)
#            else:
#                raise MissingOperatorWithoutDecomp(target, args, kwargs)
#
#        try:
#            out = lowerings[target](*args, **kwargs)
#            return out
#        except Exception as e:
#            raise LoweringException(e, target, args, kwargs).with_traceback(
#                e.__traceback__) from None
#
#    def call_module(self, target, args, kwargs):
#        raise AssertionError
#
#    def call_method(self, target, args, kwargs):
#        raise AssertionError
#
#    def output(self, target, args, kwargs):
#        result = super().output(target, args, kwargs)
#        if not isinstance(result, (tuple, list)):
#            # nested subgraphs can have singleton outputs
#            result = (result, )
#        assert isinstance(result, (tuple, list)), type(result)
#        assert all(
#            isinstance(
#                x,
#                (
#                    ir.TensorBox,
#                    ir.Constant,
#                    type(None),
#                    ir.ConstantBuffer,
#                    sympy.Expr,
#                    sympy.logic.boolalg.Boolean,
#                    int,
#                    # ir.EffectfulKernel,
#                ),
#            ) for x in result), result
#        for buf in self.buffers:
#            buf.decide_layout()
