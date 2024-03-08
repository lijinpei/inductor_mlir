from torch._dynamo import register_backend
import torch.fx

from .graph import GraphLowering
from .patch import patch_class

#def fx_codegen_and_compile_patch(inductor_fn):
#
#    def inner(
#        gm: torch.fx.GraphModule,
#        example_inputs: List[torch.Tensor],
#        cudagraphs: Optional[BoxedBool] = None,
#        num_fixed: int = 0,
#        is_backward: bool = False,
#        graph_id: Optional[int] = None,
#        cpp_wrapper: bool = False,
#        aot_mode: bool = False,
#        is_inference: bool = False,
#        # Use a dict with None value rather than a set for deterministic
#        # iteration order just in case.
#        user_visible_outputs: Optional[Dict[str, None]] = None,
#        layout_opt: Optional[bool] = None,
#        extern_node_serializer: Optional[Callable[[List[ExternKernelNode]],
#                                                  Any]] = None,
#    ):
#        export_mlir_module(gm, example_inputs, cudagraphs, num_fixed,
#                           is_backward, graph_id, cpp_wrapper, aot_mode,
#                           is_inference, user_visible_outputs, layout_opt,
#                           extern_node_serializer)
#        return inductor_fn(gm, example_inputs, cudagraphs, num_fixed,
#                           is_backward, graph_id, cpp_wrapper, aot_mode,
#                           is_inference, user_visible_outputs, layout_opt,
#                           extern_node_serializer)
#
#    return inner


@register_backend
def inductor_mlir(*args, **kwargs):
    from torch._inductor.compile_fx import compile_fx
    with patch_class(torch._inductor.compile_fx, 'GraphLowering',
                     GraphLowering), patch_class(torch._inductor.graph,
                                                 'GraphLowering',
                                                 GraphLowering):
        return compile_fx(*args, **kwargs)
