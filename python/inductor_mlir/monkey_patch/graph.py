from inductor_mlir._C.inductor_mlir.ir_builder import MLIRContext
from inductor_mlir._C.inductor_mlir.passes import PassManager, add_convert_inductor_to_triton
from ..importer import InductorImporter
from .context_manager import register_patch
import torch._inductor.graph


@register_patch(torch._inductor.graph.GraphLowering, 'run')
def patch_graph_lowering_run(origin_target):

    def graph_lowering_run(self, *args, **kw_args):
        mlir_context = MLIRContext()
        importer = InductorImporter(mlir_context, self)
        with importer.into_graph_run():
            res = origin_target(self, *args, **kw_args)
        importer.ir_builder.dump()
        pass_manager = PassManager(mlir_context)
        add_convert_inductor_to_triton(pass_manager)
        pass_manager.run(importer.mod)
        importer.ir_builder.dump()
        return res

    return graph_lowering_run
