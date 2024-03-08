from .utility import BoxedMLIRModule
from .patch import patch
import torch._inductor as _inductor
import functools
from .importer import InductorImporter
import contextlib


def patch_graph_lowering_run(origin_target, result: BoxedMLIRModule):

    def graph_lowering_run(self, *args, **kw_args):
        importer = InductorImporter(self, result)
        with importer.into_run_graph():
            res = origin_target(self, *args, **kw_args)
        return res

    return graph_lowering_run


@contextlib.contextmanager
def export_to_mlir_after_run(result: BoxedMLIRModule):
    with patch(_inductor.graph.GraphLowering, 'run',
               functools.partial(patch_graph_lowering_run, result=result)):
        yield


@contextlib.contextmanager
def replace_schedule(input: BoxedMLIRModule):
    yield
