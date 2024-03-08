import torch

import torch._inductor as _inductor
import torch._inductor.ir

from mlir.ir import Value, IntegerType
from mlir.dialects import memref, arith, math

from typing import Union
from dataclasses import dataclass


@dataclass
class IndexExprWrapper:
    val: Value
    layout: _inductor.ir.Layout
    index: list[Value]


T = Value


class BuildMLIROpsHandler:

    def __init__(self, importer):
        self.importer = importer

    @staticmethod
    def unwrap(x):
        if isinstance(x, IndexExprWrapper):
            return x.val
        return x

    def load(self, name: str, index) -> T:
        """
        Load from the memory location 'name', offset by some indexing expression 'index'.
        """
        assert isinstance(index, IndexExprWrapper)
        assert (self.importer.graph.get_buffer(name).layout == index.layout)
        return memref.load(self.importer.buffer_map[name], index.index)

    def add(self, x0: T, x1: T) -> T:
        x0 = self.unwrap(x0)
        x1 = self.unwrap(x1)
        assert x0.type == x1.type
        if IntegerType.isinstance(x0.type):
            return arith.addi(x0, x1)
        else:
            return arith.addf(x0, x1)

    def constant(self, value: Union[bool, float, int],
                 dtype: torch.dtype) -> T:
        """Produces a scalar constant of type dtype."""
        elem_type = self.importer.to_mlir_dtype(dtype)
        if IntegerType.isinstance(elem_type):
            value = int(value)
        else:
            value = float(value)
        return arith.constant(elem_type, value)

    def cos(self, x0: T) -> T:
        return math.cos(x0)

    def sin(self, x0: T) -> T:
        return math.sin(x0)

    def sub(self, x0: T, x1: T) -> T:
        x0 = self.unwrap(x0)
        x1 = self.unwrap(x1)
        assert x0.type == x1.type
        if IntegerType.isinstance(x0.type):
            return arith.subi(x0, x1)
        else:
            return arith.subf(x0, x1)
