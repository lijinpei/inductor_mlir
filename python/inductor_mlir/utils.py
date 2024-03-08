from typing import TypeVar, List, Generic
from mlir.ir import Value, ShapedType, AffineMap, AffineConstantExpr, AffineDimExpr, AffineSymbolExpr, AffineMapAttr, MemRefType
from mlir.dialects import index
import sympy
from sympy import Expr
from dataclasses import dataclass
import torch._inductor as _inductor

T = TypeVar('T', Value, Expr)

KDynamic = ShapedType.get_dynamic_size()


class MaybeStaticBase(Generic[T]):

    def __init__(self, s_val: int, d_val: T):
        self.s_val = s_val
        self.d_val = d_val

    @property
    def is_dynamic(self):
        return self.s_val == KDynamic

    @property
    def is_static(self):
        return not self.is_dynamic()

    def impl_(self, other, m):
        if self.s_val == KDynamic or other.s_val == KDynamic:
            s_val = KDynamic
        else:
            s_val = getattr(int, '__' + m + '__')(self.s_val, other.s_val)
        d_val = getattr(self, m)(self.d_val, other.d_val)
        return type(self)(s_val, d_val)

    def __add__(self, other):
        self.impl_(other, 'add')

    def __sub__(self, other):
        self.impl_(other, 'sub')

    def __mul__(self, other):
        self.impl_(other, 'mul')


class MaybeStaticValue(MaybeStaticBase[Value]):

    def __init__(self, expr, emitter):
        if isinstance(expr, int):
            s_val = expr
            d_val = emitter.emit_index_constant(expr)
        elif isinstance(expr, sympy.Integer):
            s_val = int(expr)
            d_val = emitter.emit_index_constant(s_val)
        else:
            s_val = KDynamic
            d_val = emitter.emit_index_expr(expr)
        super().__init__(s_val, d_val)

    @staticmethod
    def add(v1, v2):
        return index.add(v1, v2)

    @staticmethod
    def sub(v1, v2):
        return index.sub(v1, v2)

    @staticmethod
    def mul(v1, v2):
        return index.mul(v1, v2)


class MaybeStaticExpr(MaybeStaticBase[Expr]):

    def __init__(self, expr):
        if isinstance(expr, int):
            s_val = expr
            d_val = (expr)
        elif isinstance(expr, sympy.Integer):
            s_val = int(expr)
            d_val = expr
        else:
            s_val = KDynamic
            d_val = expr
        super().__init__(s_val, d_val)

    @staticmethod
    def add(v1, v2):
        return v1 + v2

    @staticmethod
    def sub(v1, v2):
        return v1 - v2

    @staticmethod
    def mul(v1, v2):
        return v2 * v2


MaybeStaticT = TypeVar('MaybeStaticT', MaybeStaticValue, MaybeStaticExpr)


@dataclass
class StridedBufferLayout(Generic[MaybeStaticT]):
    layout: _inductor.ir.Layout
    sizes = List[MaybeStaticT]
    strides = List[MaybeStaticT]
    offset = MaybeStaticT
    symbol_operands: List[T]
    affine_map: AffineMap

    def __init__(self, layout, sizes, strides, offset):
        self.layout = layout
        self.sizes = sizes
        self.strides = strides
        self.offset = offset
        symbol_operands = []

        def add_symbol_op(x):
            res = AffineSymbolExpr.get(len(symbol_operands))
            symbol_operands.append(x)
            return res

        if offset.is_dynamic:
            expr = add_symbol_op(offset.d_val)
        else:
            expr = AffineConstantExpr.get(offset.s_val)

        for idx, stride in enumerate(strides):
            if stride.is_dynamic:
                aff_expr = add_symbol_op(stride.d_val)
            else:
                aff_expr = AffineConstantExpr.get(stride.s_val)
            expr = expr + aff_expr * AffineDimExpr.get(idx)
        self.symbol_operands = symbol_operands
        self.affine_map = AffineMap.get(len(strides), len(symbol_operands),
                                        [expr])

    def get_static_size_all(self):
        return [x.s_val for x in self.sizes]

    def get_dynamic_size_filtered(self):
        return [x.d_val for x in self.sizes if x.is_dynamic]

    def get_memref_type(self, type_conv):
        elem_type = type_conv.to_mlir_dtype(self.layout.dtype)
        return MemRefType.get(self.get_static_size_all(), elem_type,
                              AffineMapAttr.get(self.affine_map))

    @classmethod
    def create_impl_(cls, layout, conv_expr):
        sizes = [conv_expr(x) for x in layout.size]
        if layout.stride is None:
            cur_stride = conv_expr(1)
            strides = [None] * len(sizes)
            for idx in range(len(sizes), -1, -1):
                strides[idx] = cur_stride
                cur_stride *= sizes[idx]
        else:
            strides = [conv_expr(x) for x in layout.stride]
        offset = conv_expr(layout.offset)
        return StridedBufferLayout(layout, sizes, strides, offset)

    @classmethod
    def create_with_value(cls, layout, emitter):
        return cls.create_impl_(layout, lambda x: MaybeStaticValue(x, emitter))

    @classmethod
    def create_with_expr(cls, layout):
        return cls.create_impl_(layout, lambda x: MaybeStaticExpr(x))
