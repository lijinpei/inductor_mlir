#import sympy
#from enum import OrderedEnum
#
#
#class AffineLevel(OrderedEnum):
#    Constant = 0
#    Symbol = 1
#    Dim = 2
#    Affine = 3
#    NonAffine = 4
#
#
#class InnerFnAffineAnalyzer:
#
#    def __init__(self, par_indexes, red_indexes):
#        self.input_buffers = []
#        self.output_buffers = []
#        self.par_dims = [
#            self.new_dim('par_idx' + str(x)) for x in range(len(par_indexes))
#        ]
#        self.red_dims = [
#            self.new_dim('red_idx' + str(x)) for x in range(len(par_indexes))
#        ]
#        self.next_none_affine = 0
#        self.non_affine_desc = []
#        self.dims = set()
#        self.non_affine_syms = set()
#
#    def get_affine_level(self, expr):
#        expr = sympy.simplify(expr)
#        if isinstance(expr, int):
#            return AffineLevel.Constant
#        if isinstance(expr, sympy.Add, sympy.Sub):
#            res = AffineLevel.Constant
#            for arg in expr.args:
#                arg_level = self.get_affine_level(arg)
#                res = max(arg_level, res)
#                if res == AffineLevel.NonAffine:
#                    break
#            return res
#        if isinstance(expr, sympy.Mul):
#            res = AffineLevel.Constant
#            for arg in expr.args:
#                arg_level = self.get_affine_level(arg)
#                if arg_level == AffineLevel.NonAffine:
#                    return AffineLevel.NonAffine
#                if arg_level >= AffineLevel.Dim and res >= AffineLevel.Dim:
#                    return AffineLevel.NonAffine
#                res = max(arg_level, res)
#            return res
#        if isinstance(expr, sympy.Div, sympy.Mod):
#            assert len(expr.args) == 2
#            arg0_level = self.get_affine_level(expr.args[0])
#            if arg0_level == AffineLevel.NonAffine:
#                return AffineLevel.NonAffine
#            arg1_level = self.get_affine_level(expr.args[1])
#            if arg1_level >= AffineLevel.Dim:
#                return AffineLevel.NonAffine
#            elif arg1_level == AffineLevel.Symbol:
#                return AffineLevel.Affine
#            else:
#                return arg0_level
#        assert isinstance(expr, sympy.Symbol)
#        if expr in self.non_affine_syms:
#            return AffineLevel.NonAffine
#        if expr in self.dims:
#            return AffineLevel.Dim
#        # treat value define outside of inner-fn as symbolic
#        return AffineLevel.Symbol
#
#    def is_affine(self, expr):
#        return self.get_affine_level(expr) <= AffineLevel.Affine
#
#    def new_none_affine(self, desc):
#        name = 'none_affine_' + str(self.next_none_affine)
#        self.next_none_affine += 1
#        self.non_affine_desc.append(desc)
#        res = sympy.Dummy(name)
#        self.non_affine_syms.add(res)
#        return res
#
#    def new_dim(self, name):
#        res = sympy.Dummy(name, integer=True, nonnegative=True)
#        self.dims.add(res)
#        return res
#
#    def load(self, name: str, index_):
#        self.output_buffers.append((name, index))
#        return self.new_none_affine('load(' + str(name) + ')')
#
#    def sin(self, x0):
#        return self.new_none_affine('sin(' + str(x0) + ')')
#
#    def cos(self, x0):
#        return self.new_none_affine('cos(' + str(s0) + ')')
#
#    def add(self, x0, x1):
#        return sympy.sympify(x0) + sympy.sympify(x1)
