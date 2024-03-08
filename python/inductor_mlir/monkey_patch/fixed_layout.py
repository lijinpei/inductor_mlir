import sympy

index_func = sympy.Function('index')
stride_func = sympy.Function('stride')
size_func = sympy.Function('size')
to_plain_offset_func = sympy.Function('to_plain_offset')


def patch_fixed_layout_make_indexer(origin_target):

    def fixed_layout_make_indexer(self):

        def indexer(index):
            assert len(index) == len(self.stride) == len(self.size)
            coord = index_func(*index)
            stride = stride_func(*self.stride)
            size = size_func(*self.size)
            return to_plain_offset_func(self.offset, coord, stride, size)

        return indexer

    return fixed_layout_make_indexer
