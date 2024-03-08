from torch._dynamo import register_backend


@register_backend
def inductor_mlir(*args, **kwargs):
    # do import here to avoid loading inductor into memory when it is not used
    from torch._inductor.compile_fx import compile_fx
    from .monkey_patch import patch_all
    with patch_all():
        return compile_fx(*args, **kwargs)
