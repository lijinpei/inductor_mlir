from contextlib import contextmanager, ExitStack


@contextmanager
def patch(target, attr, replacement):
    origin_target = getattr(target, attr)
    setattr(target, attr, replacement(origin_target))
    yield origin_target
    setattr(target, attr, origin_target)


all_patches = []


def register_patch(target, attr):

    def fn(repl):
        all_patches.append(patch(target, attr, repl))
        return repl

    return fn


@contextmanager
def patch_all():
    with ExitStack() as stack:
        for p in all_patches:
            stack.enter_context(p)
        yield
