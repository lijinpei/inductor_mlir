from contextlib import contextmanager


@contextmanager
def patch(target, attr, replacement):
    origin_target = getattr(target, attr)
    setattr(target, attr, replacement(origin_target))
    yield origin_target
    setattr(target, attr, origin_target)


@contextmanager
def patch_class(target, attr, replacement):
    origin_target = getattr(target, attr)
    setattr(target, attr, replacement)
    yield origin_target
    setattr(target, attr, origin_target)
