"""numba if available, otherwise a no-op decorator (pure Python: correct but ~100x slower; fine for tests)."""
try:
    from numba import njit as _njit
    HAVE_NUMBA = True

    def njit(*args, **kw):
        kw.setdefault("cache", True)
        return _njit(*args, **kw)
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kw):
        if len(args) == 1 and callable(args[0]) and not kw:
            return args[0]
        return lambda f: f
