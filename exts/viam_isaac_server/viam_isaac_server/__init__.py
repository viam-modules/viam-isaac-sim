try:
    import omni.ext  # noqa: F401
except ImportError:
    # outside of Kit (unit tests) only the pure-python parts are importable
    pass
else:
    from .extension import ViamIsaacServerExtension  # noqa: F401
