import os
from .model import SWXGModel
#from . import test_wx

#__all__ = ["SWXGModel", "test_wx"]
__all__ = ["SWXGModel"]
if not os.getenv("READTHEDOCS"):
    from . import test_wx
    __all__.append("test_wx")
