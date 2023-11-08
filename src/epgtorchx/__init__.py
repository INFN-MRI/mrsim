"""Main EPG-Torch-X package"""

import sys as _sys

if _sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    _dist_name = "epgtorchx"
    __version__ = version(_dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from . import bloch as _bloch
from . import phantoms as _phantoms
from . import optim
from . import regression

from .bloch import *  # noqa
from .bloch import base
from .bloch import ops
from .phantoms import *  # noqa

__all__ = []
__all__.extend(_bloch.__all__)
__all__.extend(_phantoms.__all__)
