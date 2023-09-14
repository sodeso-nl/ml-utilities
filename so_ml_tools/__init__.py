from . import (
    data,
    nlp,
    pd,
    tf,
    util,
    evaluate
)

# We follow Semantic Versioning (https://semver.org/spec/v2.0.0.html)
_MAJOR_VERSION = '0'
_MINOR_VERSION = '1'
_PATCH_VERSION = '28'

__version__ = '.'.join([
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
])

__all__ = [
    "data",
    "nlp",
    "pd",
    "tf",
    "util",
    "evaluate",
    "__version__"
]


