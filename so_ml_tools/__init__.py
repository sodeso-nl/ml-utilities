from . import (
    data,
    multiclass,
    nlp,
    pandas,
    regression,
    tf,
    util
)

# We follow Semantic Versioning (https://semver.org/spec/v2.0.0.html)
_MAJOR_VERSION = '0'
_MINOR_VERSION = '1'
_PATCH_VERSION = '9'

__version__ = '.'.join([
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
])

__all__ = [
    "data",
    "multiclass",
    "nlp",
    "pandas",
    "regression",
    "tf",
    "util",
    "__version__"
]


