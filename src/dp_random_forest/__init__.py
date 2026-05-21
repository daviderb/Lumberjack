import sys

try:
    if sys.version_info[:2] >= (3, 8):
        from importlib.metadata import PackageNotFoundError, version
    else:
        from importlib_metadata import PackageNotFoundError, version
except ImportError:
    PackageNotFoundError = None
    version = None

if version is None:
    __version__ = "unknown"
else:
    try:
        dist_name = "dp-random-forest"
        __version__ = version(dist_name)
    except PackageNotFoundError:
        __version__ = "unknown"
    finally:
        del version, PackageNotFoundError

from dp_random_forest.algorithms import (
    DiPriMeFlipForestClassifier,
    SNRDPForestClassifier,
)
