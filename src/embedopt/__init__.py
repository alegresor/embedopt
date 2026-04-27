"""Public API for EmbedOpt."""

import sys
from importlib import metadata as _metadata
from pathlib import Path

try:  # pragma: no cover - executed only in installed environments
    __version__ = _metadata.version("embedopt")
except _metadata.PackageNotFoundError:  # pragma: no cover
    # Package metadata is unavailable when running from a source checkout.
    __version__ = "0.0.0"

# Get Protenix installation directory (works with editable install)
try:
    import protenix

    PROTENIX_DIR = Path(protenix.__file__).parent.parent
    # Add Protenix to sys.path so runner/ and configs/ can be imported
    if str(PROTENIX_DIR) not in sys.path:
        sys.path.insert(0, str(PROTENIX_DIR))
except ImportError:
    # Fallback if protenix not installed
    PROTENIX_DIR = None

__all__ = ["PROTENIX_DIR", "__version__"]
