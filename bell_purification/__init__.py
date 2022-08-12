"""Estimate the Bell state purification scheme performance and resource cost."""

# Add imports here
from .purification import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
