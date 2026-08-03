"""XQT error types."""

from xdl.errors import XDLError


class XQTError(XDLError):
    """Base error for XQT."""


class XQTConfigError(XQTError):
    """Invalid XQT configuration."""


class XQTPipelineError(XQTError):
    """Pipeline pass execution failed."""


class XQTRegistryError(XQTError):
    """XQT registry lookup or registration failed."""


class XQTArtifactError(XQTError):
    """Artifact manifest or checksum operation failed."""


class XQTBackendError(XQTError):
    """Optional backend dependency or execution failed."""


__all__ = [
    "XQTArtifactError",
    "XQTBackendError",
    "XQTConfigError",
    "XQTError",
    "XQTPipelineError",
    "XQTRegistryError",
]
