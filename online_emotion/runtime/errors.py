"""Error taxonomy for online inference packages.

Every error raised by this package derives from :class:`OnlineInferenceError`,
so callers can catch the whole family with a single ``except``.
"""
from __future__ import annotations


class OnlineInferenceError(Exception):
    """Base class for all errors raised by this package."""


class UnknownModelError(OnlineInferenceError):
    """The requested model family is not registered."""


class UnknownWeightsError(OnlineInferenceError):
    """The requested weights key is not known for the chosen family."""


class WeightsNotAvailableError(OnlineInferenceError):
    """Weights cannot be auto-downloaded; the user must place them manually.

    The message explains exactly what to download, where to put it, and how to
    pass the path via ``weights=<path>``.
    """


class RuntimeUnavailableError(OnlineInferenceError):
    """The requested runtime/backend is unavailable or incompatible."""


class ExportError(OnlineInferenceError):
    """An export step (torchscript / onnx / trt) failed."""


class DeviceUnavailableError(OnlineInferenceError):
    """The requested device is not available on this machine."""


class SourceError(OnlineInferenceError):
    """A video/stream source could not be opened or read."""
