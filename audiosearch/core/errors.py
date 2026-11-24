"""Exceptions shared across layers (the API maps them to HTTP status codes)."""


class AudioSearchError(Exception):
    """Base class for this project's errors."""


class CollectionError(AudioSearchError, ValueError):
    """Invalid or unknown collection name."""


class RerankUnavailable(AudioSearchError):
    """Reranking could not run right now (rate limit, network, missing key)."""


class UnknownProviderError(AudioSearchError, ValueError):
    """A factory was asked for a name nobody registered."""


class MissingApiKeyError(AudioSearchError, RuntimeError):
    """A hosted provider was called without its key in .env."""
