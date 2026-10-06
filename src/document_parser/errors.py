class DocumentParseError(RuntimeError):
    """A document cannot be loaded or parsed."""


class UnsupportedFormatError(DocumentParseError):
    pass


class DependencyUnavailableError(DocumentParseError):
    pass
