class RAGException(Exception):
    """Base exception for RAG application"""

    pass


class VectorDBError(RAGException):
    """Raised when Vector DB operations fail"""

    pass


class LLMError(RAGException):
    """Raised when LLM operations fail"""

    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid"""

    pass
