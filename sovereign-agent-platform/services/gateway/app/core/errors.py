class ServiceError(Exception):
    """Base exception for service-level failures."""


class PolicyError(ServiceError):
    """Raised when policy decision cannot be obtained."""


class ValidationError(ServiceError):
    """Raised when external data fails schema validation."""
