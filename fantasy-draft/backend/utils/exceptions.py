class FantasyDataError(Exception):
    """Base exception for fantasy data collection errors."""
    pass

class APIError(FantasyDataError):
    """Raised when API requests fail."""
    pass

class DataProcessingError(FantasyDataError):
    """Raised when data processing fails."""
    pass

class ValidationError(FantasyDataError):
    """Raised when data validation fails."""
    pass