class DataFetchError(Exception):
    """Exception raised for errors during data fetching."""
    pass

class FallbackWarning(Warning):
    """Warning raised when falling back to a secondary data source."""
    pass
