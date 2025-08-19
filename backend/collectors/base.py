from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import polars as pl
import requests
from datetime import date, datetime
import time
import logging

from backend.config.settings import Config
from backend.utils.exceptions import APIError, DataProcessingError

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """Abstract base class for fantasy data collectors."""
    
    def __init__(self, config: Config, season: Optional[int] = None):
        self.config = config
        self.season = season or config.DEFAULT_SEASON
        self.session = requests.Session()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the collector."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _make_request(self, url: str, headers: Dict[str, str] = None, 
                     params: Dict[str, Any] = None, retries: int = None) -> requests.Response:
        """Make HTTP request with retry logic and error handling."""
        retries = retries or self.config.MAX_RETRIES
        headers = headers or {}
        
        for attempt in range(retries):
            try:
                logger.info(f"Making request to {url} (attempt {attempt + 1})")
                response = self.session.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=self.config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == retries - 1:
                    raise APIError(f"Failed to fetch data from {url}: {e}")
                time.sleep(self.config.RATE_LIMIT_DELAY * (attempt + 1))
    
    @abstractmethod
    def collect_data(self) -> pl.DataFrame:
        """Collect data from the source and return as DataFrame."""
        pass
    def save_data(self, df: pl.DataFrame, filename_prefix: str) -> str:
        """Save DataFrame to parquet file."""
        filename = f"{filename_prefix}-{date.today()}.parquet"
        filepath = f"{self.config.DATA_DIR}/{filename}"
        
        # Add metadata columns
        df_with_metadata = df.with_columns([
            pl.lit(date.today()).cast(pl.Date).alias('date_collected'),
            pl.lit(self.__class__.__name__.lower().replace('collector', '')).alias('source')
        ])
        
        df_with_metadata.write_parquet(filepath)
        logger.info(f"Data saved to {filepath}")
        return filepath