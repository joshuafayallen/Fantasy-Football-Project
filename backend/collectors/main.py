from typing import Dict
import polars as pl
from backend.config.settings import Config
from backend.collectors.espncollector import ESPNCollector
from backend.collectors.yahoocollector import YahooCollector

class FantasyDataCollector:
    """Wrapper to collect fantasy data from multiple sources."""

    def __init__(self, config: Config):
        self.config = config
        self.espn_collector = ESPNCollector(config)
        self.yahoo_collector = YahooCollector(config)

    def collect_all(self) -> Dict[str, pl.DataFrame]:
        """Collect data from all sources and return as a dict keyed by source."""
        data_dict: Dict[str, pl.DataFrame] = {}

        # ESPN data
        espn_df = self.espn_collector.collect_data()
        data_dict['espn'] = espn_df

        # Yahoo data
        yahoo_df = self.yahoo_collector.collect_data()
        data_dict['yahoo'] = yahoo_df

        return data_dict
