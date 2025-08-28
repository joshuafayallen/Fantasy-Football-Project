from typing import List, Dict, Any, Optional
import polars as pl
from datetime import date
import time
import logging

from backend.config.settings import Config
from backend.utils.exceptions import DataProcessingError
from backend.collectors.base import BaseCollector 

logger = logging.getLogger(__name__)

class YahooCollector(BaseCollector):
    BASE_URL = (
        "https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2/"
        "league/461.l.public;out=settings/players;"
    )
    """Scraper for Yahoo fantasy data"""

    def __init__(self, config: Config, season: int = None):
        super().__init__(config, season)
    
    def collect_data(self, positions: Optional[List[str]] = None) -> pl.DataFrame:
        """Collect Yahoo fantasy data for multiple positions."""
        
        positions = positions or self.config.YAHOO_POSITIONS
        all_players: List[Dict[str, Any]] = []

        for idx, pos in enumerate(positions):
            if idx > 0:
                time.sleep(self.config.RATE_LIMIT_DELAY)  # Rate limit between positions

            count = self.config.YAHOO_POSITION_LIMITS.get(pos)
            if count is None:
                logger.warning(f"No position limit defined for {pos}, skipping.")
                continue

            url = (
                f"{self.BASE_URL}"
                f"position={pos};start=0;count={count};"
                f"sort=expert_rank;search=;"
                f"out=auction_values,ranks;ranks=season;ranks_by_position=season;"
                f"out=expert_ranks;expert_ranks.rank_type=projected_season_remaining/"
                f"draft_analysis;cut_types=diamond;slices=last7days?format=json_f"
            )

            headers = {
                "Accept": "*/*",
                "Host": "pub-api-ro.fantasysports.yahoo.com",
                "Origin": "https://football.fantasysports.yahoo.com",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "no-cors",
                "Sec-Fetch-Site": "none",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": "Josh Allen PhD",
            }

            response = self._make_request(url, headers=headers)
            try:
                data = response.json()
                players_data = data['fantasy_content']['league']['players']
            except Exception as e:
                logger.warning(f"Failed to parse data for {pos}: {e}")
                continue

            for player_wrapper in players_data:
                player = player_wrapper.get('player')
                if not player:
                    continue
                try:
                    all_players.append(self._parse_player_data(player))
                except Exception as e:
                    logger.warning(f"Skipping {pos} player due to parse error: {e}")

        if not all_players:
            raise DataProcessingError("No Yahoo player data collected.")

        return pl.from_dicts(all_players).with_columns(
            pl.lit(date.today()).cast(pl.Date).alias('date_queried'),
            pl.lit('yahoo').alias('source')
        )

    def _parse_player_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Yahoo player dict into structured player data."""
        try:
            return {
                "player": raw_data["name"]["full"],
                "first_name": raw_data["name"]["first"],
                "last_name": raw_data["name"]["last"],
                "team_abr": raw_data.get("editorial_team_abbr"),
                "bye": raw_data.get("bye_weeks", {}).get("week"),
                "position": raw_data["display_position"],
                "projected_auction_value": raw_data.get("projected_auction_value"),
                "average_pick_yahoo": raw_data["draft_analysis"].get("average_pick"),
                "average_cost_yahoo": raw_data["draft_analysis"].get("average_cost"),
                "yahoo_rank": raw_data["player_ranks"][0]["player_rank"]["rank_value"],
            }
        except Exception as e:
            raise DataProcessingError(f"Error parsing player data: {e}")
