import time
import json
import logging
from datetime import date
from typing import List, Dict, Any, Optional

import polars as pl
import requests

from backend.config.settings import Config
from backend.utils.exceptions import DataProcessingError
from backend.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class ESPNCollector(BaseCollector):
    """Collector for ESPN fantasy data."""

    def collect_data(self) -> pl.DataFrame:
        all_players: List[Dict[str, Any]] = []

        for idx, (position, slot_id) in enumerate(self.config.ESPN_SLOT_NUMS.items()):
            if idx > 0:
                time.sleep(self.config.RATE_LIMIT_DELAY)

            limit = self.config.ESPN_POSITION_LIMITS.get(position, None)
            if limit is None:
                logger.warning(f"No position limit defined for {position}, skipping.")
                continue

            url = (
                f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{self.season}"
                "/segments/0/leaguedefaults/3?scoringPeriodId=0&view=kona_player_info"
            )
            fantasy_filter = self._build_espn_filter(slot_id, limit)
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Host": "lm-api-reads.fantasy.espn.com",
                "X-Fantasy-Source": "kona",
                "X-Fantasy-Filter": json.dumps(fantasy_filter),
                "User-Agent": "Fantasy Data Collector v1.0"
            }

            response = self._make_request(url, headers)
            try:
                espn_json = response.json().get("players", [])
            except Exception as e:
                logger.warning(f"Failed to parse data for {position}: {e}")
                continue

            for player_data in espn_json:
                try:
                    player_dict = self._parse_player_data(player_data)
        # Inject the position string into the dictionary
                    player_dict['position'] = position
                    all_players.append(player_dict)

                except Exception as e:
                    logger.warning(f"Skipping {position} player due to parse error: {e}")

        if not all_players:
            raise DataProcessingError("No ESPN player data collected.")

        return pl.from_dicts(all_players).with_columns([
            pl.lit(date.today()).cast(pl.Date).alias('date_queried'),
            pl.lit('espn').alias('source'),
           # pl.lit(limit).alias('position')
        ])

    def _build_espn_filter(self, slot_id: int, limit: int) -> Dict[str, Any]:
        return {
            "players": {
                "filterSlotIds": {"value": [slot_id]},
                "filterStatsForSourceIds": {"value": [1]},
                "filterStatsForSplitTypeIds": {"value": [0]},
                "sortAppliedStatTotal": {"sortAsc": False, "sortPriority": 3, "value": f"11{self.season}0"},
                "sortDraftRanks": {"sortPriority": 2, "sortAsc": True, "value": "PPR"},
                "sortPercOwned": {"sortAsc": False, "sortPriority": 4},
                "limit": limit,
                "offset": 0
            }
        }

    def _parse_player_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dictionary of ESPN player info."""
        try:
            player_info = raw_data["player"]
            ownership = player_info.get("ownership", {})
            draft_ranks = player_info.get("draftRanksByRankType", {})
            ppr_rank_info = draft_ranks.get("PPR", {})

            return {
                "player": player_info.get("fullName"),
                "espn_rank": ppr_rank_info.get("rank"),
                "espn_adp": ownership.get("averageDraftPosition"),
                "espn_auction_value": ownership.get("auctionValueAverage"),
                "position": player_info.get("defaultPosition"),
                "team_abr": player_info.get("editorialTeamAbbr")
            }
        except (KeyError, TypeError) as e:
            raise DataProcessingError(f"Failed to parse ESPN player data: {e}")

