import requests
import json
import time
import polars as pl
import polars.selectors as cs
from datetime import datetime

season = 2025



base_url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/players"


    


all_positions_data = {}

def espn_draft(metric="adp"):
    metric = metric.lower()
    if metric not in ["adp", "aav"]:
        raise ValueError("metric must be 'adp' or 'aav'")

    season = datetime.now().year

    # Position slot numbers
    slot_nums = {
        "QB": 0,
        "RB": 2,
        "WR": 4,
        "TE": 6,
        "K": 17,
        "DST": 16
    }

    # Position-specific limits
    # for
    position_limits = {
        "QB": 42,
        "RB": 100,
        "WR": 150,
        "TE": 60,
        'K': 35,
        'DST': 32
    }

    base_url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
        "/segments/0/leaguedefaults/3?scoringPeriodId=0&view=kona_player_info"
    )

    all_positions_data = {}

    for idx, pos in enumerate(slot_nums.keys()):
        if idx > 0:
            time.sleep(2)  # mimic R Sys.sleep(2)

        pos_idx = slot_nums[pos]
        limit = position_limits[pos]

        fantasy_filter = {
            "players": {
                "filterSlotIds": {"value": [pos_idx]},
                "filterStatsForSourceIds": {"value": [1]},
                "filterStatsForSplitTypeIds": {"value": [0]},
                "sortAppliedStatTotal": {
                    "sortAsc": False,
                    "sortPriority": 3,
                    "value": f"11{season}0"
                },
                "sortDraftRanks": {
                    "sortPriority": 2,
                    "sortAsc": True,
                    "value": "PPR"
                },
                "sortPercOwned": {
                    "sortAsc": False,
                    "sortPriority": 4
                },
                "limit": limit,
                "offset": 0,
                "filterRanksForScoringPeriodIds": {"value": [2]},
                "filterRanksForRankTypes": {"value": ["PPR"]},
                "filterRanksForSlotIds": {"value": [0, 2, 4, 6, 17, 16]},
                "filterStatsForTopScoringPeriodIds": {
                    "value": 2,
                    "additionalValue": [
                        f"00{season}",
                        f"10{season}",
                        f"11{season}0",
                        f"02{season}"
                    ]
                }
            }
        }

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Host": "lm-api-reads.fantasy.espn.com",
            "X-Fantasy-Source": "kona",
            "X-Fantasy-Filter": json.dumps(fantasy_filter),
            "User-Agent": "ffanalytics Python script (https://github.com/FantasyFootballAnalytics/ffanalytics)"
        }

        print(f"Fetching {pos} (limit={limit})...")
        resp = requests.get(base_url, headers=headers)
        resp.raise_for_status()
        espn_json = resp.json().get("players", [])

        l_players = [None] * len(espn_json)

        all_positions_data[pos] = espn_json

    return all_positions_data


get_data  = espn_draft()

test_data = get_data['QB'][0]

positions_data = {}

for position, players in get_data.items():
    extracted = []
    for p in players:
        player_info = p["player"]
        ownership = player_info.get("ownership", {})
        draft_ranks = player_info.get("draftRanksByRankType", {})
        ppr_rank_info = draft_ranks.get("PPR", {})

        
        extracted.append({
            "full_name": player_info.get("fullName"),
            "average_cost_espn": ownership.get("auctionValueAverage"),
            "average_cost_percent_change_espn": ownership.get("auctionValueAverageChange"),
            "average_pick_espn": ownership.get("averageDraftPosition"),
            "average_pick_percent_change_espn": ownership.get("averageDraftPositionPercentChange"),
            'espn_rank_2025': ppr_rank_info.get('rank')
        })
    extracted_df = pl.from_dicts(extracted)
    positions_data[position] = extracted_df



espn_data = pl.concat(list(positions_data.values()), how = 'diagonal')

espn_data.write_parquet('data/espn_fantasy.parquet')
