import requests
import polars as pl
from datetime import datetime, date
from typing import Optional
import os


class FantasyDataCollector:
    def __init__(self, data_dir="data", season: Optional[int] = None):
        self.data_dir = data_dir
        self.season = season or datetime.now().year
        os.makedirs(self.data_dir, exist_ok=True)

    def get_yahoo_data(self):
        url = (
            "https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2/league/"
            "461.l.public;out=settings/players;position=ALL;start=0;count=400;"
            "sort=average_pick;search=;out=auction_values,ranks;"
            "ranks=season;ranks_by_position=season;out=expert_ranks;"
            "expert_ranks.rank_type=projected_season_remaining/draft_analysis;"
            "cut_types=diamond;slices=last7days?format=json_f"
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

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        players_data = data["fantasy_content"]["league"]["players"]
        player_list = []

        for player_wrapper in players_data:
            player = player_wrapper.get("player")
            if not player:
                continue
            try:
                player_info = {
                    "full_name": player["name"]["full"],
                    "first_name": player["name"]["first"],
                    "last_name": player["name"]["last"],
                    "team_abr": player.get("editorial_team_abbr"),
                    "bye_week": player.get("bye_weeks", {}).get("week"),
                    "position": player["display_position"],
                    "projected_auction_value": player.get("projected_auction_value"),
                    "average_pick_yahoo": player["draft_analysis"].get("average_pick"),
                    "average_cost_yahoo": player["draft_analysis"].get("average_cost"),
                    "yahoo_rank_2025": player["player_ranks"][0]["player_rank"]["rank_value"],
                }
                player_list.append(player_info)
            except Exception as e:
                print(f"Skipping a player due to missing data: {e}")

        yahoo_fantasy_2025 = pl.from_dicts(player_list)

        fp24 = pl.read_csv(os.path.join(self.data_dir, "fpros-scoring-2024.csv")).select(
            pl.col("Player"),
            pl.col("AVG").alias("avg_2024"),
            pl.col("TTL").alias("total_2024"),
        )
        fp23 = pl.read_csv(os.path.join(self.data_dir, "fpros-scoring-2023.csv")).select(
            pl.col("Player"),
            pl.col("AVG").alias("avg_2023"),
            pl.col("TTL").alias("total_2023"),
        )

        yahoo_fantasy_2025 = (
            yahoo_fantasy_2025.join(fp24, left_on=["full_name"], right_on=["Player"], how="left")
            .join(fp23, left_on=["full_name"], right_on=["Player"], how="left")
        ).with_columns(
            pl.lit(date.today()).cast(pl.Date).alias('date_queried')
        )

        out_file = os.path.join(self.data_dir, f"yahoo_fantasy-{date.today()}.parquet")
        yahoo_fantasy_2025.write_parquet(out_file)
        print(f"Yahoo data saved to {out_file}")

        return yahoo_fantasy_2025

    def espn_draft(self, metric="adp"):
        metric = metric.lower()
        if metric not in ["adp", "aav"]:
            raise ValueError("metric must be 'adp' or 'aav'")

        slot_nums = {
            "QB": 0,
            "RB": 2,
            "WR": 4,
            "TE": 6,
            "K": 17,
            "DST": 16
        }
        
        position_limits = {
            "QB": 42,
            "RB": 100,
            "WR": 150,
            "TE": 60,
            'K': 35,
            'DST': 32
        }
        
        season = self.season
        base_url = (
            f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
            "/segments/0/leaguedefaults/3?scoringPeriodId=0&view=kona_player_info"
        )

        all_positions_data = {}

        for idx, pos in enumerate(slot_nums.keys()):
            if idx > 0:
                time.sleep(2)  # Rate limiting

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
            all_positions_data[pos] = espn_json

        # Process the collected data
        positions_data = {}
        for position, players in all_positions_data.items():
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

        espn_data = pl.concat(list(positions_data.values()), how='diagonal')
        espn_data = espn_data.with_columns(
            pl.lit(date.today()).cast(pl.Date).alias('date_queried')
        )

        out_file = os.path.join(self.data_dir, f"espn_fantasy-{date.today()}.parquet")
        espn_data.write_parquet(out_file)
        print(f"ESPN data saved to {out_file}")

        return espn_data

    def collect_all_data(self):
        """Collect data from both Yahoo and ESPN sources."""
        print("Starting data collection...")
        
        yahoo_data = self.get_yahoo_data()
        print(f"Collected {len(yahoo_data)} players from Yahoo")
        
        espn_data = self.espn_draft()
        print(f"Collected {len(espn_data)} players from ESPN")
        
        # Merge the datasets
        merged_data = yahoo_data.join(espn_data, on='full_name', how='outer')
        
        out_file = os.path.join(self.data_dir, f"fantasy_merged-{date.today()}.parquet")
        merged_data.write_parquet(out_file)
        print(f"Merged data saved to {out_file}")
        
        return {
            'yahoo': yahoo_data,
            'espn': espn_data,
            'merged': merged_data
        }