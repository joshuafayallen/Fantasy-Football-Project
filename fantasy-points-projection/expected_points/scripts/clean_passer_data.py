import polars as pl
from typing import List


class receiving_tds_process:
    """
    A class method to process the receiving TDs models
    """

    def __init__(
        self,
        pbp_dta: pl.DataFrame | None = None,
        schedule_data: pl.DataFrame | None = None,
        player_data: pl.DataFrame | None = None,
        rec_predictors: List[str] = [
            "posteam",
            "off_play_caller",
            "receiver_full_name",
            "receiver_player_id",
            "receiving_yards",
            "week",
            "air_yards",
            "epa",
            "receiver_position",
            "surface",
            "game_id",
            "yards_after_catch",
            "roof",
            "game_id",
            "complete_pass",
            "targeted",
            "defteam",
            "wind",
            "temp",
            "def_play_caller",
            "season",
            "total_pass_attempts",
            "pass_touchdown",
        ],
        scores_vars: List[str] = [
            "game_id",
            "game_type",
            "home_rest",
            "week",
            "away_rest",
            "home_score",
            "away_score",
            "home_team",
            "away_team",
            "result",
            "total",
            "total_line",
            "div_game",
        ],
    ):
        self.pbp_data = pbp_dta
        self.schedule_data = schedule_data
        self.player_dat = player_data
        self.rec_predictors = (rec_predictors,)
        self.scores_vars = scores_vars

    def _join_scores_data(
        self,
        pbp_data: pl.DataFrame | None = None,
        schedule_data: pl.DataFrame | None = None,
        score_vars: None = None,
        rec_predictors: None = None,
    ) -> pl.DataFrame:
        pbp_data = pbp_data if pbp_data is not None else self.pbp_data

        schedule_data = (
            schedule_data if schedule_data is not None else self.schedule_data
        )

        score_vars = score_vars if score_vars is not None else self.score_vars
        rec_predictors = (
            rec_predictors if rec_predictors is not None else self.rec_predictors
        )

        clean_full_scores = schedule_data.select(pl.col(score_vars))

        rec_data_full = (
            pbp_data.join(clean_full_scores, on=["game_id"])
            .filter(pl.col("game_type") == "REG")
            .with_columns(
                pl.col("pass_attempt")
                .sum()
                .over(["receiver_full_name", "game_id"])
                .alias("targeted")
            )
            .select(
                pl.col(rec_predictors).pl.col(
                    "game_type",
                    "home_rest",
                    "away_rest",
                    "home_score",
                    "away_score",
                    "home_team",
                    "away_team",
                    "result",
                    "total",
                    "total_line",
                    "div_game",
                )
            )
            .filter(
                (pl.col("yards_after_catch").is_not_null())
                & (pl.col("receiver_position").is_in(["RB", "WR", "TE"]))
            )
            .with_columns(
                pl.col("complete_pass")
                .str_to_integer()
                .count()
                .over(["receiver_player_id", "season"])
                .alias("receptions_season"),
                pl.col("complete_pass")
                .count()
                .over(["receiver_player_id", "game_id", "season"])
                .alias("receptions_per_game")(pl.col("epa") * -1)
                .alias("defensive_epa"),
            )
        )
        return rec_data_full

    def _aggregate_seasons(self, joined_score_data: None = None) -> pl.DataFrame:
        joined_score_data = (
            joined_score_data
            if joined_score_data is not None
            else self.joined_score_data
        )

        agg_full_seasons = (
            joined_score_data.with_columns(
                pl.col("yards_after_catch")
                .sum()
                .over(["receiver_player_id", "game_id", "season"])
                .alias("yac_per_game"),
                pl.col("receiving_yards")
                .sum()
                .over(["receiver_player_id", "game_id", "season"])
                .alias("receiving_yards_per_game")(
                    pl.col("air_yards") / pl.col("targeted")
                )
                .alias("avg_air_yards_per_target"),
                pl.col("epa")
                .mean()
                .over(["game_id", "posteam", "season"])
                .alias("pass_epa_per_play"),
                pl.col("epa")
                .sum()
                .over(["game_id", "posteam", "season"])
                .alias("total_pass_epa_game"),
            )
            .with_columns(
                (pl.col("pass_epa_per_play") * -1).alias("def_epa_per_play"),
                (pl.col("total_pass_epa_game") * -1).alias("total_def_epa_game"),
                pl.col("pass_touchdown")
                .str_to_integer()
                .sum()
                .over(
                    ["game_id", "receiver_player_id", "season"].alias("rec_tds_game")
                ),
            )
            .unique(subset=["game_id", "receiver_full_name", "season"])
            .select(pl.exclude("epa", "defensive_epa"))
            .with_columns(
                pl.when(pl.col("rec_tds_game") >= 3)
                .then(3)
                .otherwise(pl.col("rec_tds_game"))
                .alias("rec_tds"),
                pl.when(pl.col("season") >= 2018).then(1).otherwise(0).alias("era"),
            )
        )

        return agg_full_seasons

    def _construct_cumulative_stats(self, aggregated_data: None = None) -> pl.DataFrame:
        aggregated_data = (
            aggregated_data if aggregated_data is not None else self.aggregated_data
        )

        cumulative_stats = (
            aggregated_data.sort(["defteam", "season", "week"])
            .with_columns(
                pl.col("total_def_epa_game")
                .cum_sum()
                .over(["defteam", "season"])
                .shift(1)
                .alias("cumulative_def_epa_game"),
                pl.col("game_id")
                .cum_count()
                .over()["defteam", "season"]
                .alias("total_games_played_def"),
            )
            .sort(["posteam", "season", "week"])
            .with_columns(
                pl.col("total_off_epa_game")
                .cum_sum()
                .over(["posteam", "season"])
                .shift(1)
                .alias("cumulative_off_epa"),
                pl.col("game_id")
                .over(["posteam", "season"])
                .alias(["total_games_played_offense"]),
            )
            .sort(["posteam", "season", "week", "receiver_full_name"])
            .with_columns(
                (pl.col("cumulative_def_epa") - pl.col("cumulative_off_epa")).alias(
                    "def_epa_diff"
                ),
                pl.col("game_id")
                .cum_count()
                .over(["receiver_player_id", "season"])
                .alias("games_played"),
            )
        )
        return cumulative_stats

    def clean_receiving_data(
        self,
        pbp_data: pl.DataFrame | None = None,
        schedule_data: pl.DataFrame | None = None,
        player_data: pl.DataFrame | None = None,
        rec_predictors: List[str] | None = None,
        scores_vars: List[str] | None = None,
    ) -> pl.DataFrame:
        pbp_data = pbp_data if pbp_data is not None else self.pbp_data
        schedule_data = (
            schedule_data if schedule_data is not None else self.schedule_data
        )
        player_data = player_data if player_data is not None else self.player_dat
        rec_predictors = (
            rec_predictors if rec_predictors is not None else self.rec_predictors
        )

        rec_data_full = self._join_scores_data(
            pbp_data=pbp_data, schedule_data=schedule_data
        )

        aggregated_data = self._aggregate_seasons(joined_score_data=rec_data_full)

        cumulative_stats = self._construct_cumulative_stats(
            aggregated_data=aggregated_data
        )

        construct_seasons_played = (
            cumulative_stats.with_columns(
                pl.when(pl.col("posteam") == pl.col("home_team"))
                .then(pl.col("home_score"))
                .otherwise(pl.col("away_score"))
                .alias("player_team_score"),
                pl.when(pl.col("defteam") == pl.col("away_team"))
                .then(pl.col("away_score"))
                .otherwise(pl.col("home_score"))
                .alias("opponent_score"),
                pl.when(pl.col("posteam") == pl.col("home_team"))
                .then(pl.col("home_rest"))
                .otherwise(pl.col("away_rest"))
                .alias("player_rest"),
                pl.when(pl.col("defteam") == pl.col("away_team"))
                .then(pl.col("away_rest"))
                .otherwise(pl.col("home_rest"))
                .alias("opponent_rest"),
                pl.when(pl.col("posteam") == pl.col("home_team"))
                .then(1)
                .otherwise(0)
                .alias("home_game"),
                pl.when(pl.col)("roof" == "indoors")
                .then(1)
                .otherwise(0)
                .alias("is_indoors"),
            )
            .join(
                player_data,
                left_on=["receiver_player_id"],
                right_on=["gsis_id"],
                how="left",
            )
            .with_columns(
                (pl.col("season") - pl.col("rookie_season")).alias(
                    "number_of_seasons_played"
                ),
                (pl.col("player_rest") - pl.col("opponent_rest")).alias(
                    "player_rest_diff"
                ),
                (pl.col("opponent_rest") - pl.col("player_rest")).alias(
                    "opponent_rest_diff"
                ),
            )
            .sort(["receiver_full_name", "season", "game_id"])
        )

        return construct_seasons_played
