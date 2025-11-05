import polars as pl
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import preliz as pz
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import nflreadpy as nfl
import os
from scipy.stats import norm


os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


seed = sum(map(ord, "receivingyardsproject"))
rng = np.random.default_rng(seed)

full_pass_data = pl.scan_parquet("processed_data/processed_passers_*.parquet").collect()

full_scores = nfl.load_schedules()

player_exp = nfl.load_players().select(
    pl.col("gsis_id", "display_name", "birth_date", "rookie_season")
)


clean_full_scores = full_scores.select(
    pl.col(
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
    )
)

rec_predictors = [
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
    "no_huddle",
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
]


rec_data_full = (
    full_pass_data.join(clean_full_scores, on=["game_id"])
    .filter(pl.col("game_type") == "REG")
    .with_columns(
        pl.col("pass_attempt")
        .sum()
        .over(["receiver_full_name", "game_id"])
        .alias("targeted"),
        pl.col("pass_attempt")
        .sum()
        .over(["posteam", "game_id"])
        .alias("total_pass_attempts"),
    )
    .select(
        pl.col(rec_predictors),
        pl.col(
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
        ),
    )
    .filter(
        (pl.col("yards_after_catch").is_not_null())
        & (pl.col("receiver_position").is_in(["RB", "TE", "WR"]))
    )
    .with_columns(
        pl.col("complete_pass")
        .str.to_integer()
        .count()
        .over("receiver_player_id", "season")
        .alias("receptions_season"),
        pl.col("complete_pass")
        .str.to_integer()
        .count()
        .over(["receiver_player_id", "game_id", "season"])
        .alias("receptions_per_game"),
        (pl.col("epa") * -1).alias("defensive_epa"),
    )
)


agg_full_seasons = (
    rec_data_full.with_columns(
        pl.col("yards_after_catch")
        .sum()
        .over(["receiver_full_name", "game_id", "season"])
        .alias("yac_per_game"),
        pl.col("receiving_yards")
        .sum()
        .over(["receiver_full_name", "game_id", "season"]),
        (pl.col("air_yards") / pl.col("total_pass_attempts"))
        .alias("avg_depth_of_target")
        .over(["posteam", "game_id", "season"]),
        (pl.col("receiving_yards") / pl.col("receptions_per_game")).alias(
            "yards_per_catch"
        ),
        pl.col("epa")
        .mean()
        .over(["game_id", "posteam", "season"])
        .alias("pass_epa_per_play"),
        pl.col("defensive_epa")
        .mean()
        .over(["game_id", "defteam", "season"])
        .alias("def_epa_per_play"),
        pl.col("pass_touchdown").str.to_integer(),
        pl.col("epa")
        .sum()
        .over(["game_id", "posteam", "season"])
        .alias("total_off_epa_game"),
        pl.col("defensive_epa")
        .sum()
        .over(["game_id", "defteam", "season"])
        .alias("total_def_epa_game"),
    )
    .with_columns(
        pl.col("pass_touchdown")
        .sum()
        .over(["receiver_full_name", "game_id", "season"])
        .alias("rec_tds_game")
    )
    .unique(subset=["game_id", "receiver_full_name", "season"])
    .select(
        # get rid of the per play to not have any confusion
        pl.exclude("epa", "defensive_epa")
    )
    .with_columns(
        pl.when(pl.col("rec_tds_game") > 0).then(1).otherwise(0).alias("rec_tds")
    )
    .with_columns(
        pl.col("rec_tds_game")
        .sum()
        .over(["receiver_full_name", "season"])
        .alias("rec_tds_season"),
    )
)


# we are goinng to effectively do games played

cumulative_stats = (
    agg_full_seasons.sort(["defteam", "season", "week"])
    .with_columns(
        pl.col("total_def_epa_game")
        .cum_sum()
        .over(["defteam", "season"])
        .shift(1)
        .alias("cumulative_def_epa"),
        pl.col("game_id")
        .cum_count()
        .over(["defteam", "season"])
        .alias("total_games_played_def"),
    )
    .with_columns(
        (pl.col("cumulative_def_epa") / pl.col("total_games_played_def")).alias(
            "def_epa_per_game"
        )
    )
    .sort(["posteam", "season", "week"])
    .with_columns(
        pl.col("total_off_epa_game")
        .cum_sum()
        .over(["posteam", "season"])
        .shift(1)
        .alias("cumulative_off_epa"),
        pl.col("game_id")
        .cum_count()
        .over(["posteam", "season"])
        .alias("total_games_played_offense"),
    )
    .with_columns(
        (pl.col("cumulative_off_epa") / pl.col("total_games_played_offense")).alias(
            "off_epa_per_game"
        )
    )
    .with_columns(
        (pl.col("cumulative_def_epa") - pl.col("cumulative_off_epa")).alias(
            "def_epa_diff"
        ),  # going into the game how much better is the defense playing than the offense
        pl.col("game_id")
        .cum_count()
        .over(["receiver_full_name", "season"])
        .alias("games_played"),
    )
    .join(player_exp, left_on=["receiver_player_id"], right_on="gsis_id", how="left")
    .with_columns(
        (pl.col("season") - pl.col("rookie_season")).alias("number_of_seasons_played"),
        pl.col("birth_date").str.to_date().dt.year().alias("birth_year"),
    )
    .with_columns(
        (pl.col("season") - pl.col("birth_year")).alias("age"),
        pl.when(pl.col("roof") == "indoors").then(1).otherwise(0).alias("is_indoors"),
    )
    .with_columns(
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
    )
    .sort(["posteam", "season", "week", "receiver_full_name"])
    .with_columns(
        (pl.col("player_rest") - pl.col("opponent_rest")).alias("player_rest_diff"),
        (pl.col("opponent_rest") - pl.col("player_rest")).alias("opponent_rest_diff"),
    )
    .sort(["receiver_full_name", "season", "game_id"])
    .fill_null(0)
)


factors_numeric = [
    "player_rest_diff",
    "def_epa_diff",
    "wind",
    "temp",
    "total_line",
    "avg_depth_of_target",
]

factors = factors_numeric + ["div_game", "home_game", "is_indoors"]

factors_numeric_train = cumulative_stats.select(pl.col(factors))

means = factors_numeric_train.select(
    [pl.col(c).mean().alias(c) for c in factors_numeric]
)
sds = factors_numeric_train.select([pl.col(c).std().alias(c) for c in factors_numeric])

factors_numeric_sdz = factors_numeric_train.with_columns(
    [((pl.col(c) - means[0, c]) / sds[0, c]).alias(c) for c in factors_numeric]
).with_columns(
    pl.Series("home_game", cumulative_stats["home_game"]),
    pl.Series("div_game", cumulative_stats["div_game"]),
    pl.Series("is_indoors", cumulative_stats["is_indoors"]),
)

cumulative_stats_pd = cumulative_stats.to_pandas()


unique_games = cumulative_stats_pd["games_played"].sort_values().unique()
unique_seasons = cumulative_stats_pd["number_of_seasons_played"].sort_values().unique()

off_play_caller = cumulative_stats_pd["off_play_caller"].sort_values().unique()
def_play_caller = cumulative_stats_pd["def_play_caller"].sort_values().unique()

unique_players = cumulative_stats_pd["receiver_full_name"].sort_values().unique()

cumulative_stats.group_by(["rec_tds"]).agg(pl.len())


player_idx = pd.Categorical(
    cumulative_stats_pd["receiver_full_name"], categories=unique_players
).codes

games_idx = pd.Categorical(
    cumulative_stats_pd["games_played"], categories=unique_games
).codes

off_play_caller_idx = pd.Categorical(
    cumulative_stats_pd["off_play_caller"], categories=off_play_caller
).codes

def_play_caller_idx = pd.Categorical(
    cumulative_stats_pd["def_play_caller"], categories=def_play_caller
).codes

coords = {
    "factors": factors,
    "gameday": unique_games,
    "seasons": unique_seasons,
    "obs_id": cumulative_stats_pd.index,
    "player": unique_players,
    "off_play_caller": off_play_caller,
    "def_play_caller": def_play_caller,
    "time_scale": ["games", "season"],
}

empirical_probs = cumulative_stats_pd["rec_tds"].value_counts(normalize=True).to_numpy()

cumulative_probs = empirical_probs.cumsum()[:-1]

cutpoints_standard = norm.ppf(cumulative_probs)

delta_prior = np.diff(cutpoints_standard)

seasons_gp_prior, ax = pz.maxent(pz.InverseGamma(), lower=1, upper=8)

plt.xlim(0, 10)

seasons_m, seasons_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[
        0,
        cumulative_stats.select(pl.col("number_of_seasons_played").max()).to_series()[
            0
        ],
    ],
    lengthscale_range=[1.0, 8],
    cov_func="matern52",
)

plt.close("all")

# the short term prior is actually pretty decent
short_term_form, _ = pz.maxent(pz.InverseGamma(), lower=2, upper=5)


within_m, within_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[
        0,
        cumulative_stats.select(pl.col("games_played").max()).to_series()[0],
    ],
    lengthscale_range=[2, 5],
    cov_func="matern52",
)

plt.close("all")

# effectively the difference between a player is about 2 touchdowns
touchdown_dist, ax = pz.maxent(pz.Exponential(), 0, 1)


observed_ratio = (
    cumulative_stats["rec_tds_game"].value_counts(normalize=True).sort("rec_tds_game")
)

with pm.Model(coords=coords) as rec_tds_era_adjusted:
    factor_data = pm.Data("factor_data", factors_numeric_sdz, dims=("obs_id", "factor"))
    games_id = pm.Data("games_id", games_idx, dims="obs_id")
    player_id = pm.Data("player_id", player_idx, dims="obs_id")
    season_id = pm.Data(
        "season_id",
        cumulative_stats["number_of_seasons_played"].to_numpy(),
        dims="obs_id",
    )

    rec_tds_obs = pm.Data(
        "rec_tds_obs", cumulative_stats["rec_tds"].to_numpy(), dims="obs_id"
    )

    x_gamedays = pm.Data("x_gamedays", unique_games, dims="gameday")[:, None]
    x_seasons = pm.Data("x_seasons", unique_seasons, dims="seasons")[:, None]

    # ref notebook sets it at the max of goals scored of the games so we are going to do the same
    intercept_sigma = 4
    sd = touchdown_dist.to_pymc("touchdown_sd")

    baseline_sigma = pt.sqrt(intercept_sigma**2 + sd**2 / len(coords["player"]))

    baseline = baseline_sigma * pm.Normal("baseline")

    player_effect = pm.Deterministic(
        "player_effect",
        baseline + pm.ZeroSumNormal("player_effect_raw", sigma=sd, dims="player"),
        dims="player",
    )

    # bumbing this up a bit
    alpha_scale, upper_scale = 0.021, 2.0
    gps_sigma = pm.Exponential(
        "gps_sigma", lam=-np.log(alpha_scale) / upper_scale, dims="time_scale"
    )

    ls = pm.InverseGamma(
        "ls",
        alpha=np.array([short_term_form.alpha, seasons_gp_prior.alpha]),
        beta=np.array([short_term_form.beta, seasons_gp_prior.beta]),
        dims="time_scale",
    )

    cov_games = gps_sigma[0] ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ls[0])
    cov_seasons = gps_sigma[1] ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ls[1])

    gp_games = pm.gp.HSGP(m=[within_m], c=within_c, cov_func=cov_games)
    gp_season = pm.gp.HSGP(
        m=[seasons_m], c=seasons_c, cov_func=cov_seasons, parametrization="centered"
    )

    basis_vectors_game, sqrt_psd_game = gp_games.prior_linearized(X=x_gamedays)

    basis_coeffs_games = pm.Normal("basis_coeffs_games", shape=gp_games.n_basis_vectors)

    f_games = pm.Deterministic(
        "f_games",
        basis_vectors_game @ (basis_coeffs_games * sqrt_psd_game),
        dims="gameday",
    )

    basis_vectors_season, sqrt_psd_season = gp_season.prior_linearized(X=x_seasons)

    basis_coeffs_season = pm.Normal(
        "basis_coeffs_season", shape=gp_season.n_basis_vectors
    )

    f_season = pm.Deterministic(
        "f_season",
        basis_vectors_season @ (basis_coeffs_season * sqrt_psd_season),
        dims="seasons",
    )

    alpha = pm.Deterministic(
        "alpha",
        player_effect[player_id] + f_season[season_id] + f_games[games_id],
        dims="obs_id",
    )
    slope = pm.Normal("slope", sigma=0.25, dims="factors")

    eta = pm.Deterministic(
        "eta", alpha + pm.math.dot(factor_data, slope), dims="obs_id"
    )
    cutpoints_off = 4

    delta_mean = pm.Normal(
        "delta_mean", mu=delta_prior * cutpoints_off, sigma=1, shape=2
    )

    delta_sig = pm.Exponential("delta_sig", 1, shape=2)

    player_delta = delta_mean + delta_sig * pm.Normal(
        "player_delta", shape=(len(coords["player"]), 2)
    )

    cutpoints = pm.Deterministic(
        "cutpoints",
        pt.concatenate(
            [
                pt.full((player_effect.shape[0], 1), cutpoints_off),
                pt.cumsum(pt.softplus(player_delta), axis=-1) + cutpoints_off,
            ],
            axis=-1,
        ),
    )

    pm.OrderedLogistic(
        "tds_scored",
        cutpoints=cutpoints[player_id],
        eta=eta,
        observed=rec_tds_obs,
        dims="obs_id",
    )


with rec_tds_era_adjusted:
    idata = pm.sample_prior_predictive()


with rec_tds_era_adjusted:
    idata.extend(pm.sample(nuts_sampler="numpyro", random_seed=rng, target_accept=0.99))

idata.sample_stats.diverging.sum().data


az.rhat(
    idata, var_names=["basis_coeffs_season", "basis_coeffs_games"]
).max().to_pandas().round(2)


az.ess(idata).min().to_pandas().sort_values().round()

az.plot_energy(idata)
