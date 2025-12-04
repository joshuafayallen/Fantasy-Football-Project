import polars as pl
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import preliz as pz
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import nflreadpy as nfl
import xarray as xr
import seaborn as sns
import os
from scipy.stats import norm

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=18"


seed = sum(map(ord, "rushinyardsproject"))
rng = np.random.default_rng(seed)

# hmm there is no rush position which is kind of annoying
# once we add
full_rush_data = pl.scan_parquet("processed_data/processed_rushers_*.parquet").collect()

full_scores = pl.read_csv(
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

player_exp = nfl.load_players().select(
    pl.col(
        "gsis_id",
        "display_name",
        "birth_date",
        "rookie_season",
        "position",
        "position_group",
        "weight",
    )
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
        "total_line",
        "away_team",
        "result",
        "total",
        "total_line",
        "div_game",
    )
)


rush_predictors = [
    "posteam",
    "off_play_caller",
    "rush_attempts",
    "full_name",
    "rusher_player_id",
    "week",
    "epa",
    "surface",
    # "no_huddle",
    "game_id",
    "roof",
    "game_id",
    "defteam",
    "wind",
    "temp",
    "def_play_caller",
    "season",
    "rushing_yards",
    "rush_touchdown",
]


rush_data_full = (
    full_rush_data.join(clean_full_scores, on=["game_id"])
    .filter(pl.col("game_type") == "REG")
    .with_columns(
        pl.col("rush_attempt")
        .sum()
        .over(["full_name", "game_id"])
        .alias("rush_attempts"),
        pl.col("rush_attempt")
        .cum_sum()
        .over(["posteam", "game_id"])
        .alias("total_rush_attempts"),
    )
    .select(
        pl.col(rush_predictors),
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
    .with_columns(
        (pl.col("epa") * -1).alias("defensive_epa"),
    )
)


agg_full_seasons = (
    rush_data_full.with_columns(
        pl.col("rushing_yards").sum().over(["full_name", "game_id", "season"]),
        pl.col("epa")
        .mean()
        .over(["game_id", "posteam", "season"])
        .alias("pass_epa_per_play"),
        pl.col("defensive_epa")
        .mean()
        .over(["game_id", "defteam", "season"])
        .alias("def_epa_per_play"),
        pl.col("rush_touchdown").str.to_integer(),
        pl.col("epa")
        .sum()
        .over(["game_id", "posteam", "season"])
        .alias("total_rush_epa_game"),
        pl.col("defensive_epa")
        .sum()
        .over(["game_id", "defteam", "season"])
        .alias("total_def_epa_game"),
    )
    .with_columns(
        pl.col("rush_touchdown")
        .sum()
        .over(["full_name", "game_id", "season"])
        .alias("rush_tds_game"),
        (pl.col("rushing_yards") / pl.col("rush_attempts")).alias("yards_per_attempt"),
    )
    .unique(subset=["game_id", "full_name", "season"])
    .select(
        # get rid of the per play to not have any confusion
        pl.exclude("epa", "defensive_epa")
    )
    .with_columns(
        pl.when(pl.col("rush_tds_game") >= 3)
        .then(3)
        .otherwise(pl.col("rush_tds_game"))
        .alias("rush_tds")
    )
    .with_columns(
        pl.col("rush_tds_game")
        .sum()
        .over(["full_name", "season"])
        .alias("rush_tds_season"),
        pl.when(pl.col("season") >= 2018).then(1).otherwise(0).alias("era"),
    )
)


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
        pl.col("total_rush_epa_game")
        .cum_sum()
        .over(["posteam", "season"])
        .shift(1)
        .alias("cumulative_off_epa"),
        pl.col("game_id")
        .cum_count()
        .over(["posteam", "season"])
        .alias("total_games_played_offense"),
        pl.col("rushing_yards")
        .cum_sum()
        .over(["posteam", "season"])
        .shift(1)
        .alias("rush_yards_game"),
        pl.col("rush_attempts")
        .cum_sum()
        .over(["posteam", "season"])
        .shift(1)
        .alias("cumulative_rush_attempts"),
    )
    .with_columns(
        (pl.col("rush_yards_game") / pl.col("cumulative_rush_attempts")).alias(
            "rush_yards_attempt_cumulative"
        )
    )
    .with_columns(
        (pl.col("cumulative_def_epa") - pl.col("cumulative_off_epa")).alias(
            "def_epa_diff"
        ),  # going into the game how much better is the defense playing than the offense
        pl.col("game_id")
        .cum_count()
        .over(["full_name", "season"])
        .alias("games_played"),
    )
    .join(player_exp, left_on=["rusher_player_id"], right_on="gsis_id", how="left")
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
    .sort(["posteam", "season", "week", "full_name"])
    .with_columns(
        (pl.col("player_rest") - pl.col("opponent_rest")).alias("player_rest_diff"),
        (pl.col("opponent_rest") - pl.col("player_rest")).alias("opponent_rest_diff"),
    )
    .sort(["full_name", "season", "game_id"])
    .filter(pl.col("position_group").is_in(["RB", "QB"]))
    .fill_null(0)
)


season = range(2002, 2025)

## unfortunately it is kind of difficult to find the
## starting rotation so we can't include
## any info on the differnce in mass between
## the lines this would probably be more useful
## if we also had what type of run it was
rosters = nfl.load_rosters(seasons=season)


factors_numeric = [
    "player_rest_diff",
    "def_epa_diff",
    "wind",
    "temp",
    "total_line",
    "rush_yards_attempt_cumulative",
]

factors = factors_numeric + ["div_game", "home_game", "is_indoors", "era"]

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
    pl.Series("era", cumulative_stats["era"]),
)

cumulative_stats_pd = cumulative_stats.to_pandas()


unique_games = cumulative_stats_pd["games_played"].sort_values().unique()
unique_seasons = cumulative_stats_pd["number_of_seasons_played"].sort_values().unique()

off_play_caller = cumulative_stats_pd["off_play_caller"].sort_values().unique()
def_play_caller = cumulative_stats_pd["def_play_caller"].sort_values().unique()

unique_players = cumulative_stats_pd["full_name"].sort_values().unique()

cumulative_stats.group_by(["rush_tds"]).agg(pl.len())


player_idx = pd.Categorical(
    cumulative_stats_pd["full_name"], categories=unique_players
).codes

seasons_idx = pd.Categorical(
    cumulative_stats_pd["number_of_seasons_played"], categories=unique_seasons
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

empirical_probs = (
    cumulative_stats_pd["rush_tds"].value_counts(normalize=True).to_numpy()
)

cumulative_probs = empirical_probs.cumsum()[:-1]

cutpoints_standard = norm.ppf(cumulative_probs)

delta_prior = np.diff(cutpoints_standard)

seasons_gp_prior, ax = pz.maxent(pz.InverseGamma(), lower=2, upper=6)

plt.xlim(0, 18)
plt.close("all")
seasons_m, seasons_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[
        0,
        cumulative_stats.select(pl.col("number_of_seasons_played").max()).to_series()[
            0
        ],
    ],
    lengthscale_range=[2, 6],
    cov_func="matern52",
)


short_term_form, _ = pz.maxent(pz.InverseGamma(), lower=2, upper=5)


within_m, within_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[
        0,
        cumulative_stats.select(pl.col("games_played").max()).to_series()[0],
    ],
    lengthscale_range=[2, 5],
    cov_func="matern52",
)


touchdown_dist, ax = pz.maxent(pz.Exponential(), 0, 1)

with pm.Model(coords=coords) as rush_tds:
    factor_data = pm.Data("factor_data", factors_numeric_sdz, dims=("obs_id", "factor"))
    games_id = pm.Data("games_id", games_idx, dims="obs_id")
    player_id = pm.Data("player_id", player_idx, dims="obs_id")
    season_id = pm.Data(
        "season_id",
        seasons_idx,
        dims="obs_id",
    )

    rec_tds_obs = pm.Data(
        "rush_tds_obs", cumulative_stats["rush_tds"].to_numpy(), dims="obs_id"
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
    alpha_scale, upper_scale = 0.03, 2.0
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
    gp_season = pm.gp.HSGP(m=[seasons_m], c=seasons_c, cov_func=cov_seasons)

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


with rush_tds:
    idata = pm.sample_prior_predictive(compile_kwargs={"mode": "NUMBA"})


implied_cats = az.extract(idata.prior_predictive, var_names=["tds_scored"])

fig, axes = plt.subplots(ncols=2)

axes[0] = (
    implied_cats.isel(obs_id=0)
    .to_pandas()
    .reset_index(drop=True)
    .value_counts(normalize=True)
    .sort_index()
    .plot(kind="bar", rot=0, alpha=0.8, ax=axes[0])
)
axes[0].set(
    xlabel="Touchdowns",
    ylabel="Proportion",
    title="Prior allocation of TDs for observation 0",
)

axes[1] = (
    cumulative_stats_pd["rush_tds"]
    .value_counts(normalize=True)
    .sort_index()
    .plot(kind="bar", rot=0, alpha=0.8, ax=axes[1])
)

axes[1].set(
    xlabel="Touchdowns", ylabel="Proportion", title="Observed TDs for Observation 0"
)
## How big the sigma of the slope is
# 2.0 you are really underestimating 0
# around 1.0 - 1.5 the prior predictive mean dead on
# lets use 2.0 to give us a little bit of room when we start to introduce data
az.plot_ppc(idata, group="prior", observed=True)

with rush_tds:
    idata.extend(pm.sample(nuts_sampler="numpyro", random_seed=rng, target_accept=0.99))

az.rhat(
    idata, var_names=["basis_coeffs_season", "basis_coeffs_games"]
).max().to_pandas().round(2)


idata.sample_stats.diverging.sum().data

az.ess(idata).min().to_pandas().sort_values().round(2)


with rush_tds:
    idata.extend(
        pm.sample_posterior_predictive(idata, compile_kwargs={"mode": "NUMBA"})
    )
    idata.extend(pm.compute_log_likelihood(idata))


idata.to_netcdf("models/idata-rush-tds")


with pm.Model(coords=coords) as rush_tds2:
    factor_data = pm.Data("factor_data", factors_numeric_sdz, dims=("obs_id", "factor"))
    games_id = pm.Data("games_id", games_idx, dims="obs_id")
    player_id = pm.Data("player_id", player_idx, dims="obs_id")
    season_id = pm.Data(
        "season_id",
        seasons_idx,
        dims="obs_id",
    )

    rec_tds_obs = pm.Data(
        "rush_tds_obs", cumulative_stats["rush_tds"].to_numpy(), dims="obs_id"
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
    alpha_scale, upper_scale = 0.03, 2.0
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
    gp_season = pm.gp.HSGP(m=[seasons_m], c=seasons_c, cov_func=cov_seasons)

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

    delta_sig = pm.Exponential("delta_sig", 2, shape=2)

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


with rush_tds2:
    idata2 = pm.sample_prior_predictive(compile_kwargs={"mode": "NUMBA"})


az.plot_ppc(idata2, group="prior", observed=True)

with rush_tds2:
    idata2.extend(
        pm.sample(nuts_sampler="numpyro", target_accept=0.99, random_seed=rng)
    )


az.ess(idata2).min().to_pandas().sort_values().round(2)

az.rhat(
    idata2, var_names=["basis_coeffs_season", "basis_coeffs_games"]
).max().to_pandas().round(2)

idata2.sample_stats.diverging.sum().data

with rush_tds2:
    idata2.extend(
        pm.sample_posterior_predictive(idata2, compile_kwargs={"mode": "NUMBA"})
    )
    idata2.extend(pm.compute_log_likelihood(idata2))


az.plot_ppc(idata2)


mods = ["OG mod", "Delta Sig changed"]


mods_dict = dict(zip(mods, [idata, idata2]))

## honestly the og model works well
## the delta sig could be higher but when we change the delta sig
## we push another parameter
az.compare(mods_dict)


mindex_coords = xr.Coordinates.from_pandas_multiindex(
    cumulative_stats_pd.set_index(
        [
            "full_name",
            "number_of_seasons_played",
            "games_played",
            "season",
            "position_group",
        ]
    ).index,
    "obs_id",
)

idata.posterior = idata.posterior.assign_coords(mindex_coords)
idata.posterior_predictive = idata.posterior_predictive.assign_coords(mindex_coords)


implied_probs_post = (
    idata.posterior["tds_scored_probs"]
    .rename({"tds_scored_probs_dim_0": "obs_id", "tds_scored_probs_dim_1": "event"})
    .assign_coords(mindex_coords)
)

player = "Josh Allen"

player_probs_post = implied_probs_post.sel(full_name=player)


colors = plt.cm.viridis(np.linspace(0.05, 0.95, 4))

cols = 2
unique_seasons = np.unique(player_probs_post.season)
num_seasons = len(unique_seasons)
rows = (num_seasons + cols - 1) // cols

fig, axes = plt.subplots(
    rows, cols, figsize=(12, 2.5 * rows), layout="constrained", sharey=True
)

axes = axes.flatten()


player_probs_post_df = player_probs_post.to_dataframe()

for season, (i, ax) in zip(unique_seasons, enumerate(axes)):
    dates = player_probs_post.sel(season=season)["games_played"]
    y_plot = player_probs_post.sel(season=season)

    for event in player_probs_post.event.to_numpy():
        az.plot_hdi(
            x=dates,
            y=y_plot.sel(event=event),
            hdi_prob=0.89,
            color=colors[event],
            fill_kwargs={"alpha": 0.4},
            ax=ax,
            smooth=False,
        )
        ax.plot(
            dates,
            y_plot.sel(event=event).mean(("chain", "draw")),
            lw=2,
            ls="--",
            label=f"{event}",
            color=colors[event],
            alpha=0.9,
        )

    ax.set(xlabel="Day", ylabel="Probability", title=f"{season}")
    sns.despine()
    if i == 0:
        ax.legend(fontsize=10, frameon=True, title="TDs", ncols=4)

# remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(f"{player.title()}\nTDs Probabilities per Season", fontsize=18)

axes[3].axvline(x=7, c="grey", ls="--")


idata.reset_index("season")
