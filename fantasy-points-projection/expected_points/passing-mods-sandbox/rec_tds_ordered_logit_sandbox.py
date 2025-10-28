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
        pl.when(pl.col("rec_tds_game") >= 3)
        .then(3)
        .otherwise(pl.col("rec_tds_game"))
        .alias("rec_tds")
    )
    .with_columns(
        pl.col("rec_tds_game")
        .sum()
        .over(["receiver_full_name", "season"])
        .alias("rec_tds_season"),
    )
)

agg_full_seasons.select(pl.col("rec_tds_game").max())


# we are goinng to effectively do games played

construct_games_played = (
    agg_full_seasons.with_columns(
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
        (pl.col("player_team_score") - pl.col("opponent_score")).alias(
            "player_team_score_diff"
        ),
        (pl.col("opponent_score") - pl.col("player_team_score")).alias(
            "opponent_score_diff"
        ),
        (pl.col("player_rest") - pl.col("opponent_rest")).alias("player_rest_diff"),
        (pl.col("opponent_rest") - pl.col("player_rest")).alias("opponent_rest_diff"),
        (pl.col("total_off_epa_game") - pl.col("total_def_epa_game")).alias(
            "receiver_epa_diff"
        ),
        (pl.col("total_def_epa_game") - pl.col("total_off_epa_game")).alias(
            "def_epa_diff"
        ),
    )
    .with_columns(
        pl.col("game_id")
        .cum_count()
        .over(["receiver_full_name", "season"])
        .alias("games_played"),
        (pl.col("receiving_yards") - pl.col("receiving_yards").mean()).alias(
            "receiving_yards_c"
        ),
    )
    .join(player_exp, left_on=["receiver_player_id"], right_on="gsis_id", how="left")
    .with_columns(
        (pl.col("season") - pl.col("rookie_season")).alias("number_of_seasons_played"),
        pl.col("birth_date").str.to_date().dt.year().alias("birth_year"),
    )
    .with_columns(
        (pl.col("season") - pl.col("birth_year")).alias("age"),
        pl.when(pl.col("roof") == "indoors").then(1).otherwise(0).alias("is_indoors"),
        ((pl.col("rec_tds_game") - 0) / (pl.lit(4) - pl.lit(0))).alias(
            "rec_tds_game_prop"
        ),
    )
    .sort(["receiver_full_name", "season", "game_id"])
)

## lets check the distribution of number of receiving tds in a game
## functionally the data are bounded you can't score negative touchdowns
## the technical upper limit is 5 but nobody has done done it since Jerry did it in the 90's
construct_games_played.select(pl.col("rec_tds_game").max())

plt.hist(construct_games_played["rec_tds_game"])


plt.close("all")
# this looks good
## bounded between 0 and 1 was what we were lookin for
construct_games_played.group_by(pl.col("rec_tds_game_prop")).agg(pl.len()).sort(
    "rec_tds_game_prop"
)

plt.hist(construct_games_played["rec_tds_game_prop"])


# For whatever reason constructing idx natively in polars
# and then feeding it to pymc is kind of a pain
construct_games_played_pd = construct_games_played.to_pandas()


unique_games = construct_games_played_pd["games_played"].sort_values().unique()
unique_seasons = (
    construct_games_played_pd["number_of_seasons_played"].sort_values().unique()
)

off_play_caller = construct_games_played_pd["off_play_caller"].sort_values().unique()
def_play_caller = construct_games_played_pd["def_play_caller"].sort_values().unique()

unique_players = construct_games_played_pd["receiver_full_name"].sort_values().unique()


player_idx = pd.Categorical(
    construct_games_played_pd["receiver_full_name"], categories=unique_players
).codes

games_idx = pd.Categorical(
    construct_games_played_pd["games_played"], categories=unique_games
).codes

off_play_caller_idx = pd.Categorical(
    construct_games_played_pd["off_play_caller"], categories=off_play_caller
).codes

def_play_caller_idx = pd.Categorical(
    construct_games_played_pd["def_play_caller"], categories=def_play_caller
).codes


factors_numeric = [
    "player_rest_diff",
    "def_epa_diff",
    "receiver_epa_diff",
    "wind",
    "temp",
    "total_pass_attempts",
    "avg_depth_of_target",
]

factors = factors_numeric + ["div_game", "home_game", "is_indoors"]

factors_numeric_train = construct_games_played.select(pl.col(factors))

means = factors_numeric_train.select(
    [pl.col(c).mean().alias(c) for c in factors_numeric]
)
sds = factors_numeric_train.select([pl.col(c).std().alias(c) for c in factors_numeric])

factors_numeric_sdz = factors_numeric_train.with_columns(
    [((pl.col(c) - means[0, c]) / sds[0, c]).alias(c) for c in factors_numeric]
).with_columns(
    pl.Series("home_game", construct_games_played["home_game"]),
    pl.Series("div_game", construct_games_played["div_game"]),
    pl.Series("is_indoors", construct_games_played["is_indoors"]),
)


coords = {
    "factors": factors,
    "gameday": unique_games,
    "seasons": unique_seasons,
    "obs_id": construct_games_played_pd.index,
    "player": unique_players,
    "off_play_caller": off_play_caller,
    "def_play_caller": def_play_caller,
    "time_scale": ["games", "season"],
}

fig, ax = plt.subplots(ncols=2)


ax[0].hist(construct_games_played["number_of_seasons_played"])

pz.maxent(pz.InverseGamma(), lower=0.0, upper=8, ax=ax[1])
ax[1].set_xlim(0, 17.5)
ax[1].legend().set_visible(False)
plt.close("all")


seasons_gp_prior, ax = pz.maxent(pz.InverseGamma(), lower=1, upper=8)

plt.xlim(0, 10)

seasons_m, seasons_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[
        0,
        construct_games_played.select(
            pl.col("number_of_seasons_played").max()
        ).to_series()[0],
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
        construct_games_played.select(pl.col("games_played").max()).to_series()[0],
    ],
    lengthscale_range=[2, 5],
    cov_func="matern52",
)

plt.close("all")

# effectively the difference between a player is about 2 touchdowns
touchdown_dist, ax = pz.maxent(pz.Exponential(), 0, 1)


observed_ratio = (
    construct_games_played["rec_tds_game"]
    .value_counts(normalize=True)
    .sort("rec_tds_game")
)


observed_ratio


empirical_probs = (
    construct_games_played_pd["rec_tds"].value_counts(normalize=True).to_numpy()
)

cumulative_probs = empirical_probs.cumsum()[:-1]

cutpoints_standard = norm.ppf(cumulative_probs)

delta_prior = np.diff(cutpoints_standard)

delta_prior


with pm.Model(coords=coords) as rec_tds_mod:
    factor_data = pm.Data("factor_data", factors_numeric_sdz, dims=("obs_id", "factor"))
    games_id = pm.Data("games_id", games_idx, dims="obs_id")
    player_id = pm.Data("player_id", player_idx, dims="obs_id")
    season_id = pm.Data(
        "season_id",
        construct_games_played["number_of_seasons_played"].to_numpy(),
        dims="obs_id",
    )

    rec_tds_obs = pm.Data(
        "rec_tds_obs", construct_games_played["rec_tds"].to_numpy(), dims="obs_id"
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


with rec_tds_mod:
    idata = pm.sample_prior_predictive()


mindex_coords_original = xr.Coordinates.from_pandas_multiindex(
    construct_games_played_pd.set_index(
        [
            "receiver_full_name",
            "number_of_seasons_played",
            "games_played",
            "season",
            "receiver_position",
        ]
    ).index,
    "obs_id",
)

idata.prior = idata.prior.rename({"tds_scored_probs_dim_1": "event"})

idata.prior = idata.prior.assign_coords(mindex_coords_original)
idata.prior_predictive = idata.prior_predictive.assign_coords(mindex_coords_original)
idata.observed_data = idata.observed_data.assign_coords(mindex_coords_original)

az.plot_forest(idata.prior["cutpoints"].sel(cutpoints_dim_0=0))


implied_probs_prior = (
    az.extract(idata.prior, var_names="tds_scored_probs")
    .rename({"tds_scored_probs_dim_0": "obs_id"})
    .assign_coords(mindex_coords_original)
)


_, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, layout="constrained")
axes = axes.flatten()

implied_probs_prior

for i, ax in enumerate(axes):
    az.plot_dist(implied_probs_prior.isel(obs_id=0, event=i), ax=ax)
    ax.set(xlabel="Probability", yticklabels="", title=f"{i} tds", xlim=[0, 1])
    ax.set_yticks([])
    sns.despine()
plt.suptitle("Prior probability of each event", fontsize=17)


implied_cats = az.extract(idata.prior_predictive, var_names=["tds_scored"])

plt.close("all")

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
    construct_games_played_pd["rec_tds"]
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


f_within_prior = idata.prior["f_games"]
f_long_prior = idata.prior["f_season"]

index = pd.MultiIndex.from_product(
    [unique_seasons, unique_games],
    names=["season_nbr", "gameday"],
)
unique_combinations = pd.DataFrame(index=index).reset_index()

f_long_prior_aligned = f_long_prior.sel(
    seasons=unique_combinations["season_nbr"].to_numpy()
).rename({"seasons": "timestamp"})
f_long_prior_aligned["timestamp"] = unique_combinations.index

f_within_prior_aligned = f_within_prior.sel(
    gameday=unique_combinations["gameday"].to_numpy()
).rename({"gameday": "timestamp"})
f_within_prior_aligned["timestamp"] = unique_combinations.index

f_total_prior = f_long_prior_aligned + f_within_prior_aligned

some_draws = rng.choice(f_total_prior.draw, size=20, replace=True)

_, axes = plt.subplot_mosaic(
    """
    AB
    CC
    """,
    figsize=(12, 7.5),
    layout="constrained",
)

axes["A"].plot(
    f_within_prior.gameday,
    az.extract(f_within_prior)["f_games"].isel(sample=0),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
    label="random draws",
)
axes["A"].plot(
    f_within_prior.gameday,
    az.extract(f_within_prior)["f_games"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_within_prior.gameday,
    y=f_within_prior,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9, "label": r"$83\%$ HDI"},
    ax=axes["A"],
    smooth=False,
)
axes["A"].plot(
    f_within_prior.gameday,
    f_within_prior.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
    label="Mean",
)
axes["A"].set(
    xlabel="Gameday", ylabel="Nbr TDs", title="Within season variation\nShort GP"
)
axes["A"].legend(fontsize=10, frameon=True, ncols=3)

axes["B"].plot(
    f_long_prior.seasons,
    az.extract(f_long_prior)["f_season"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_long_prior.seasons,
    y=f_long_prior,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["B"],
    smooth=False,
)
axes["B"].plot(
    f_long_prior.seasons,
    f_long_prior.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
)
axes["B"].set(
    xlabel="Season", ylabel="Nbr TDs", title="Across seasons variation\nAging curve"
)

axes["C"].plot(
    f_total_prior.timestamp,
    az.extract(f_total_prior)["x"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_total_prior.timestamp,
    y=f_total_prior,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["C"],
    smooth=False,
)
axes["C"].plot(
    f_total_prior.timestamp,
    f_total_prior.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
)
axes["C"].set(xlabel="Timestamp", ylabel="Nbr TDS", title="Total GP")
plt.suptitle("Prior GPs", fontsize=18)


with rec_tds_mod:
    idata.extend(pm.sample(nuts_sampler="numpyro", target_accept=0.99))

## I would prefer if we were at zero for this one
idata.sample_stats.diverging.sum().data

## with the centered season gp we get the rhats under control
az.rhat(
    idata, var_names=["basis_coeffs_season", "basis_coeffs_games"]
).max().to_pandas().round(2)


## The effective sample sizes are looking okay
## I would love if we had larger ESS's for delta sig and touchdown sd
## but otherwise everything else looks good
az.ess(idata).min().to_pandas().sort_values().round()


f_within_posterior = idata.posterior["f_games"]
f_long_posterior = idata.posterior["f_season"]

index = pd.MultiIndex.from_product(
    [unique_seasons, unique_games],
    names=["season_nbr", "gameday"],
)
unique_combinations = pd.DataFrame(index=index).reset_index()

f_long_posterior_aligned = f_long_posterior.sel(
    seasons=unique_combinations["season_nbr"].to_numpy()
).rename({"seasons": "timestamp"})
f_long_posterior_aligned["timestamp"] = unique_combinations.index

f_within_posterior_aligned = f_within_posterior.sel(
    gameday=unique_combinations["gameday"].to_numpy()
).rename({"gameday": "timestamp"})
f_within_posterior_aligned["timestamp"] = unique_combinations.index

f_total_posterior = f_long_posterior_aligned + f_within_posterior_aligned

some_draws = rng.choice(f_total_prior.draw, size=20, replace=True)

_, axes = plt.subplot_mosaic(
    """
    AB
    CC
    """,
    figsize=(12, 7.5),
    layout="constrained",
)

axes["A"].plot(
    f_within_posterior.gameday,
    az.extract(f_within_posterior)["f_games"].isel(sample=0),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
    label="random draws",
)
axes["A"].plot(
    f_within_posterior.gameday,
    az.extract(f_within_posterior)["f_games"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_within_posterior.gameday,
    y=f_within_posterior,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9, "label": r"$83\%$ HDI"},
    ax=axes["A"],
    smooth=False,
)
axes["A"].plot(
    f_within_posterior.gameday,
    f_within_posterior.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
    label="Mean",
)
axes["A"].set(
    xlabel="Gameday", ylabel="Nbr TDs", title="Within season variation\nShort GP"
)
axes["A"].legend(fontsize=10, frameon=True, ncols=3)

axes["B"].plot(
    f_long_posterior.seasons,
    az.extract(f_long_posterior)["f_season"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_long_posterior.seasons,
    y=f_long_posterior,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["B"],
    smooth=False,
)
axes["B"].plot(
    f_long_posterior.seasons,
    f_long_posterior.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
)
axes["B"].set(
    xlabel="Season", ylabel="Nbr TDs", title="Across seasons variation\nAging curve"
)

axes["C"].plot(
    f_total_posterior.timestamp,
    az.extract(f_total_posterior)["x"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_total_posterior.timestamp,
    y=f_total_posterior,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["C"],
    smooth=False,
)
axes["C"].plot(
    f_total_posterior.timestamp,
    f_total_posterior.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
)
axes["C"].set(xlabel="Timestamp", ylabel="Nbr TDS", title="Total GP")
plt.suptitle("Posterior GPs", fontsize=18)

with rec_tds_mod:
    idata.extend(
        pm.sample_posterior_predictive(idata, compile_kwargs={"mode": "NUMBA"})
    )


idata.posterior = idata.posterior.assign_coords(mindex_coords_original)
idata.posterior_predictive = idata.posterior_predictive.assign_coords(
    mindex_coords_original
)

idata.observed_data = idata.observed_data.assign_coords(mindex_coords_original)

post_preds = idata.posterior_predictive.reset_index("obs_id")


replacement_list = (
    construct_games_played.unique(["receiver_full_name", "season"])
    .with_columns(
        pl.col("rec_tds_season")
        .rank(method="ordinal")
        .over(["receiver_position", "season"])
        .alias("position_rank")
    )
    .filter((pl.col("position_rank") <= 5) & (pl.col("season") == 2024))[
        "receiver_full_name"
    ]
)


elite_list = (
    construct_games_played.unique(["receiver_full_name", "season"])
    .with_columns(
        pl.col("rec_tds_season")
        .rank(method="ordinal", descending=True)
        .over(["receiver_position", "season"])
        .alias("position_rank")
    )
    .filter((pl.col("position_rank") <= 5) & (pl.col("season") == 2024))
    .sort(["receiver_position", "position_rank"])["receiver_full_name"]
)


rpl_pef = post_preds["tds_scored"].where(
    (
        (post_preds["receiver_full_name"].isin(replacement_list))
        & (post_preds["season"] == 2024)
    ),
    drop=True,
)

elite_perf = post_preds["tds_scored"].where(
    (
        (post_preds["receiver_full_name"].isin(elite_list))
        & (post_preds["season"] == 2024)
    ),
    drop=True,
)

PAR = (
    elite_perf.groupby(["receiver_full_name"]).mean("obs_id") - rpl_pef.mean("obs_id")
).rename("PAR")


az.plot_forest(PAR, combined=True, colors="#6c1d0e", figsize=(8, 12))

ax = plt.gca()

labs = [item.get_text() for item in ax.get_yticklabels()]
cleaned_labs = []
for i in labs:
    clean = i.replace("PAR[", "").replace("]", "").replace("[", "")
    cleaned_labs.append(clean)

clean_labs = sorted(cleaned_labs)

## this looks fine
## the problem now is that we are comparing apples and oranges
## It makes sense that the receivers are crushing the over expected metric
## we don't expect TE and RBs to be in the same stratosphere as a wr


ax.set_yticklabels(clean_labs)
ax.axvline(c="k", ls="--", alpha=0.8)

##  we can get get and apple to apples comparision with


rpl_pef = post_preds["tds_scored"].where(
    (
        (post_preds["receiver_full_name"].isin(replacement_list))
        & (post_preds["season"] == 2024)
        & (post_preds["receiver_position"] == "TE")
    ),
    drop=True,
)

elite_perf = post_preds["tds_scored"].where(
    (
        (post_preds["receiver_full_name"].isin(elite_list))
        & (post_preds["season"] == 2024)
        & (post_preds["receiver_position"] == "TE")
    ),
    drop=True,
)

PAR = (
    elite_perf.groupby(["receiver_full_name"]).mean("obs_id") - rpl_pef.mean("obs_id")
).rename("PAR")


az.plot_forest(PAR, combined=True, colors="#6c1d0e", figsize=(8, 12))

ax = plt.gca()


labs = [item.get_text() for item in ax.get_yticklabels()]
cleaned_labs = []
for i in labs:
    clean = i.replace("PAR[", "").replace("]", "").replace("[", "")
    cleaned_labs.append(clean)

clean_labs = sorted(cleaned_labs)
ax.set_yticklabels(clean_labs)
ax.axvline(c="k", ls="--", alpha=0.8)
ax.set(title="Performance Above Replacement", xlabel="Receiving Touchdown")
# in all honesty I don't love graphing in matplotlib
# I know how to clean up the look of the plot but I still find them less aesthtically appealing


posterior_predictions = post_preds.to_dataframe()

post_preds_clean = pl.from_pandas(posterior_predictions.reset_index()).clean_names()


post_preds_clean.write_parquet(
    "rec-tds-posterior/rec-tds-posterior.parquet", use_pyarrow=True
)
