import polars as pl
import pandas as pd
import pymc as pm
import preliz as pz
import numpy as np
from scipy.special import logit
import matplotlib.pyplot as plt
import arviz as az
import nflreadpy as nfl
import xarray as xr
import os

## just copied from  https://github.com/BlakeRMills/MetBrewer/blob/main/Python/met_brewer/palettes.py


#
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
    .filter(
        ## for development lets just get rid of players who never score
        ## e.g. this will get rid of blocking tightends, full backs, and WR's that we probably are not
        ## going to play all that often in fantasy
        ## while we love them
        pl.col("rec_tds_season") > 0
    )
)


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
    )
    .sort(["receiver_full_name", "season", "game_id"])
)


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
    "player_team_score_diff",
    "player_rest_diff",
    "def_epa_per_play",
]


factors = factors_numeric + ["div_game", "home_game", "is_indoors"]

factors_numeric_train = construct_games_played.select(factors_numeric)

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
}


## the argument that Alex and Max make are that the last 2-6 seasons
## tell us something about the current season
## the problem is that in football most players have a 3 year career
## part of the reason that soccer players have longer careers is
## that they don't don't run into each other really fast and really hard all the time
#  but they also start their professional careers at a younger age so
# if you are kind of a bust than you still have a slightly longer career than a player coming out of college
# you also don't have a draft process you may start in the pro academy and then get promoted
# or play against a similar competition so you are going to have a better idea of the tallen
# football doesn't have the luxury so we need a prior that is a little more pessimisitic about your career lenth
fig, ax = plt.subplots(ncols=2)


ax[0].hist(construct_games_played["number_of_seasons_played"])

pz.maxent(pz.InverseGamma(), lower=0.01, upper=8, ax=ax[1])
ax[1].set_xlim(0, 17.5)
ax[1].legend().set_visible(False)
plt.close("all")


seasons_gp_prior, ax = pz.maxent(pz.InverseGamma(), lower=0.01, upper=8)


seasons_m, seasons_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[
        0,
        construct_games_played.select(
            pl.col("number_of_seasons_played").max()
        ).to_series()[0],
    ],
    lengthscale_range=[0.01, 8],
    cov_func="matern52",
)

plt.close("all")

# the short term prior is actually pretty decent
short_term_form, _ = pz.maxent(pz.InverseGamma(), lower=2, upper=6)

plt.xlim(0, 20)

med_form, ax = pz.maxent(pz.InverseGamma(), lower=12, upper=18)


plt.close("all")

construct_games_played.select(pl.col("games_played").mean())

within_m, within_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[
        0,
        construct_games_played.select(pl.col("games_played").max()).to_series()[0],
    ],
    lengthscale_range=[2, 6],
    cov_func="matern52",
)

plt.close("all")

# effectively the difference between a player is about 2 touchdowns
touchdown_dist, ax = pz.maxent(pz.Exponential(), 0, 2)

plt.hist(data=construct_games_played, x="rec_tds_game")

# afer subsetting we get like an 8% chance
# before subsetting it was like a 6% chance
ratio_td_catches = (
    construct_games_played["rec_tds"].value_counts(normalize=True).sort("rec_tds")
)

ratio_td_catches

with pm.Model(coords=coords) as receiving_mod_long:
    gameday_id = pm.Data("gameday_id", games_idx, dims="obs_id")
    seasons_id = pm.Data(
        "season_id",
        construct_games_played_pd["number_of_seasons_played"],
        dims="obs_id",
    )

    off_id = pm.Data("off_play_caller_id", off_play_caller_idx, dims="obs_id")

    def_id = pm.Data("def_play_caller_id", def_play_caller_idx, dims="obs_id")

    x_gamedays = pm.Data("X_gamedays", unique_games, dims="gameday")[:, None]
    x_season = pm.Data("x_season", unique_seasons, dims="seasons")[:, None]

    fct_data = pm.Data(
        "factor_num_data",
        factors_numeric_sdz.to_numpy(),
        dims=("obs_id", "factors"),
    )

    player_id = pm.Data("player_id", player_idx, dims="obs_id")

    td_obs = pm.Data(
        "rec_obs", construct_games_played_pd["rec_tds"].to_numpy(), dims="obs_id"
    )

    sigma_player = touchdown_dist.to_pymc("player_sigma")

    # setting this at the mean
    player_effects = pm.Normal(
        "player_effects",
        mu=logit(construct_games_played_pd["rec_tds"].mean()),
        sigma=sigma_player,
        dims="player",
    )

    ls_games = short_term_form.to_pymc("games_lengthscale_prior")

    # the upper scale is effectively touchdown no touchdown
    # 1% is maybe a little to pessimistic
    # we are going to set it at a tick lower than the observed
    # chance you score a touchdown | on catching a ball
    alpha_scale, upper_scale = 0.24, 1.1

    sigma_games = pm.Exponential("sigma_game", -np.log(alpha_scale) / upper_scale)

    cov_games = sigma_games**2 * pm.gp.cov.Matern52(input_dim=1, ls=ls_games)

    gp_within = pm.gp.HSGP(m=[within_m], c=within_c, cov_func=cov_games)

    basis_vectors_within, sqrt_within = gp_within.prior_linearized(X=x_gamedays)
    basis_coefs_within = pm.Normal(
        "basis_coeffs_within", shape=gp_within.n_basis_vectors
    )
    f_within = pm.Deterministic(
        "f_within",
        basis_vectors_within @ (basis_coefs_within * sqrt_within),
        dims="gameday",
    )

    sigma_season = pm.Exponential("sigma_season", -np.log(alpha_scale) / upper_scale)

    ls_season = seasons_gp_prior.to_pymc(name="seasons_lengthscale_prior")

    cov_season = sigma_season**2 * pm.gp.cov.Matern52(1, ls=ls_season)

    gp_season = pm.gp.HSGP(
        m=[seasons_m], c=seasons_c, cov_func=cov_season, parametrization="centered"
    )

    basis_vectors_long, sqrt_season = gp_season.prior_linearized(X=x_season)

    basis_coefs_long = pm.Normal("basis_coeffs_long", shape=gp_season.n_basis_vectors)

    f_season = pm.Deterministic(
        "f_season",
        basis_vectors_long @ (basis_coefs_long * sqrt_season),
        dims="seasons",
    )

    slope_num = pm.Normal("slope_num", sigma=0.5, dims="factors")

    alpha = pm.Deterministic(
        "alpha",
        player_effects[player_id] + f_within[gameday_id] + f_season[seasons_id],
        dims="obs_id",
    )

    mu_player = pm.Deterministic(
        "mu_player",
        pm.math.sigmoid(alpha + pm.math.dot(fct_data, slope_num)),
        dims="obs_id",
    )

    p = pm.Bernoulli("tds_scored", p=mu_player, observed=td_obs, dims="obs_id")


with receiving_mod_long:
    trace = pm.sample(nuts_sampler="numpyro", random_seed=rng, target_accept=0.99)


az.plot_trace(
    trace,
    var_names=[
        "slope_num",
        "sigma_season",
        "sigma_game",
        "games_lengthscale_prior",
        "seasons_lengthscale_prior",
        "player_effects",
        "player_sigma",
    ],
)

## adding epa back makes the sampling struggle
## if we loosen the prior on the factors that does not really help
az.plot_ess(
    trace,
    kind="evolution",
    var_names=[RV.name for RV in receiving_mod_long.free_RVs if RV.size.eval() <= 3],
    grid=(5, 2),
    textsize=25,
)
az.plot_energy(trace)

f_long_post = trace.posterior["f_season"]
f_within_post = trace.posterior["f_within"]


index = pd.MultiIndex.from_product(
    [unique_seasons, unique_games],
    names=["seasons", "gameday"],
)
unique_combinations = pd.DataFrame(index=index).reset_index()


f_long_post_aligned = f_long_post.sel(
    seasons=unique_combinations["seasons"].to_numpy()
).rename({"seasons": "timestamp"})
f_long_post_aligned["timestamp"] = unique_combinations.index

f_within_post_aligned = f_within_post.sel(
    gameday=unique_combinations["gameday"].to_numpy()
).rename({"gameday": "timestamp"})
f_within_post_aligned["timestamp"] = unique_combinations.index

f_total_post = f_long_post_aligned + f_within_post_aligned

some_draws = rng.choice(int(4 * 1000), size=20, replace=True)

_, axes = plt.subplot_mosaic(
    """
    AB
    CC
    """,
    figsize=(12, 7.5),
    layout="constrained",
)

axes["A"].plot(
    f_within_post.gameday,
    az.extract(f_within_post)["f_within"].isel(sample=0),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
    label="random draws",
)
axes["A"].plot(
    f_within_post.gameday,
    az.extract(f_within_post)["f_within"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_within_post.gameday,
    y=f_within_post,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9, "label": r"$83\%$ HDI"},
    ax=axes["A"],
    smooth=False,
)
axes["A"].plot(
    f_within_post.gameday,
    f_within_post.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
    label="Mean",
)
axes["A"].set(
    xlabel="Gameday", ylabel="Nbr tds", title="Within season variation\nShort GP"
)
axes["A"].legend(fontsize=10, frameon=True, ncols=3)

axes["B"].plot(
    f_long_post.seasons,
    az.extract(f_long_post)["f_season"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_long_post.seasons,
    y=f_long_post,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["B"],
    smooth=False,
)
axes["B"].plot(
    f_long_post.seasons,
    f_long_post.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
)
axes["B"].set(
    xlabel="Season", ylabel="Nbr tds", title="Across seasons variation\nAging curve"
)

axes["C"].plot(
    f_total_post.timestamp,
    az.extract(f_total_post)["x"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_total_post.timestamp,
    y=f_total_post,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["C"],
    smooth=False,
)
axes["C"].plot(
    f_total_post.timestamp,
    f_total_post.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
)
axes["C"].set(xlabel="Timestamp", ylabel="Nbr tds", title="Total GP")
plt.suptitle("Posterior GPs", fontsize=18)


figA, axes = plt.subplot_mosaic(
    """
    A
    """,
    figsize=(6, 3.75),
    layout="constrained",
)

axes["A"].plot(
    f_within_post.gameday,
    az.extract(f_within_post)["f_within"].isel(sample=0),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
    label="random draws",
)
axes["A"].plot(
    f_within_post.gameday,
    az.extract(f_within_post)["f_within"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_within_post.gameday,
    y=f_within_post,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9, "label": r"$83\%$ HDI"},
    ax=axes["A"],
    smooth=False,
)
axes["A"].plot(
    f_within_post.gameday,
    f_within_post.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
    label="Mean",
)
axes["A"].set(xlabel="Gameday", ylabel="Nbr tds", title="")
axes["A"].legend(fontsize=10, frameon=True, ncols=3, loc="upper right")


figB, axes = plt.subplot_mosaic(
    """
    B
    """,
    figsize=(6, 3.75),
    layout="constrained",
)

axes["B"].plot(
    f_long_post.seasons,
    az.extract(f_long_post)["f_season"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_long_post.seasons,
    y=f_long_post,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["B"],
    smooth=False,
)
axes["B"].plot(
    f_long_post.seasons,
    f_long_post.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
)
axes["B"].set(xlabel="Season", ylabel="Nbr tds", title="")


figC, axes = plt.subplot_mosaic(
    """
    C
    """,
    figsize=(12, 3.75),
    layout="constrained",
)

axes["C"].plot(
    f_total_post.timestamp,
    az.extract(f_total_post)["x"].isel(sample=some_draws),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)
az.plot_hdi(
    x=f_total_post.timestamp,
    y=f_total_post,
    hdi_prob=0.83,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["C"],
    smooth=False,
)
axes["C"].plot(
    f_total_post.timestamp,
    f_total_post.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
)
axes["C"].set(xlabel="Timestamp", ylabel="Nbr tds", title="")


check = pl.from_pandas(az.summary(trace).reset_index()).clean_names()

bad_rhats = check.filter(pl.col("r_hat") > 1.0)


factors_numeric2 = ["player_rest_diff", "def_epa_diff", "receiver_epa_diff"]

factors2 = factors_numeric2 + ["div_game", "home_game", "is_indoors"]

factors_numeric_train2 = construct_games_played.select(factors_numeric2)

means = factors_numeric_train2.select(
    [pl.col(c).mean().alias(c) for c in factors_numeric2]
)
sds = factors_numeric_train2.select(
    [pl.col(c).std().alias(c) for c in factors_numeric2]
)

factors_numeric_sdz2 = factors_numeric_train2.with_columns(
    [((pl.col(c) - means[0, c]) / sds[0, c]).alias(c) for c in factors_numeric2]
).with_columns(
    pl.Series("home_game", construct_games_played["home_game"]),
    pl.Series("div_game", construct_games_played["div_game"]),
    pl.Series("is_indoors", construct_games_played["is_indoors"]),
)

coords2 = {
    "factors": factors2,
    "gameday": unique_games,
    "seasons": unique_seasons,
    "obs_id": construct_games_played_pd.index,
    "player": unique_players,
    "off_play_caller": off_play_caller,
    "def_play_caller": def_play_caller,
}


with pm.Model(coords=coords2) as rec_mod_epa:
    gameday_id = pm.Data("gameday_id", games_idx, dims="obs_id")
    seasons_id = pm.Data(
        "season_id",
        construct_games_played_pd["number_of_seasons_played"],
        dims="obs_id",
    )

    off_id = pm.Data("off_play_caller_id", off_play_caller_idx, dims="obs_id")

    def_id = pm.Data("def_play_caller_id", def_play_caller_idx, dims="obs_id")

    x_gamedays = pm.Data("X_gamedays", unique_games, dims="gameday")[:, None]
    x_season = pm.Data("x_season", unique_seasons, dims="seasons")[:, None]

    fct_data = pm.Data(
        "factor_num_data",
        factors_numeric_sdz2.to_numpy(),
        dims=("obs_id", "factors"),
    )

    player_id = pm.Data("player_id", player_idx, dims="obs_id")

    td_obs = pm.Data(
        "rec_obs", construct_games_played_pd["rec_tds"].to_numpy(), dims="obs_id"
    )

    sigma_player = touchdown_dist.to_pymc("player_sigma")

    # setting this at the mean
    player_effects = pm.Normal(
        "player_effects",
        mu=logit(construct_games_played_pd["rec_tds"].mean()),
        sigma=sigma_player,
        dims="player",
    )

    ls_games = short_term_form.to_pymc("games_lengthscale_prior")

    # the upper scale is effectively touchdown no touchdown
    # 1% is maybe a little to pessimistic
    # we are going to set it at a tick lower than the observed
    # chance you score a touchdown | on catching a ball
    alpha_scale, upper_scale = 0.24, 1.1

    sigma_games = pm.Exponential("sigma_game", -np.log(alpha_scale) / upper_scale)

    cov_games = sigma_games**2 * pm.gp.cov.Matern52(input_dim=1, ls=ls_games)

    gp_within = pm.gp.HSGP(m=[within_m], c=within_c, cov_func=cov_games)

    basis_vectors_within, sqrt_within = gp_within.prior_linearized(X=x_gamedays)
    basis_coefs_within = pm.Normal(
        "basis_coeffs_within", shape=gp_within.n_basis_vectors
    )
    f_within = pm.Deterministic(
        "f_within",
        basis_vectors_within @ (basis_coefs_within * sqrt_within),
        dims="gameday",
    )

    sigma_season = pm.Exponential("sigma_season", -np.log(alpha_scale) / upper_scale)

    ls_season = seasons_gp_prior.to_pymc(name="seasons_lengthscale_prior")

    cov_season = sigma_season**2 * pm.gp.cov.Matern52(1, ls=ls_season)

    gp_season = pm.gp.HSGP(
        m=[seasons_m], c=seasons_c, cov_func=cov_season, parametrization="centered"
    )

    basis_vectors_long, sqrt_season = gp_season.prior_linearized(X=x_season)

    basis_coefs_long = pm.Normal("basis_coeffs_long", shape=gp_season.n_basis_vectors)

    f_season = pm.Deterministic(
        "f_season",
        basis_vectors_long @ (basis_coefs_long * sqrt_season),
        dims="seasons",
    )

    slope_num = pm.Normal("slope_num", sigma=0.5, dims="factors")

    alpha = pm.Deterministic(
        "alpha",
        player_effects[player_id] + f_within[gameday_id] + f_season[seasons_id],
        dims="obs_id",
    )

    mu_player = pm.Deterministic(
        "mu_player",
        pm.math.sigmoid(alpha + pm.math.dot(fct_data, slope_num)),
        dims="obs_id",
    )

    p = pm.Bernoulli("tds_scored", p=mu_player, observed=td_obs, dims="obs_id")


with rec_mod_epa:
    trace2 = pm.sample(nuts_sampler="numpyro", random_seed=rng, target_accept=0.99)


fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(14, 20))

vars_epa = [RV.name for RV in rec_mod_epa.free_RVs if RV.size.eval() <= 3]

vars_def_only = [RV.name for RV in receiving_mod_long.free_RVs if RV.size.eval() <= 3]

for i in range(5):
    if i < len(vars_epa):
        az.plot_ess(
            trace2,
            kind="evolution",
            var_names=[vars_epa[i]],
            textsize=10,
            ax=axes[i, 0],
        )
        axes[i, 0].set_title(f"Epa Model-{vars_epa[i]}")
    if i < len(vars_def_only):
        az.plot_ess(
            trace,
            kind="evolution",
            var_names=[vars_def_only[i]],
            textsize=10,
            ax=axes[i, 1],
        )
        axes[i, 1].set_title(f"Def EPA Only-{vars_def_only[i]}")
    else:
        axes[i, 1].axis("off")

    plt.tight_layout()


az.plot_energy(trace2)


## now lets add in weather factors and game script stuff

factors_numeric3 = [
    "player_rest_diff",
    "def_epa_diff",
    "receiver_epa_diff",
    "wind",
    "temp",
    "total_pass_attempts",
    "avg_depth_of_target",
]

factors3 = factors_numeric3 + ["div_game", "home_game", "is_indoors"]

factors_numeric_train3 = construct_games_played.select(pl.col(factors3))

means = factors_numeric_train3.select(
    [pl.col(c).mean().alias(c) for c in factors_numeric3]
)
sds = factors_numeric_train3.select(
    [pl.col(c).std().alias(c) for c in factors_numeric3]
)

factors_numeric_sdz3 = factors_numeric_train3.with_columns(
    [((pl.col(c) - means[0, c]) / sds[0, c]).alias(c) for c in factors_numeric3]
).with_columns(
    pl.Series("home_game", construct_games_played["home_game"]),
    pl.Series("div_game", construct_games_played["div_game"]),
    pl.Series("is_indoors", construct_games_played["is_indoors"]),
)


coords3 = {
    "factors": factors3,
    "gameday": unique_games,
    "seasons": unique_seasons,
    "obs_id": construct_games_played_pd.index,
    "player": unique_players,
    "off_play_caller": off_play_caller,
    "def_play_caller": def_play_caller,
}

with pm.Model(coords=coords3) as rec_mod_add_weather:
    gameday_id = pm.Data("gameday_id", games_idx, dims="obs_id")
    seasons_id = pm.Data(
        "season_id",
        construct_games_played_pd["number_of_seasons_played"],
        dims="obs_id",
    )

    off_id = pm.Data("off_play_caller_id", off_play_caller_idx, dims="obs_id")

    def_id = pm.Data("def_play_caller_id", def_play_caller_idx, dims="obs_id")

    x_gamedays = pm.Data("X_gamedays", unique_games, dims="gameday")[:, None]
    x_season = pm.Data("x_season", unique_seasons, dims="seasons")[:, None]

    fct_data = pm.Data(
        "factor_num_data",
        factors_numeric_sdz3.to_numpy(),
        dims=("obs_id", "factors"),
    )

    player_id = pm.Data("player_id", player_idx, dims="obs_id")

    td_obs = pm.Data(
        "rec_obs", construct_games_played_pd["rec_tds"].to_numpy(), dims="obs_id"
    )

    sigma_player = touchdown_dist.to_pymc("player_sigma")

    # setting this at the mean
    player_effects = pm.Normal(
        "player_effects",
        mu=logit(construct_games_played_pd["rec_tds"].mean()),
        sigma=sigma_player,
        dims="player",
    )

    ls_games = short_term_form.to_pymc("games_lengthscale_prior")

    # the upper scale is effectively touchdown no touchdown
    # 1% is maybe a little to pessimistic
    # we are going to set it at a tick lower than the observed
    # chance you score a touchdown | on catching a ball
    alpha_scale, upper_scale = 0.24, 1.1

    sigma_games = pm.Exponential("sigma_game", -np.log(alpha_scale) / upper_scale)

    cov_games = sigma_games**2 * pm.gp.cov.Matern52(input_dim=1, ls=ls_games)

    gp_within = pm.gp.HSGP(m=[within_m], c=within_c, cov_func=cov_games)

    basis_vectors_within, sqrt_within = gp_within.prior_linearized(X=x_gamedays)
    basis_coefs_within = pm.Normal(
        "basis_coeffs_within", shape=gp_within.n_basis_vectors
    )
    f_within = pm.Deterministic(
        "f_within",
        basis_vectors_within @ (basis_coefs_within * sqrt_within),
        dims="gameday",
    )

    sigma_season = pm.Exponential("sigma_season", -np.log(alpha_scale) / upper_scale)

    ls_season = seasons_gp_prior.to_pymc(name="seasons_lengthscale_prior")

    cov_season = sigma_season**2 * pm.gp.cov.Matern52(1, ls=ls_season)

    gp_season = pm.gp.HSGP(
        m=[seasons_m], c=seasons_c, cov_func=cov_season, parametrization="centered"
    )

    basis_vectors_long, sqrt_season = gp_season.prior_linearized(X=x_season)

    basis_coefs_long = pm.Normal("basis_coeffs_long", shape=gp_season.n_basis_vectors)

    f_season = pm.Deterministic(
        "f_season",
        basis_vectors_long @ (basis_coefs_long * sqrt_season),
        dims="seasons",
    )

    slope_num = pm.Normal("slope_num", sigma=0.5, dims="factors")

    alpha = pm.Deterministic(
        "alpha",
        player_effects[player_id] + f_within[gameday_id] + f_season[seasons_id],
        dims="obs_id",
    )

    mu_player = pm.Deterministic(
        "mu_player",
        pm.math.sigmoid(alpha + pm.math.dot(fct_data, slope_num)),
        dims="obs_id",
    )

    p = pm.Bernoulli("tds_scored", p=mu_player, observed=td_obs, dims="obs_id")


factors_numeric4 = ["player_rest_diff", "def_epa_diff"]

factors4 = factors_numeric4 + ["div_game", "home_game", "is_indoors"]

factors_numeric_train4 = construct_games_played.select(factors_numeric4)

means = factors_numeric_train4.select(
    [pl.col(c).mean().alias(c) for c in factors_numeric4]
)
sds = factors_numeric_train4.select(
    [pl.col(c).std().alias(c) for c in factors_numeric4]
)

factors_numeric_sdz4 = factors_numeric_train4.with_columns(
    [((pl.col(c) - means[0, c]) / sds[0, c]).alias(c) for c in factors_numeric4]
).with_columns(
    pl.Series("home_game", construct_games_played["home_game"]),
    pl.Series("div_game", construct_games_played["div_game"]),
    pl.Series("is_indoors", construct_games_played["is_indoors"]),
)

coords4 = {
    "factors": factors4,
    "gameday": unique_games,
    "seasons": unique_seasons,
    "obs_id": construct_games_played_pd.index,
    "player": unique_players,
    "off_play_caller": off_play_caller,
    "def_play_caller": def_play_caller,
}


with pm.Model(coords=coords4) as rec_mod_just_def_epa:
    gameday_id = pm.Data("gameday_id", games_idx, dims="obs_id")
    seasons_id = pm.Data(
        "season_id",
        construct_games_played_pd["number_of_seasons_played"],
        dims="obs_id",
    )

    off_id = pm.Data("off_play_caller_id", off_play_caller_idx, dims="obs_id")

    def_id = pm.Data("def_play_caller_id", def_play_caller_idx, dims="obs_id")

    x_gamedays = pm.Data("X_gamedays", unique_games, dims="gameday")[:, None]
    x_season = pm.Data("x_season", unique_seasons, dims="seasons")[:, None]

    fct_data = pm.Data(
        "factor_num_data",
        factors_numeric_sdz.to_numpy(),
        dims=("obs_id", "factors"),
    )

    player_id = pm.Data("player_id", player_idx, dims="obs_id")

    td_obs = pm.Data(
        "rec_obs", construct_games_played_pd["rec_tds"].to_numpy(), dims="obs_id"
    )

    sigma_player = touchdown_dist.to_pymc("player_sigma")

    # setting this at the mean
    player_effects = pm.Normal(
        "player_effects",
        mu=logit(construct_games_played_pd["rec_tds"].mean()),
        sigma=sigma_player,
        dims="player",
    )

    ls_games = short_term_form.to_pymc("games_lengthscale_prior")

    # the upper scale is effectively touchdown no touchdown
    # 1% is maybe a little to pessimistic
    # we are going to set it at a tick lower than the observed
    # chance you score a touchdown | on catching a ball
    alpha_scale, upper_scale = 0.24, 1.1

    sigma_games = pm.Exponential("sigma_game", -np.log(alpha_scale) / upper_scale)

    cov_games = sigma_games**2 * pm.gp.cov.Matern52(input_dim=1, ls=ls_games)

    gp_within = pm.gp.HSGP(m=[within_m], c=within_c, cov_func=cov_games)

    basis_vectors_within, sqrt_within = gp_within.prior_linearized(X=x_gamedays)
    basis_coefs_within = pm.Normal(
        "basis_coeffs_within", shape=gp_within.n_basis_vectors
    )
    f_within = pm.Deterministic(
        "f_within",
        basis_vectors_within @ (basis_coefs_within * sqrt_within),
        dims="gameday",
    )

    sigma_season = pm.Exponential("sigma_season", -np.log(alpha_scale) / upper_scale)

    ls_season = seasons_gp_prior.to_pymc(name="seasons_lengthscale_prior")

    cov_season = sigma_season**2 * pm.gp.cov.Matern52(1, ls=ls_season)

    gp_season = pm.gp.HSGP(
        m=[seasons_m], c=seasons_c, cov_func=cov_season, parametrization="centered"
    )

    basis_vectors_long, sqrt_season = gp_season.prior_linearized(X=x_season)

    basis_coefs_long = pm.Normal("basis_coeffs_long", shape=gp_season.n_basis_vectors)

    f_season = pm.Deterministic(
        "f_season",
        basis_vectors_long @ (basis_coefs_long * sqrt_season),
        dims="seasons",
    )

    slope_num = pm.Normal("slope_num", sigma=0.5, dims="factors")

    alpha = pm.Deterministic(
        "alpha",
        player_effects[player_id] + f_within[gameday_id] + f_season[seasons_id],
        dims="obs_id",
    )

    mu_player = pm.Deterministic(
        "mu_player",
        pm.math.sigmoid(alpha + pm.math.dot(fct_data, slope_num)),
        dims="obs_id",
    )

    p = pm.Bernoulli("tds_scored", p=mu_player, observed=td_obs, dims="obs_id")

with rec_mod_just_def_epa:
    trace4 = pm.sample(nuts_sampler="numpyro", random_seed=1994, target_accept=0.99)


with rec_mod_add_weather:
    trace3 = pm.sample_prior_predictive()


with receiving_mod_long:
    pm.compute_log_likelihood(trace)
    trace.extend(pm.sample_posterior_predictive(trace))


with rec_mod_epa:
    pm.compute_log_likelihood(trace2)
    trace2.extend(pm.sample_posterior_predictive(trace2))

#
with rec_mod_add_weather:
    trace3 = pm.sample_prior_predictive()
    trace3.extend(
        pm.sample(nuts_sampler="numpyro", random_seed=rng, target_accept=0.99)
    )
    trace3.extend(pm.sample_posterior_predictive(trace3))
    trace3.extend(pm.compute_log_likelihood(trace3))


mods = [
    "Just DEF EPA",
    "EPA + Weather + targets + pass attempts",
    "OFF DEF EPA",
    "Score + DEF EPA",
]


mods_dict = dict(zip(mods, [trace4, trace3, trace2, trace]))

az.compare(mods_dict)

mods2 = [
    "OFF + DEF EPA",
    "SCORE + DEF EPA",
    '"EPA + Weather + targets + pass attempts"',
]

mods_dict2 = dict(zip(mods2, [trace2, trace, trace3]))

mods_dict2
# these are more or less fine
comps = az.compare(mods_dict2)


az.plot_ppc(trace3)

az.plot_ess(
    trace3,
    kind="evolution",
    var_names=[RV.name for RV in rec_mod_add_weather.free_RVs if RV.size.eval() <= 3],
    grid=(5, 2),
    textsize=25,
)


az.plot_energy(trace3)

## So it is really hard to define "replacements"
## so lets take players in the lowest quartile
## the problem is that the positions are not neccessariloy
## comparble so lets do this by position

elite = (
    construct_games_played.unique(["receiver_full_name", "season", "receiver_position"])
    .with_columns(
        pl.col("rec_tds_season")
        .rank(descending=True)
        .over(["receiver_position", "season"])
        .alias("rank_season")
    )
    .filter((pl.col("rank_season") <= 10) & (pl.col("season") == 2023))
    .sort(["receiver_position", "rank_season"])["receiver_full_name"]
)


mindex_coords_original = xr.Coordinates.from_pandas_multiindex(
    construct_games_played_pd.set_index(
        ["receiver_full_name", "season", "games_played"]
    ).index,
    "obs_id",
)

construct_games_played_pd.columns


trace3.posterior = trace3.posterior.assign_coords(mindex_coords_original)

trace3.posterior_predictive = trace3.posterior_predictive.assign_coords(
    mindex_coords_original
)

elite_players = elite.to_list()


trace3_cop = trace3.copy()


trace3_cop = trace3_cop.posterior.sel(player=elite_players)

means = trace3.posterior.player_effects.mean()


## this is generally the plot i am looking for
az.plot_forest(
    trace3_cop,
    var_names=["player_effects"],
    combined=True,
    colors="#6c1d0e",
)


ax = plt.gca()
labs = [item.get_text() for item in ax.get_yticklabels()]

cleaned_labs = []
for i in labs:
    clean = i.replace("player_effects[", "").replace("]", "").replace("[", "")
    cleaned_labs.append(clean)

ax.set_yticklabels(cleaned_labs)
plt.axvline(x=means, c="grey", ls="--")


## now lets make a plot by seasons
## the issue is that we are doing it by number of seasons played rather than
## the current season. So we just need to map it from the rookie season

# Lets take Christian McCaffrey
# This will be interesting because he gets traded from San Francisco in 2022


PLAYER = "Christian McCaffrey"


player_probs_post = trace3.posterior.mu_player.sel(receiver_full_name=PLAYER)
cols = 2
player_unique_seasons = np.unique(player_probs_post.season)
num_seasons = len(player_unique_seasons)
player_rookie_year = construct_games_played.filter(
    pl.col("receiver_full_name") == PLAYER
).select(pl.col("rookie_season").unique())["rookie_season"][0]
rows = (num_seasons + cols) // 2

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(12, 2.5 * rows),
    layout="constrained",
    sharey=True,
    sharex="row",
)


axes = axes.flatten()
for season, (i, ax) in zip(player_unique_seasons, enumerate(axes)):
    dates = player_probs_post.sel(season=season)["games_played"]
    y_plot = player_probs_post.sel(season=season)

    obs_plot = (
        construct_games_played.filter(
            (pl.col("receiver_full_name") == PLAYER) & (pl.col("season") == season)
        )
        .select("rec_tds")
        .to_series()
        .value_counts(normalize=True, sort=False)
        .sort("rec_tds")
        .to_pandas()
    )
    pm.gp.util.plot_gp_dist(
        x=dates.to_numpy(),
        samples=az.extract(y_plot, var_names=["mu_player"]).to_numpy().T,
        ax=ax,
        palette="viridis",
    )  # have to use built in palettes :(

    ax.set(xlabel="Week", ylabel="Probability", title=f"{season}")


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


plt.suptitle(f"{PLAYER}'s\n Expected Receiving TDS per Season", fontsize=18)


# axes[5]

axes[5].axvline(x=7, c="grey", ls="--")
axes[5].annotate(
    "Trade to 49ers",
    xy=(7, 0.55),  # Point to annotate (the line at week 7)
    xytext=(10, 0.65),  # Position of the text
    xycoords="data",
    textcoords="data",
    arrowprops=dict(facecolor="black", shrinkA=5, shrinkB=5, arrowstyle="->"),
    horizontalalignment="left",
    verticalalignment="center",
    fontsize=10,
)
