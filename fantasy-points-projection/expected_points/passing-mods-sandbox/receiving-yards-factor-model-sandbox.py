import polars as pl
import pandas as pd
import pymc as pm
import preliz as pz
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import nflreadpy as nfl

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
]


rec_data_full = (
    full_pass_data.with_columns(
        pl.col("pass_attempt")
        .sum()
        .over(["receiver_full_name", "game_id"])
        .alias("targeted"),
        pl.col("pass_attempt")
        .sum()
        .over(["posteam", "game_id"])
        .alias("total_pass_attempts"),
    )
    .filter((pl.col("complete_pass") == "1") & (pl.col("week") <= 18))
    .select(pl.col(rec_predictors))
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

c = rec_data_full.filter(
    (pl.col("posteam") == "SF") & (pl.col("defteam") == "SEA")
).sort(["season", "week"])


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
        # how efficient was the offense in the game
        pl.col("epa")
        .mean()
        .over(["game_id", "posteam", "season"])
        .alias("off_epa_per_play"),
        # how efficient was the defense
        pl.col("defensive_epa")
        .mean()
        .over(["game_id", "defteam", "season"])
        .alias("def_epa_per_play"),
    )
    .unique(subset=["game_id", "receiver_full_name", "season"])
    .select(
        # get rid of the per play to not have any confusion
        pl.exclude("epa", "defensive_epa")
    )
)


## lets construct a lookup table
## We want players who are at or around their position means


game_id_check = (
    agg_full_seasons.group_by(["game_id", "season", "receiver_full_name"])
    .agg(pl.len().alias("count"))
    .filter(pl.col("count") > 1)
)


joined_scores = (
    agg_full_seasons.join(clean_full_scores, on=["game_id"], how="left")
    .filter(
        ((pl.col("receptions_season") >= 45) & (pl.col("receiver_position") == "WR"))
        | ((pl.col("receptions_season") >= 35) & (pl.col("receiver_position") == "TE"))
        | ((pl.col("receptions_season") >= 30) & (pl.col("receiver_position") == "RB"))
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
    )
)


# we do have to construct the week indexes a bit different
# since we have to construct indices for each player
# we are goinng to effectively do games played

construct_games_played = (
    joined_scores.with_columns(
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
    .with_columns((pl.col("season") - pl.col("birth_year")).alias("age"))
    .sort(["receiver_full_name", "season", "game_id"])
)

construct_games_played_pd = construct_games_played.to_pandas()

construct_games_played_pd.head()

unique_games = construct_games_played_pd["games_played"].unique()
unique_seasons = construct_games_played_pd["season"].unique()

off_play_caller = construct_games_played_pd["off_play_caller"].unique()
def_play_caller = construct_games_played_pd["def_play_caller"].unique()

unique_players = construct_games_played_pd["receiver_full_name"].unique()


player_idx = pd.Categorical(
    construct_games_played_pd["receiver_full_name"], categories=unique_players
).codes

games_idx = pd.Categorical(
    construct_games_played_pd["week"], categories=unique_games
).codes

off_play_caller_idx = pd.Categorical(
    construct_games_played_pd["off_play_caller"], categories=off_play_caller
).codes

def_play_caller_idx = pd.Categorical(
    construct_games_played_pd["def_play_caller"], categories=def_play_caller
).codes

factors_numeric = [
    "player_team_score_diff",
    "opponent_score_diff",
    "player_rest_diff",
    "opponent_rest_diff",
    "avg_depth_of_target",
    "off_epa_per_play",
    "def_epa_per_play",
    "total_pass_attempts",
    "wind",
    "temp",
]

factor_data = construct_games_played.select(pl.col(factors_numeric)).with_columns(
    [
        ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std())
        for col in construct_games_played.select(factors_numeric).columns
    ]
)

construct_games_played.group_by("season").agg(
    pl.col("receiving_yards").mean().alias("avg_receiving"),
    pl.col("receiving_yards").std().alias("std_receiving"),
)


coords = {
    "factors_num": factors_numeric,
    "gameday": unique_games,
    "seasons": unique_seasons,
    "obs_id": np.arange(len(construct_games_played)),
    "player": unique_players,
    "off_play_caller": off_play_caller,
    "def_play_caller": def_play_caller,
}


seasons_gp_prior, ax = pz.maxent(pz.Gamma(), lower=2, upper=8)

plt.close("all")


seasons_m, seasons_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[
        0,
        construct_games_played.select(
            pl.col("number_of_seasons_played").max()
        ).to_series()[0],
    ],
    lengthscale_range=[2, 8],
    cov_func="matern52",
)

# how
short_term_form, ax = pz.maxent(pz.Gamma(), lower=2, upper=5, mass=0.95)


dist_nu_new, ax = pz.maxent(pz.Gamma(), lower=20, upper=150, mass=0.95)

within_m, within_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[
        0,
        construct_games_played.select(pl.col("games_played").max()).to_series()[0],
    ],
    lengthscale_range=[4, 18],
    cov_func="matern52",
)


games_sd = (
    construct_games_played.group_by("game_id")
    .agg(pl.col("receiving_yards").mean().alias("avg"))
    .select(pl.col("avg").std().alias("sd"))
    .to_series()[0]
)

player_sds = (
    construct_games_played.group_by("receiver_full_name")
    .agg(pl.col("receiving_yards_c").mean().alias("avg"))
    .select(pl.col("avg").std().alias("sd"))
    .to_series()[0]
)


obs_sd = construct_games_played.select(pl.col("receiving_yards").std()).to_series()[0]


construct_games_played.select(pl.col("receiving_yards").mean())
dist_exp, ax = pz.maxent(pz.Exponential(), lower=20, upper=150, mass=0.95)


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
        "factor_num_data", factor_data.to_numpy(), dims=("obs_id", "factors_num")
    )

    player_id = pm.Data("player_id", player_idx, dims="obs_id")

    rec_obs = pm.Data(
        "rec_obs", construct_games_played["receiving_yards"].to_numpy(), dims="obs_id"
    )

    # effectively a difference of 20ish yards per player
    # so the difference between Chase and Jefferson is likely kind of small
    # but the difference between those two and say like alec pierce is kind of big
    sigma_player = pm.Exponential("player_sigma", 1 / 30)

    player_effects = pm.Normal("player_z", mu=50, sigma=sigma_player, dims="player")

    ls_games = pm.Gamma(
        "ls_games", alpha=short_term_form.alpha, beta=short_term_form.beta
    )

    # the issue seems to be just using games
    sigma_games = pm.Exponential("sigma_game", 1 / 50)

    cov_games = sigma_games**2 * pm.gp.cov.Matern52(input_dim=1, ls=ls_games)

    ls_season = pm.Gamma("ls_season", **seasons_gp_prior.params_dict)

    gp_within = pm.gp.HSGP(m=[16], c=1.5, cov_func=cov_games)
    f_within = gp_within.prior(
        "f_within", X=x_gamedays, hsgp_coeffs_dims="basis_coeffs_within", dims="gameday"
    )

    # since we are predicting at a game level we need this prior to be a little less crazy
    sigma_season = pm.Exponential("sigma_season", 1 / 30)

    cov_season = sigma_season**2 * pm.gp.cov.Matern52(1, ls=ls_season)

    gp_season = pm.gp.HSGP(m=[12], c=1.5, cov_func=cov_season)

    f_season = gp_season.prior(
        "f_season", X=x_season, hsgp_coeffs_dims="basis_coeffs_seasons", dims="seasons"
    )

    slope_num = pm.Normal("slope_num", sigma=0.5, dims="factors_num")

    # effectively the difference between a good coach and a bad coach is worth about 5ish receiving yards

    sigma_coach = pm.HalfNormal("sigma_coach", 1)

    off_coach_effect = pm.Normal(
        "slope_off", mu=0, sigma=sigma_coach, dims="off_play_caller"
    )

    alpha = pm.Deterministic(
        "alpha",
        player_effects[player_id]
        + off_coach_effect[off_id]
        + f_within[gameday_id]
        + f_season[seasons_id],
        dims="obs_id",
    )

    mu_player = pm.Deterministic(
        "mu_player", alpha + pm.math.dot(fct_data, slope_num), dims="obs_id"
    )
    nu = dist_nu_new.to_pymc(name="nu")

    # we still need a fair amount of variance to explain
    # so lets say like random stuff is worth about 30ish yards
    # between games
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1 / 30)

    rec_yards = pm.StudentT(
        "receiving_yards",
        nu=nu,
        mu=mu_player,
        sigma=sigma_obs,
        observed=rec_obs,
        dims="obs_id",
    )
    trace = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    p = pm.sample_prior_predictive()
    pm.sample_posterior_predictive(trace, receiving_mod_long, extend_inferencedata=True)


trace.sample_stats["diverging"].values.sum()

az.plot_ess(
    trace,
    kind="evolution",
    var_names=[RV.name for RV in receiving_mod_long.free_RVs if RV.size.eval() <= 3],
    grid=(5, 2),
    textsize=25,
)

az.plot_energy(trace)

az.plot_trace(
    trace,
    var_names=[
        "sigma_season",
        "sigma_game",
        "sigma_obs",
        "ls_season",
        "ls_games",
        "player_sigma",
    ],
)


az.plot_ppc(trace, num_pp_samples=100)
plt.xlim(-100, 250)

index = pd.MultiIndex.from_product(
    [unique_seasons, unique_games], names=["number_of_seasons_played", "gameday"]
)

unique_combos = pd.DataFrame(index=index).reset_index()

f_long_post = trace.posterior["f_season"]
f_games_post = trace.posterior["f_games"]


f_long_post_aligned = f_long_post.sel(
    seasons=unique_combos["number_of_seasons_played"].to_numpy()
).rename({"seasons": "timestamp"})

f_long_post_aligned["timestamp"] = unique_combos.index

f_games_post_aligned = f_games_post.sel(
    gameday=unique_combos["gameday"].to_numpy()
).rename({"gameday": "timestamp"})

some_samps = rng.choice(4000, size=20, replace=True)

_, axes = plt.subplot_mosaic("""AB""", figsize=(12, 7.5), layout="constrained")


axes["A"].plot(
    f_long_post.seasons,
    az.extract(f_long_post)["f_season"].isel(sample=some_samps),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
    label="random draws",
)
az.plot_hdi(
    x=f_long_post.seasons,
    y=f_long_post,
    hdi_prob=0.87,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9, "label": r"$87\%$ HDI"},
    ax=axes["A"],
    smooth=False,
)
axes["A"].plot(
    f_long_post.seasons,
    f_long_post.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
    label="Mean",
)
axes["A"].set(
    xlabel="Season", ylabel="Receiving Yards", title="Between Sesason Varition"
)

axes["B"].plot(
    f_games_post.gameday,
    az.extract(f_games_post)["f_games"].isel(sample=some_samps),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)


az.plot_hdi(
    x=f_games_post.gameday,
    y=f_games_post,
    hdi_prob=0.87,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["B"],
    smooth=False,
)

axes["B"].plot(
    f_games_post.gameday,
    f_games_post.mean(("chain", "draw")),
    color="#FBE64D",
    lw=2.5,
)
axes["B"].set(xlabel="Games", ylabel="Receiving Yards", title="Between Game Variation")
