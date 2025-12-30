import polars as pl
import pandas as pd
import pymc as pm
import preliz as pz
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import nflreadpy as nfl
import seaborn as sns
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=18"


seed = sum(map(ord, "receivingyardsproject"))
rng = np.random.default_rng(seed)

full_pass_data = pl.scan_parquet("processed_data/processed_passers_*.parquet").collect()

full_scores = pl.read_csv(
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

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
        "total_line",
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
    # "no_huddle",
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
        .cum_sum()
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
    .filter((pl.col("receiver_position").is_in(["RB", "TE", "WR"])))
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
        .alias("total_pass_epa_game"),
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
        pl.when(pl.col("season") >= 2018).then(1).otherwise(0).alias("era"),
    )
)

agg_full_seasons.filter(pl.col("receiver_full_name") == "Marshall Faulk")

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
        pl.col("total_pass_epa_game")
        .cum_sum()
        .over(["posteam", "season"])
        .shift(1)
        .alias("cumulative_off_epa"),
        pl.col("game_id")
        .cum_count()
        .over(["posteam", "season"])
        .alias("total_games_played_offense"),
        pl.col("air_yards")
        .cum_sum()
        .over(["posteam", "season"])
        .shift(1)
        .alias("cumulative_air_yards_game"),
        pl.col("targeted")
        .cum_sum()
        .over(["posteam", "season"])
        .shift(1)
        .alias("cumulative_targets"),
    )
    .with_columns(
        (pl.col("cumulative_air_yards_game") / pl.col("cumulative_targets")).alias(
            "air_yards_per_pass_attempt"
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
        (pl.col("receiving_yards") - pl.col("receiving_yards").mean()).alias(
            "receiving_yards_c"
        ),
        pl.col("targeted")
        .sum()
        .over(["receiver_full_name", "game_id", "season"])
        .alias("targets_game"),
        pl.col("targeted")
        .sum()
        .over(["receiver_full_name", "season"])
        .alias("targets_season"),
        pl.when(pl.col("receiving_yards") <= 0)
        .then(0)
        .otherwise(1)
        .alias("rec_yards_binary"),
    )
    .sort(["receiver_full_name", "season", "game_id"])
    .fill_null(0)
    .filter(
        pl.col("receiving_yards") <= 336
    )  ## just get rid of any players abocve the single game receiving yards record
)


factors_numeric = [
    "player_rest_diff",
    "def_epa_diff",
    "wind",
    "temp",
    "total_line",
    "air_yards_per_pass_attempt",
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

unique_players = cumulative_stats_pd["receiver_full_name"].sort_values().unique()

cumulative_stats.group_by(["rec_tds"]).agg(pl.len())


player_idx = pd.Categorical(
    cumulative_stats_pd["receiver_full_name"], categories=unique_players
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

sns.kdeplot(cumulative_stats, x="receiving_yards")


_, axes = plt.subplot_mosaic(
    """
    AB
    CD
    """,
    layout="constrained",
)

axes["A"].hist(
    data=cumulative_stats.filter(pl.col("receiver_full_name") == "Ja'Marr Chase"),
    x="receiving_yards",
)


axes["A"].set(title="Ja'Marr Chase Receiving Distribution")

axes["B"].hist(
    data=cumulative_stats.filter(pl.col("receiver_full_name") == "George Kittle"),
    x="receiving_yards",
)

axes["B"].set(title="George Kittle Receiveing Distribution")


axes["C"].hist(
    data=cumulative_stats.filter(pl.col("receiver_full_name") == "De'Von Achane"),
    x="receiving_yards",
)

axes["C"].set(title="De'Von Achane Receiving Yards")

axes["D"].hist(
    data=cumulative_stats.filter(pl.col("receiver_full_name") == "Josh Downs"),
    x="receiving_yards",
)

axes["D"].set(title="Josh Downs Receiving Yards")

## for the most part elite to good pass catchers tend to be around 50 yards p/g
# lets look at some all timrs

_, axes = plt.subplot_mosaic(
    """
    AB
    CD
    """,
    layout="constrained",
)

#  Larry is second all time so
# we are just going to go with larr
axes["A"].hist(
    data=cumulative_stats.filter(pl.col("receiver_full_name") == "Larry Fitzgerald"),
    x="receiving_yards",
)


axes["A"].set(title="Larry Fitzgerald \n Receiving Distribution")

axes["B"].hist(
    data=cumulative_stats.filter(pl.col("receiver_full_name") == "Tony Gonzalez"),
    x="receiving_yards",
)

axes["B"].set(title="Tony Gonzalez Receiveing Distribution")


axes["C"].hist(
    data=cumulative_stats.filter(pl.col("receiver_full_name") == "Marshall Faulk"),
    x="receiving_yards",
)

axes["C"].set(title="Marshall Faulk Receiving Yards")

axes["D"].hist(
    data=cumulative_stats.filter(pl.col("receiver_full_name") == "Julian Edelman"),
    x="receiving_yards",
)

axes["D"].set(title="Julian Edelman Receiving Yards")


## For like the elite of the elite most of them have kind of a uniform distribution around 50-100 receiving yard games with the odd stinker

just_larry = cumulative_stats.filter(pl.col("receiver_full_name") == "Larry Fitzgerald")

g = sns.FacetGrid(just_larry, col="season", col_wrap=2).tight_layout()

g.map_dataframe(sns.barplot, x="games_played", y="receiving_yards")

for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)


## honestly 10-20 ish yard swings are not all that wild.
## even for a wide receiver as consistent as larry

plt.close("all")


just_calvin = cumulative_stats.filter(pl.col("receiver_full_name") == "Calvin Johnson")

g = sns.FacetGrid(just_calvin, col="season", col_wrap=2).tight_layout()

g.map_dataframe(sns.barplot, x="games_played", y="receiving_yards")

for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)


##  reggie wayne will be an interesting test case since he was on a consitently good team

just_reggie = cumulative_stats.filter(pl.col("receiver_full_name") == "Reggie Wayne")

g = sns.FacetGrid(just_reggie, col="season", col_wrap=2).tight_layout()

g.map_dataframe(sns.barplot, x="games_played", y="receiving_yards")

for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)

plt.close("all")

yards_dist, _ = pz.maxent(pz.HalfNormal(), 0, 150, mass=0.90)


# how we set the sigma was thinking about the
# probability that a player had a 2 touchdown game
# we did it like this

# this gives us a lambda value of about 1.93

# We could try to do a similar thing but since it is continous
# each of the probabilites are super small
# instead I think we are just going to go with a pretty permisive prior

plt.close("all")
sigma_gp, _ = pz.maxent(pz.Exponential(), 0, 10, mass=0.95)

## we will just work with a defaultish prior for
# Most of the mass is going to be between 0-100
# this is a little bit more generous the general vagueish priors
cumulative_stats["receiving_yards"].mean()

plt.close("all")
nu_prior, _ = pz.maxent(pz.Gamma(), lower=0, upper=15, mass=0.90)

obs_sigma, _ = pz.maxent(pz.HalfNormal(), 0, 15, mass=0.90)


with pm.Model(coords=coords) as receiving_yards:
    factor_data = pm.Data("factor_data", factors_numeric_sdz, dims=("obs_id", "factor"))

    games_id = pm.Data("games_id", games_idx, dims="obs_id")
    player_id = pm.Data("player_id", player_idx, dims="obs_id")
    season_id = pm.Data("season_id", seasons_idx, dims="obs_id")

    receiving_obs = pm.Data(
        "receiving_obs", cumulative_stats["receiving_yards"].to_numpy(), dims="obs_id"
    )

    x_gamedays = pm.Data("x_gamedays", unique_games, dims="gameday")[:, None]
    x_seasons = pm.Data("x_seasons", unique_seasons, dims="seasons")[:, None]

    player_sigma = yards_dist.to_pymc("player_sigma")
    player_means = pm.Normal("player_means", mu=0, sigma=10, dims="player")

    player_effect = pm.Deterministic(
        "player_effect",
        pm.Normal(
            "player_effects_raw", mu=player_means, sigma=player_sigma, dims="player"
        ),
        dims="player",
    )

    gps_sigma = pm.Exponential("gps_sigma", lam=sigma_gp.lam, dims="time_scale")

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
    # set this at 2.5 intially will tune in after the prior predictive checks
    # since we have lots of players with negative receiving yards we are just going to put an even tighter prior on the global parameters
    slope = pm.Normal("slope", sigma=0.05, dims="factors")

    mu = pm.Deterministic("mu", alpha + pm.math.dot(factor_data, slope), dims="obs_id")

    nu = nu_prior.to_pymc("nu")

    # setting this pretty wide
    observed_sigma = obs_sigma.to_pymc("observed_sigma")

    pm.Normal(
        "receiving_yards",
        # nu = nu,
        mu=mu,
        sigma=observed_sigma,
        observed=receiving_obs,
        dims="obs_id",
    )


with receiving_yards:
    idata = pm.sample_prior_predictive()

## this doesnt look all that great
plt.close("all")
az.plot_ppc(idata, group="prior", observed=True)

plt.xlim(-10, 200)
## lets use the real data to seee how funky this looks

with receiving_yards:
    idata.extend(pm.sample(nuts_sampler="numpyro", random_seed=rng, target_accept=0.99))


with receiving_yards:
    idata.extend(
        pm.sample_posterior_predictive(idata, compile_kwargs={"mode": "NUMBA"})
    )

## hmm I think the problem is that there is just a weirdish relationshp
# in yards which isnt perfectly linear
## we also have a good bit of mass that isnt being captured in the tails

az.plot_ppc(idata)

plt.close("all")


## so the problem is the huge spike between 0 and 30 which makes it pretty hard
# to fit anything we are going to go with a hurdle model
# so we have to model the zero process
# the most offending group for negative receiving yards is receiver
#
cumulative_stats.filter(pl.col("receiving_yards") <= 0).group_by(
    "receiver_position"
).agg(pl.len())

hurdle_sigma, _ = pz.maxent(pz.HalfNormal(), 0, 2, mass=0.90)

with pm.Model(coords=coords) as receiving_yards_hurdle:
    # Data containers
    factor_data = pm.Data(
        "factor_data", factors_numeric_sdz, dims=("obs_id", "factors")
    )
    games_id = pm.Data("games_id", games_idx, dims="obs_id")
    player_id = pm.Data("player_id", player_idx, dims="obs_id")
    season_id = pm.Data("season_id", seasons_idx, dims="obs_id")
    receiving_obs = pm.Data(
        "receiving_obs", cumulative_stats["receiving_yards"].to_numpy(), dims="obs_id"
    )

    # Binary indicator: 1 if any yards, 0 if zero yards
    has_yards = pm.Data(
        "has_yards",
        (cumulative_stats["receiving_yards"] > 0).to_numpy().astype(int),
        dims="obs_id",
    )

    # Only use positive yards for the positive component
    positive_yards = cumulative_stats.filter(pl.col("receiving_yards") > 0)[
        "receiving_yards"
    ].to_numpy()
    positive_yards_idx = np.where(cumulative_stats["receiving_yards"].to_numpy() > 0)[0]

    x_gamedays = pm.Data("x_gamedays", unique_games, dims="gameday")[:, None]
    x_seasons = pm.Data("x_seasons", unique_seasons, dims="seasons")[:, None]

    # ========== HURDLE COMPONENT (Binary: Any Yards vs Zero) ==========
    hurdle_player_sigma = hurdle_sigma.to_pymc("hurdle_player_sigma")
    hurdle_player_means = pm.Normal("hurdle_player_means", mu=0, sigma=1, dims="player")
    hurdle_player_effect = pm.Deterministic(
        "hurdle_player_effect",
        pm.Normal(
            "hurdle_player_effects_raw",
            mu=hurdle_player_means,
            sigma=hurdle_player_sigma,
            dims="player",
        ),
        dims="player",
    )

    # GP components for hurdle
    hurdle_gps_sigma = pm.Exponential(
        "hurdle_gps_sigma", lam=sigma_gp.lam, dims="time_scale"
    )
    hurdle_ls = pm.InverseGamma(
        "hurdle_ls",
        alpha=np.array([short_term_form.alpha, seasons_gp_prior.alpha]),
        beta=np.array([short_term_form.beta, seasons_gp_prior.beta]),
        dims="time_scale",
    )

    hurdle_cov_games = hurdle_gps_sigma[0] ** 2 * pm.gp.cov.Matern52(
        input_dim=1, ls=hurdle_ls[0]
    )
    hurdle_cov_seasons = hurdle_gps_sigma[1] ** 2 * pm.gp.cov.Matern52(
        input_dim=1, ls=hurdle_ls[1]
    )

    hurdle_gp_games = pm.gp.HSGP(m=[within_m], c=within_c, cov_func=hurdle_cov_games)
    hurdle_gp_season = pm.gp.HSGP(
        m=[seasons_m], c=seasons_c, cov_func=hurdle_cov_seasons
    )

    hurdle_basis_vectors_game, hurdle_sqrt_psd_game = hurdle_gp_games.prior_linearized(
        X=x_gamedays
    )
    hurdle_basis_coeffs_games = pm.Normal(
        "hurdle_basis_coeffs_games", shape=hurdle_gp_games.n_basis_vectors
    )
    hurdle_f_games = pm.Deterministic(
        "hurdle_f_games",
        hurdle_basis_vectors_game @ (hurdle_basis_coeffs_games * hurdle_sqrt_psd_game),
        dims="gameday",
    )

    hurdle_basis_vectors_season, hurdle_sqrt_psd_season = (
        hurdle_gp_season.prior_linearized(X=x_seasons)
    )
    hurdle_basis_coeffs_season = pm.Normal(
        "hurdle_basis_coeffs_season", shape=hurdle_gp_season.n_basis_vectors
    )
    hurdle_f_season = pm.Deterministic(
        "hurdle_f_season",
        hurdle_basis_vectors_season
        @ (hurdle_basis_coeffs_season * hurdle_sqrt_psd_season),
        dims="seasons",
    )

    hurdle_alpha = pm.Deterministic(
        "hurdle_alpha",
        hurdle_player_effect[player_id]
        + hurdle_f_season[season_id]
        + hurdle_f_games[games_id],
        dims="obs_id",
    )

    hurdle_slope = pm.Normal("hurdle_slope", sigma=0.5, dims="factors")
    hurdle_eta = pm.Deterministic(
        "hurdle_eta",
        hurdle_alpha + pm.math.dot(factor_data, hurdle_slope),
        dims="obs_id",
    )

    p_hurdle = pm.Deterministic("p_hurdle", pm.math.invlogit(hurdle_eta), dims="obs_id")

    player_sigma = pm.HalfNormal("player_sigma", 0.75)
    player_means = pm.Normal("player_means", mu=0, sigma=3, dims="player")
    player_effect = pm.Deterministic(
        "player_effect",
        pm.Normal(
            "player_effects_raw", mu=player_means, sigma=player_sigma, dims="player"
        ),
        dims="player",
    )

    gps_sigma = pm.Exponential("gps_sigma", lam=sigma_gp.lam, dims="time_scale")
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

    intercept = pm.Normal("intercept", mu=3.5, sigma=1)

    alpha = pm.Deterministic(
        "alpha",
        player_effect[player_id] + f_season[season_id] + f_games[games_id],
        dims="obs_id",
    )

    slope = pm.Normal("slope", sigma=0.2, dims="factors")
    mu = pm.Deterministic(
        "mu", intercept + alpha + pm.math.dot(factor_data, slope), dims="obs_id"
    )

    observed_sigma = obs_sigma.to_pymc("observed_sigma")

    pm.Bernoulli("has_yards_obs", p=p_hurdle, observed=has_yards, dims="obs_id")

    positive_mu = mu[positive_yards_idx]
    pm.TruncatedNormal(
        "positive_yards_obs",
        mu=positive_mu,
        sigma=observed_sigma,
        lower=0,
        observed=positive_yards,
    )


with receiving_yards_hurdle:
    idata_hurdle = pm.sample_prior_predictive()


az.plot_ppc(idata_hurdle, group="prior", observed=True)


with receiving_yards_hurdle:
    idata_hurdle.extend(pm.sample(nuts_sampler="nutpie", random_seed=rng))


with receiving_yards_hurdle:
    idata_hurdle.extend(pm.sample_posterior_predictive(idata))


idata_hurdle.sample_stats["diverging"].sum().data


az.plot_ppc(idata_hurdle)


with pm.Model(coords=coords) as receiving_yards_log:
    factor_data = pm.Data(
        "factor_data", factors_numeric_sdz, dims=("obs_id", "factors")
    )
    games_id = pm.Data("games_id", games_idx, dims="obs_id")
    player_id = pm.Data("player_id", player_idx, dims="obs_id")
    season_id = pm.Data("season_id", seasons_idx, dims="obs_id")
    receiving_obs = pm.Data(
        "receiving_obs", cumulative_stats["receiving_yards"].to_numpy(), dims="obs_id"
    )
    x_gamedays = pm.Data("x_gamedays", unique_games, dims="gameday")[:, None]
    x_seasons = pm.Data("x_seasons", unique_seasons, dims="seasons")[:, None]

    hurdle_player_sigma = yards_dist.to_pymc("hurdle_player_sigma")
    hurdle_player_means = pm.Normal("hurdle_player_means", mu=0, sigma=1, dims="player")
    hurdle_player_effect = pm.Deterministic(
        "hurdle_player_effect",
        pm.Normal(
            "hurdle_player_effects_raw",
            mu=hurdle_player_means,
            sigma=hurdle_player_sigma,
            dims="player",
        ),
        dims="player",
    )

    # GP for hurdle (Games)
    hurdle_gps_sigma = pm.Exponential("hurdle_gps_sigma", lam=0.75, dims="time_scale")
    hurdle_ls = pm.InverseGamma(
        "hurdle_ls",
        alpha=np.array([short_term_form.alpha, seasons_gp_prior.alpha]),
        beta=np.array([short_term_form.beta, seasons_gp_prior.beta]),
        dims="time_scale",
    )
    hurdle_cov_games = hurdle_gps_sigma[0] ** 2 * pm.gp.cov.Matern52(
        input_dim=1, ls=hurdle_ls[0]
    )
    hurdle_gp_games = pm.gp.HSGP(m=[within_m], c=within_c, cov_func=hurdle_cov_games)
    hurdle_basis_vectors_game, hurdle_sqrt_psd_game = hurdle_gp_games.prior_linearized(
        X=x_gamedays
    )
    hurdle_basis_coeffs_games = pm.Normal(
        "hurdle_basis_coeffs_games", shape=hurdle_gp_games.n_basis_vectors
    )
    hurdle_f_games = pm.Deterministic(
        "hurdle_f_games",
        hurdle_basis_vectors_game @ (hurdle_basis_coeffs_games * hurdle_sqrt_psd_game),
        dims="gameday",
    )

    # GP for hurdle (Seasons)
    hurdle_cov_seasons = hurdle_gps_sigma[1] ** 2 * pm.gp.cov.Matern52(
        input_dim=1, ls=hurdle_ls[1]
    )
    hurdle_gp_season = pm.gp.HSGP(
        m=[seasons_m], c=seasons_c, cov_func=hurdle_cov_seasons
    )
    hurdle_basis_vectors_season, hurdle_sqrt_psd_season = (
        hurdle_gp_season.prior_linearized(X=x_seasons)
    )
    hurdle_basis_coeffs_season = pm.Normal(
        "hurdle_basis_coeffs_season", shape=hurdle_gp_season.n_basis_vectors
    )

    hurdle_f_season = pm.Deterministic(
        "hurdle_f_season",
        hurdle_basis_vectors_season
        @ (hurdle_basis_coeffs_season * hurdle_sqrt_psd_season),
        dims="seasons",
    )

    hurdle_alpha = pm.Deterministic(
        "hurdle_alpha",
        hurdle_player_effect[player_id]
        + hurdle_f_season[season_id]
        + hurdle_f_games[games_id],
        dims="obs_id",
    )

    hurdle_slope = pm.Normal("hurdle_slope", sigma=0.01, dims="factors")

    # This is the linear predictor for the HURDLE (phi in HurdleLognormal)
    hurdle_eta = pm.Deterministic(
        "hurdle_eta",
        hurdle_alpha + pm.math.dot(factor_data, hurdle_slope),
        dims="obs_id",
    )
    player_sigma = pm.HalfNormal("player_sigma", 0.75)
    player_means = pm.Normal("player_means", mu=1, sigma=1, dims="player")
    player_effect = pm.Deterministic(
        "player_effect",
        pm.Normal(
            "player_effects_raw", mu=player_means, sigma=player_sigma, dims="player"
        ),
        dims="player",
    )

    # GP components for positive yards (Games and Seasons)
    gps_sigma = pm.Exponential("gps_sigma", lam=2, dims="time_scale")
    ls = pm.InverseGamma(
        "ls",
        alpha=np.array([short_term_form.alpha, seasons_gp_prior.alpha]),
        beta=np.array([short_term_form.beta, seasons_gp_prior.beta]),
        dims="time_scale",
    )
    cov_games = gps_sigma[0] ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ls[0])
    gp_games = pm.gp.HSGP(m=[within_m], c=within_c, cov_func=cov_games)
    basis_vectors_game, sqrt_psd_game = gp_games.prior_linearized(X=x_gamedays)
    basis_coeffs_games = pm.Normal("basis_coeffs_games", shape=gp_games.n_basis_vectors)
    f_games = pm.Deterministic(
        "f_games",
        basis_vectors_game @ (basis_coeffs_games * sqrt_psd_game),
        dims="gameday",
    )

    cov_seasons = gps_sigma[1] ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ls[1])
    gp_season = pm.gp.HSGP(m=[seasons_m], c=seasons_c, cov_func=cov_seasons)
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

    slope = pm.Normal("slope", sigma=0.01, dims="factors")

    log_mu = pm.Deterministic(
        "log_mu", alpha + pm.math.dot(factor_data, slope), dims="obs_id"
    )

    log_sigma = pm.HalfNormal("log_sigma", sigma=0.05)

    pm.HurdleLogNormal(
        "receiving_yards_obs",
        psi=pm.math.invlogit(hurdle_eta),
        mu=log_mu,
        sigma=log_sigma,
        observed=receiving_obs,
        dims="obs_id",
    )


with receiving_yards_log:
    idata = pm.sample_prior_predictive()


az.plot_ppc(idata, observed=True, group="prior")


check = cumulative_stats.filter(pl.col("receiving_yards") <= 0)


check.height / cumulative_stats.height


plt.close("all")

pz.maxent(pz.AsymmetricLaplace(), lower=-10, upper=250)

pz.AsymmetricLaplace(kappa=0.5, mu=30, b=15).plot_pdf()
# pz.AsymmetricLaplace(kappa = 0.1, mu = 30 , b =15).plot_pdf()

# so the kappa and the b are going the key component to tune
# the

plt.close("all")
pz.AsymmetricLaplace(q=0.22, mu=30, b=15).plot_pdf()

plt.close("all")
# so this works
pz.AsymmetricLaplace(q=0.22, mu=30, b=10).plot_pdf()

b_val, _ = pz.maxent(pz.HalfNormal(), 0, 10)

yards_dist, _ = pz.maxent(pz.HalfNormal(), 0, 5, mass=0.90)


with pm.Model(coords=coords) as quantile_receiving:
    factor_data = pm.Data(
        "factor_data", factors_numeric_sdz, dims=("obs_id", "factors")
    )

    games_id = pm.Data("games_id", games_idx, dims="obs_id")

    player_id = pm.Data("player_id", player_idx, dims="obs_id")

    season_id = pm.Data("season_id", seasons_idx, dims="obs_id")

    receiving_obs = pm.Data(
        "receiving_obs", cumulative_stats["receiving_yards"].to_numpy(), dims="obs_id"
    )

    x_gamedays = pm.Data("x_gamedays", unique_games, dims="gameday")[:, None]

    x_seasons = pm.Data("x_seasons", unique_seasons, dims="seasons")[:, None]

    player_sigma = yards_dist.to_pymc("player_sigma")

    player_means = pm.Normal("player_means", mu=10, sigma=5, dims="player")

    player_effect = pm.Deterministic(
        "player_effect",
        pm.Normal(
            "player_effects_raw", mu=player_means, sigma=player_sigma, dims="player"
        ),
        dims="player",
    )

    # GP components for positive yards (Games and Seasons)
    gps_sigma = pm.Exponential("gps_sigma", sigma_gp.lam, dims="time_scale")

    ls = pm.InverseGamma(
        "ls",
        alpha=np.array([short_term_form.alpha, seasons_gp_prior.alpha]),
        beta=np.array([short_term_form.beta, seasons_gp_prior.beta]),
        dims="time_scale",
    )

    cov_games = gps_sigma[0] ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ls[0])

    gp_games = pm.gp.HSGP(m=[within_m], c=within_c, cov_func=cov_games)

    basis_vectors_game, sqrt_psd_game = gp_games.prior_linearized(X=x_gamedays)

    basis_coeffs_games = pm.Normal("basis_coeffs_games", shape=gp_games.n_basis_vectors)

    f_games = pm.Deterministic(
        "f_games",
        basis_vectors_game @ (basis_coeffs_games * sqrt_psd_game),
        dims="gameday",
    )

    cov_seasons = gps_sigma[1] ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ls[1])
    gp_season = pm.gp.HSGP(m=[seasons_m], c=seasons_c, cov_func=cov_seasons)
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

    slope = pm.Normal("slope", sigma=1.0, dims="factors")

    mu = pm.Deterministic("mu", alpha + pm.math.dot(factor_data, slope), dims="obs_id")

    tau = pm.Data("tau", 0.5)

    scale_base = b_val.to_pymc("scale_param")
    scale = pm.Deterministic("scale", scale_base * (1 + 2 * pm.math.abs(tau - 0.5)))

    obs = pm.AsymmetricLaplace(
        "obs",
        mu=mu,
        b=scale,
        q=tau,  # set at median
        observed=receiving_obs,
        dims="obs_id",
    )


with quantile_receiving:
    idata_quant = pm.sample_prior_predictive()

plt.close("all")

az.plot_ppc(idata_quant, group="prior", observed=True)


with quantile_receiving:
    idata_quant.extend(pm.sample(nuts_sampler="numpyro", random_seed=rng))


with quantile_receiving:
    idata_quant.extend(
        pm.sample_posterior_predictive(idata_quant, compile_kwargs={"mode": "NUMBA"})
    )

az.ess(idata_quant).min().to_pandas().sort_values().round()


idata_quant.sample_stats["diverging"].sum().data

az.plot_ppc(idata_quant)


players_interested_in = [
    "Ja'Marr Chase",
    "Justin Jefferson",
    "Randy Moss",
    "Calvin Johnson",
    "Marshall Faulk",
    "Kendrick Bourne",
    "Christian McCaffrey",
]

plot_small = idata_quant.posterior["player_effect"].sel(player=players_interested_in)


az.plot_forest(plot_small, var_names=["player_effect"], combined=True)

with receiving_yards:
    idata.extend(pm.compute_log_likelihood(idata))


with quantile_receiving:
    idata_quant.extend(pm.compute_log_likelihood(idata_quant))


mods = ["quantile regression", "ols"]

mods_dict = dict(zip(mods, [idata_quant, idata]))

az.compare(mods_dict)

loo_result = az.loo(idata, pointwise=True)
k_values = loo_result.pareto_k.values
