import polars as pl
import polars.selectors as cs
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import preliz as pz
import pymc_bart as pmb
from arviz_plots import plot_convergence_dist

RANDOM_SEED = 5781
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

raw_df = pl.read_parquet("model_data/fantasy-points-data.parquet")

exclude_these = [
    "player_id",
    "season",
    "season_type",
    "team",
    "opponent_team",
    "week",
    "season",
    "headshot_url",
    "position",
    "sacks_suffered",
    "air_yards_share",
    "wopr",
    "pacr",
]

raw_df.columns

cleanish = (
    raw_df.select(pl.exclude(exclude_these + ["^.*name$"]))
    # Fill nulls with appropriate values (median for numeric columns)
    .with_columns(
        [
            pl.col(col).fill_null(0)
            for col in raw_df.select(pl.exclude(exclude_these + ["^.*name$"])).columns
            if raw_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]
    )
    .to_dummies(["position_group", "era"])
)


features = cleanish.select(
    pl.exclude(
        [
            "fantasy_points",
            "fantasy_points_ppr",
            "fumble_recovery_yards_own",
            "fumble_recovery_opp",
            "fumble_recovery_yards_opp",
            "fumble_recovery_tds",
            "rushing_first_downs",
            "passing_first_downs",
            "receiving_first_downs",
            "rushing_first_downs",
            "mock_date",
        ]
    )
).to_pandas()

cleanish

outcome = cleanish.select(pl.col("fantasy_points")).to_pandas()

y = outcome["fantasy_points"]

y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std


with pm.Model() as bart_simple:
    sigma = pm.HalfNormal("sigma", 0.5)
    # intercept = pm.Normal('intercept', mu = 0, sigma = 1)
    mu = pmb.BART("mu", features, y_scaled, m=20)

    Y_obs = pm.StudentT(
        "y_obs", nu=10, mu=mu, sigma=sigma, observed=y_scaled, shape=y_scaled.shape
    )
    idata = pm.sample(random_seed=RANDOM_SEED)


vi_results = pmb.compute_variable_importance(idata, mu, features)

pmb.plot_variable_importance(vi_results)

var_imp = (
    pl.DataFrame(
        {"features": vi_results["labels"], "importance": vi_results["r2_mean"]}
    )
    .with_columns(
        pl.col("importance").rank(method="dense", descending=True).alias("rank")
    )
    .sort("rank")
)

rolling_mean_cols = [
    "passing_yards",
    "completions",
    "passing_air_yards",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "fantasy_points",
    "target_share",
    "punt_return_yards",
    "kickoff_return_yards",
    "passing_cpoe",
    "completions",
    "carries",
    "targets",
]

cumulative_cols = [
    "rushing_epa",
    "passing_epa",
    "receiving_epa",
    "rushing_tds",
    "passing_tds",
    "passing_yards",
    "rushing_fumbles",
    "receiving_tds",
]

make_features = (
    raw_df.sort(["player_id", "season", "week"])
    .with_columns(
        *[
            pl.col(c)
            .rolling_mean(window_size=3)
            .over("player_id")
            .alias(f"{c}_rolling_mean")
            for c in rolling_mean_cols
        ]
    )
    .with_columns(
        *[
            pl.col(c)
            .cum_sum()
            .over(["player_id", "season"])
            .shift(1)
            .alias(f"{c}_cumulative")
            for c in cumulative_cols
        ]
    )
)

features = (
    make_features.select(
        cs.ends_with("cumulative"),
        cs.ends_with("rolling_mean"),
        pl.col("position_group"),
    )
    .to_dummies("position_group")
    .fill_null(0)
).to_pandas()


with pm.Model() as bart_dynamic:
    sigma = pm.HalfNormal("sigma", 0.5)
    # intercept = pm.Normal('intercept', mu = 0, sigma = 1)
    mu = pmb.BART("mu", features, y_scaled, m=50)

    Y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)


with bart_dynamic:
    idata = pm.sample_prior_predictive()


az.plot_ppc(idata, group="prior", observed=True)

with bart_dynamic:
    idata.extend(pm.sample(random_seed=RANDOM_SEED))


with bart_dynamic:
    idata.extend(pm.sample_posterior_predictive(idata))


az.plot_ppc(idata)

plt.xlim(y.min(), y.max())


vi_results = pmb.compute_variable_importance(idata, mu, features)


pmb.plot_variable_importance(vi_results=vi_results)

var_imp = (
    pl.DataFrame(
        {"features": vi_results["labels"], "importance": vi_results["r2_mean"]}
    )
    .with_columns(
        pl.col("importance").rank(method="dense", descending=True).alias("rank")
    )
    .sort("rank")
)


## lets use the asymetric laplace to predict the median
plt.close("all")


pz.maxent(pz.AsymmetricLaplace(), lower=y.min(), upper=y.max())

plt.close("all")

pz.Beta(2, 2).plot_pdf()


with pm.Model() as bart_laplace:
    mu = pmb.BART("mu", features.values, y_scaled, m=20)

    b = pm.HalfNormal("b", 0.5)

    q = pm.Beta("q", alpha=2, beta=2)

    Y_obs = pm.AsymmetricLaplace("y_obs", mu=mu, b=b, q=q, observed=y_scaled)


with bart_laplace:
    idata = pm.sample_prior_predictive()


az.plot_ppc(idata, group="prior")


with bart_laplace:
    idata.extend(pm.sample(random_seed=RANDOM_SEED))


with bart_laplace:
    idata.extend(pm.sample_posterior_predictive(idata))


az.plot_ppc(idata)

plt.xlim(y_scaled.min(), y_scaled.max())

plot_convergence_dist(idata)


var_imp = pmb.compute_variable_importance(idata, mu, features)
pmb.plot_variable_importance(var_imp)
