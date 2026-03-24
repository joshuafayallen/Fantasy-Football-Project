import polars as pl
import polars.selectors as cs
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import pymc_bart as pmb
from arviz_plots import plot_convergence_dist

RANDOM_SEED = 5781
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

raw_df = pl.read_parquet("model_data/fantasy-points-data.parquet")

outcome = raw_df.select(pl.col("fantasy_points")).to_pandas()

y = outcome["fantasy_points"]

y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

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
