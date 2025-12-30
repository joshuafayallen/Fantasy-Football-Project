import polars as pl
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
        ]
    )
).to_pandas()

outcome = cleanish.select(pl.col("fantasy_points")).to_pandas()

y = outcome["fantasy_points"]

y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

with pm.Model() as bart_simple:
    sigma = pm.HalfNormal("sigma", y.std())
    mu = pmb.BART("mu", features, y, m=100)
    Y = pm.Normal("y", mu=mu, sigma=sigma, observed=y, shape=y.shape)


with bart_simple:
    idata = pm.sample_prior_predictive()


az.plot_ppc(idata, group="prior", observed=True)

plt.xlim(y.min(), y.max())


with bart_simple:
    idata.extend(pm.sample(random_seed=RANDOM_SEED))


plot_convergence_dist(idata)

y.mean()

with bart_simple:
    idata.extend(pm.sample_posterior_predictive(idata))


az.plot_ppc(idata)
