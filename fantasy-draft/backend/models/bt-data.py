import arviz as az
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xarray as xr
import pytensor.tensor as pt 
import numpy as np
import polars as pl
import polars.selectors as cs
import preliz as pz
import pymc as pm
import nutpie
from great_tables import GT


raw_fantasy_data_pos = (
    pl.scan_parquet("processed_data")
    .collect()
    .sort("avg_expert_ranking")
    .with_columns(
        pl.col("birth_date").dt.year().alias("birth_year"),
        pl.when(pl.col("year").is_null())
        .then(2025)
        .otherwise(pl.col("year"))
        .alias("year"),
    )
    .with_columns(
        (pl.col('year') - pl.col('birth_year')).alias('age')
    ).with_columns(
        pl.col('average_pick_yahoo', 'espn_adp').round(mode = 'half_to_even')
    ))
filled_data_set = raw_fantasy_data_pos.fill_nan(0).fill_null(0)


def standardize_columns(cols, group):
    return [
        (
            (pl.col(col) - pl.col(col).mean().over(group))
            / pl.col(col).std().over(group)
        ).alias(f"{col}_zstd")
        for col in cols
    ]


def center_columns(cols, group):
    return [
        (pl.col(col) - pl.col(col).mean().over(group)).alias(f"{col}_centered")
        for col in cols
    ]


### we have a ton of covariates per position so lets
### narrow this down a bit
## we probably want to maximize on the efficiency stats
## yards_per_attempt, yards_per_reception, target_share, red_zone_tgt_share
### passing_yards_per_attempt
## TD production is also important
## so we should start off by centering it


sd_cols = (
    filled_data_set.with_columns(
        *standardize_columns(
            [
                "avg_expert_ranking",
                "yards_per_attempt",
                "yards_per_reception",
                "target_share",
                "red_zone_targets",
                "passing_yards_per_attempt",
                "rushing_yards",
                "receiving_yards",
            ],
            group="pos",
        )
    )
    .fill_nan(None)
    .with_columns(
        *center_columns(["receiving_tds", "passing_tds", "rushing_tds"], group="pos"),
        pl.col("yahoo_rank").str.to_integer(),
    )
)


analysis_data = sd_cols.select(
    pl.col("player", "pos", 'age', 'id', 'espn_adp', 'average_pick_yahoo'), cs.matches("rank|centered|zstd")
)


# just pre sorting
bt_data = (
    analysis_data.unique("player")
    .filter((pl.col("yahoo_rank").is_not_null()) & (pl.col("espn_rank") > 0) & (pl.col('avg_expert_ranking') <= 250))
    .sort("avg_expert_ranking")
    .rename({"id":'pid'})
)




bt_dat_long = bt_data.filter((pl.col('yahoo_rank') <= 550) & (pl.col('espn_rank') <= 550)).unpivot(on = ['yahoo_rank', 'espn_rank'], index = ['player', 'pos', 'age', 'pid'], variable_name='ranking_site', value_name='rank').filter(
    pl.col('age') > 0
)

site_encoder = LabelEncoder()

pos_encoder = LabelEncoder()

pos_encoder.fit(bt_dat_long.select(pl.col('pos').unique()).to_series().to_list())


site_encoder.fit(bt_dat_long.select(pl.col("ranking_site").unique()).to_series().to_list())


bt_dat_long_filtered = bt_dat_long.with_columns(
    pl.Series('site_id', site_encoder.transform(bt_dat_long['ranking_site'])),
    pl.Series('pos_id', pos_encoder.transform(bt_dat_long['pos']))
)

n_players = len(bt_dat_long_filtered.select(pl.col('pid').unique()).to_series().to_list())

n_sites = len(site_encoder.classes_)

rankings_dat = bt_dat_long_filtered.with_columns(
    pl.col('pid').str.to_integer()
).group_by('ranking_site').agg(
    pl.col('pid').sort_by('rank')
)

unique_player_ids = sorted(bt_dat_long_filtered.get_column('pid').unique().to_list())
player_id_to_idx = {pid: idx for idx, pid in enumerate(unique_player_ids)}
n_players = len(unique_player_ids)
n_sites = len(site_encoder.classes_)
n_poss = len(pos_encoder.classes_)

rankings_dat = (bt_dat_long_filtered
    .group_by('ranking_site')
    .agg(pl.col('pid').sort_by('rank'))
)

player_positions_df = (bt_dat_long_filtered
        .group_by(['pid', 'pos_id'])
        .len()
        .group_by('pid')
        .agg([
            pl.col('pos_id').sort_by('len', descending=True).first().alias('primary_position'),
            pl.col('len').max().alias('max_count')
        ])
    )

position_counts = bt_dat_long_filtered.group_by('pos_id').len().sort('pos_id')

n_players = len(unique_player_ids)
n_positions = len(bt_dat_long_filtered.select(pl.col('pos').unique()).to_series().to_list())

rank_indices = []
for ranking in rankings_dat['pid'].to_list():
    rank_indices.append([player_id_to_idx[pid] for pid in ranking])

player_positions = np.full(n_players, -1, dtype=int)
    
    # Fill in known positions
for row in player_positions_df.iter_rows(named=True):

    player_id = row['pid']
    if player_id in player_id_to_idx:

        player_idx = player_id_to_idx[player_id]
        player_positions[player_idx] = row['primary_position']


    # Fill in known positions
for row in player_positions_df.iter_rows(named=True):

    player_id = row['pid']
    if player_id in player_id_to_idx:

        player_idx = player_id_to_idx[player_id]
        player_positions[player_idx] = row['primary_position']

player_positions



pz.Exponential(1/20).plot_pdf()
pz.Exponential(1/10).plot_pdf()
pz.Exponential(1/15).plot_pdf()
plt.xlim(0, 100)

with pm.Model() as pl_mod_batched:
    nu_raw = pm.Exponential('nu_raw', 1/15)
    global_mean = pm.StudentT("global_mean", mu = 0, sigma = 1, nu = nu_raw)
    global_sigma = pm.HalfStudentT('global_sigma', nu = nu_raw , sigma = 1)


    position_effects = pm.Normal('position_effects', 0, 1, shape = n_positions)
    
    position_means = position_effects[player_positions]

    player_abilities_raw = pm.Normal('player_abilities_raw', mu = global_mean + position_means, sigma = global_sigma, shape = n_players)

    player_abilities = pm.Deterministic(
        'player_abilities', player_abilities_raw - pt.mean(player_abilities_raw)
    )

    total_logp = 0
    for ranking in rank_indices:
        if len(ranking) >= 2:
            ranking_tensor = pt.constant(np.array(ranking, dtype=np.int32))
            ranking_abilities = player_abilities[ranking_tensor]
            numerator = ranking_abilities[pos]
            denominator = pt.logsumexp(ranking_abilities[pos:])
            total_logp += numerator - denominator
    
    
    
    pm.Potential('likelihood', total_logp)


compiled_mod = nutpie.compile_pymc_model(pl_mod_batched)

sampled_mod = nutpie.sample(compiled_mod)

az.plot_trace(sampled_mod)

az.plot_energy(sampled_mod)

plt.close('all')

posterior_abilities = sampled_mod.posterior['player_abilities'].values
n_posterior_samples =
