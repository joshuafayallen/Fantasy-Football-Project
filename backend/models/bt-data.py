import arviz as az
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import polars as pl 
import polars.selectors as cs
import preliz as pz
import pymc as pm


# we need to get a rough sketch of age 
# I don't think it really matters all that much that the accuracy of the ages will be a bit off
# the functional difference between 30 in 


raw_fantasy_data_pos = pl.scan_parquet(
    'processed_data'
).filter(pl.col('pos').is_in(['DST']).not_()).collect().sort('avg_expert_ranking').with_columns(
    pl.col('birth_date').dt.year().alias('birth_year'),
    pl.when(
    pl.col('year').is_null()).then(2025).otherwise(pl.col('year')
    ).alias('year')
   
).with_columns(
     (pl.col('year') - pl.col('birth_year')).alias('age')
).select(
    pl.exclude('^.*_right$')
)

raw_fantasy = pl.scan_parquet(
    'processed_data'
).collect().filter(pl.col('pos') == 'DST')

## when working with qb's for whatever reason the mean function failed 
## so we have to fill the nans

bring_back_in_d = pl.concat([raw_fantasy_data_pos, raw_fantasy], how = 'diagonal').select(pl.exclude('^.*_right$')).fill_nan(0).fill_null(0)


def standardize_columns(cols, group):
    return [
        (
            (pl.col(col) - pl.col(col).mean().over(group)) /
            pl.col(col).std().over(group)
        ).alias(f"{col}_zstd")
        for col in cols
    ]

def center_columns(cols, group):
    return[
        (
            pl.col(col) - pl.col(col).mean().over(group)
        ).alias(f"{col}_centered")
        for col in cols
    ]

### we have a ton of covariates per position so lets 
### narrow this down a bit 
## we probably want to maximize on the efficiency stats 
## yards_per_attempt, yards_per_reception, target_share, red_zone_tgt_share
### passing_yards_per_attempt 
## TD production is also important
## so we should start off by centering it



sd_cols = bring_back_in_d.with_columns(
    *standardize_columns(['avg_expert_ranking', 'yards_per_attempt', 'yards_per_reception', 'target_share', 'red_zone_targets', 'passing_yards_per_attempt', 'rushing_yards', 'receiving_yards'], group = 'pos')
).fill_nan(None).with_columns(
    *center_columns(['receiving_tds', 'passing_tds', 'rushing_tds'], group = 'pos'), 
    pl.col('yahoo_rank').str.to_integer()
)

analysis_data = sd_cols.select(
    pl.col('player', 'pos'), cs.matches('rank|centered|zstd')
)

# just pre sorting
bt_data = analysis_data.unique(
    'player'
).filter(
    pl.col('yahoo_rank').is_not_null()
).sort('avg_expert_ranking')

sites = ['espn_rank', 'yahoo_rank']
pairs = []

for site in sites:
    site_id = site.split("_")[0]
    players = bt_data['player'].to_list()
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pairs.append({'site': site_id, 'winner': players[i], 'loser': players[j]})


pairs_df = pl.DataFrame(pairs)

players = pairs_df["winner"].unique().to_list() + pairs_df["loser"].unique().to_list()
players = list(set(players))  # Ensure all unique players
player_idx = {p: i for i, p in enumerate(players)}

sites = pairs_df["site"].unique().to_list()
site_idx = {s: i for i, s in enumerate(sites)}

pos = analysis_data['pos'].unique().to_list()
pos_ids = {s:i for i, s in enumerate(pos)}

pairs_df_add_id = pairs_df.with_columns([
    pl.col("winner").map_elements(player_idx.get, return_dtype=pl.Int64).alias("winner_id"),
    pl.col("loser").map_elements(player_idx.get,return_dtype=pl.Int64).alias("loser_id"),
    pl.col("site").map_elements(site_idx.get,return_dtype=pl.Int64).alias("site_id"),
])

winner_ids = pairs_df_add_id['winner_id'].to_numpy()
loser_ids = pairs_df_add_id['loser_id'].to_numpy()

with pm.Model() as bt_model:

    player_skill_raw = pm.Normal('player_skill_raw', mu = 0, sd = 1, shape = len(players))
    # lets start off with a half normal
    # I don't expect a ton of heavy tails 
    player_sd_raw = pm.HalfNormal('player_sd_raw', 1.0)
    player_skills = pm.Deterministic('player_skills',  player_skill_raw * player_sd_raw)
    logit_skills = player_skills[winner_ids] - player_skills[loser_ids]
    likelihood = pm.Bernoulli(
        'ranking_likliehood', logit_p = logit_skills, observed = n
    )