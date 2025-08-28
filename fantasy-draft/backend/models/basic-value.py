import polars as pl 
import polars.selectors as cs
import pandas as pd 
import pymc as pm 
import pytensor.tensor as pt
import numpy as np
import nutpie 

raw_fantasy_dat = pl.scan_parquet("processed_data/**/*.parquet").select(
    pl.col('player', 'pos','fpts_g' ,'espn_rank', 'yahoo_rank', 'espn_adp')
).unique('player').with_columns(
    pl.col('yahoo_rank').str.to_integer()
).filter(pl.col('espn_adp').is_not_null()).collect()


aggregated_ranks = raw_fantasy_dat.with_columns(
    pl.concat_list(['espn_rank', 'yahoo_rank']).alias('rankings'),
    pl.mean_horizontal('espn_rank', 'yahoo_rank').alias('consensus_rank')
)

get_value = aggregated_ranks.with_columns(
    (pl.col('espn_adp') - pl.col('consensus_rank')).alias('value_score')
).sort('espn_rank').select(pl.exclude('rankings'))

n_teams = 10

draft_board = get_value.sort("espn_adp").with_columns(
    ((pl.arange(0, pl.len()) % n_teams) + 1).alias("pick_number"),
    ((pl.arange(0, pl.len()) // n_teams) + 1).alias("round")
)


draft_board = draft_board.with_columns(
    pl.when(pl.col("round") % 2 == 0)  # even round
      .then(n_teams + 1 - pl.col("pick_number"))
      .otherwise(pl.col("pick_number"))
      .alias("pick_number_snake")
).with_columns(
    pl.col('value_score').round_sig_figs(2)
)

ref = draft_board.select(
    pl.col('player', 'espn_rank', 'value_score', 'espn_adp', 'fpts_g', 'round', 'pick_number_snake')
).sort('espn_rank')


