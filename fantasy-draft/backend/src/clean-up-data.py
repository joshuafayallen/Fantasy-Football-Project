import polars as pl 
import janitor.polars
import polars.selectors as cs

## we do have some problems with the existing data 
## IN a long format we get a ton of redundancies 

big_stats = pl.read_parquet('data/fantasy-pros-stats.parquet').filter(
    pl.col('player').is_not_null()
).fill_null(0)



ranking_data = pl.read_parquet('data/add-adp-to-ffverse.parquet').with_columns(
    pl.coalesce('source', 'source_right').alias('source')
)

add_stats_to_ranking = ranking_data.join(
    big_stats, on = 'player', how = 'left'
)

add_epa = pl.read_csv('data/nfl-elo.csv').clean_names().rename(
    {
        'play': 'offensive_epa_per_play',
        'pass': 'offensive_pass_epa',
        'rush': 'offensive_rush_epa',
        'play_duplicated_0': "defensive_epa_per_play",
        'rush_duplicated_0': 'defesnsive_rush_epa',
        'pass_duplicated_0': 'defensive_pass_epa',
        'play_duplicated_1': 'net_epa_per_play'
    }
)

sos = pl.read_csv('data/nfl-sos.csv').clean_names().rename(
    {
        'win_total': 'projected_win_total',
        'original_rating': 'projected_sos'
    }
).select(
    pl.col('team', 'projected_win_total', 'projected_sos', 'avg_opp_rating')
).join(add_epa, on = 'team').with_columns(
    pl.when(
        pl.col('team') == 'OAK'
    )
    .then(pl.lit('LV'))
    .otherwise(pl.col('team')
    ).alias('team')
)

replacement_table = {
    'JAC': 'JAX', 
    'NEP': 'NE',
    'LVR': 'NE',
    'SFO': 'SF', 
    'GBP': 'GB',
    'TBB': 'TB',
    'NOS': 'NO',
    'KCC': 'KC'
}

tm_name = add_stats_to_ranking.select(
    pl.col('tm').unique()
)

make_nice_team_names = add_stats_to_ranking.with_columns(
    pl.col('tm').replace_strict(replacement_table, default = pl.col('tm')).alias('tm')
)

add_elo_ratings = make_nice_team_names.join(
    sos, left_on=['tm'], right_on=['team'], how = 'left'
)

add_elo_ratings.write_parquet(
    'processed_data', partition_by=['pos']
)


