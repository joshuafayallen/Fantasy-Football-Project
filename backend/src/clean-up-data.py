import polars as pl 
import janitor.polars
import polars.selectors as cs

## we do have some problems with the existing data 
## IN a long format we get a ton of redundancies 

big_stats = pl.read_parquet('data/fantasy-pros-stats.parquet').filter(
    pl.col('player').is_not_null()
)

# handle the 2022 class 
draft_2022 = ['Brock Purdy',
             'Desmond Ridder',
             'Malik Willis',
             'Matt Coral',
             'Sam Howell',
             'Bailey Zappe',
             'Kenny Pickett',
             'Skylar Thompson',
             'Chris Olave',
             'Garrett Wilson',
             'Kenneth Walker III',
             'Christian Watson',
             'Dameon Pierce',
             'Drake London',
             'George Pickens',
             'Jameson Williams',
             'Jahan Dotson',
             'Breece Hall',
             'Skyy Moore',
             'Trey McBride',
             'James Cook',
             'Rachaad White',
             'Brian Robinson Jr.',
             'David Bell',
             'Danny Gray',
             'Cade Otton',
             'Zamir White',
             'Erik Ezukanma',
             'Charlie Kolar',
             'Romeo Doubs',
             'Isaiah Likely',
             'Khalil Shakir',
             'Jerome Ford',
             'Kyren Williams',
             'Jalen Nailor' ]

# keeping for posterity
add_rookie_flags = big_stats.sort(['player', 'year']).with_columns(
    pl.col('player').count().over('player').alias('appearances'),
    pl.col('player').cum_count().over('player').alias('first_appearance')).with_columns(
    pl.when((pl.col('appearances') <= 2) & (pl.col('first_appearance') == 1)).then(pl.lit('rookie_year')).when((pl.col('player').is_in(draft_2022)) & (pl.col('first_appearance') == 1)).then(pl.lit('rookie_year')).otherwise(pl.lit('veteran')).alias('rookie_flag'))

add_rookie_flags = add_rookie_flags.fill_null(0)

#wide_stats = big_stats.unpivot(
#    index = ['player', 'year'], 
#    on = cs.by_dtype(pl.Int64, pl.Float64)
#).with_columns(
#    pl.concat_str([
#        pl.col('variable'),
#        pl.lit('_'), 
#        pl.col('year')]
#    ).alias('variable')
#).pivot(
#    'variable',
#    index = ['player'], 
#    values = 'value'
#)
#
#check = wide_stats.filter(
#    pl.col('player') == 'Josh Allen'
#)

# filtering by ownership will get rid of some of the riff raff
## basically we don't want to get rid of rookies like ashton jeanty. But we don't neccssaarily want
## udf's that are fighting to be on the active roster


ranking_data = pl.read_parquet('data/add-adp-to-ffverse.parquet').with_columns(
    pl.coalesce('source', 'source_right').alias('source')
)

add_stats_to_ranking = ranking_data.join(
    add_rookie_flags, on = 'player', how = 'left'
).with_columns(
    pl.when(pl.col('rookie_flag').is_null()).then(pl.lit('rookie_year')).otherwise(pl.col('rookie_flag')).alias('rookie_flag')).select(
        pl.exclude('^.*_right$', 'first_appearance', 'appearances')
    ).fill_null(0)

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


