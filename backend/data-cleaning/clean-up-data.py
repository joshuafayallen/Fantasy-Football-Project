import polars as pl 
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


add_rookie_flags = big_stats.sort(['player', 'year']).with_columns(
    pl.col('player').count().over('player').alias('appearances'),
    pl.col('player').cum_count().over('player').alias('first_appearance')).with_columns(
    pl.when((pl.col('appearances') <= 2) & (pl.col('first_appearance') == 1)).then(pl.lit('rookie_year')).when((pl.col('player').is_in(draft_2022)) & (pl.col('first_appearance') == 1)).then(pl.lit('rookie_year')).otherwise(pl.lit('veteran')).alias('rookie_flag'))


# the rookie flag seems to be working well 
# now lets just set these to zero 

fill_null_values = add_rookie_flags.fill_null(0)

# filtering by ownership will get rid of some of the riff raff
## basically we don't want to get rid of rookies like ashton jeanty. But we don't neccssaarily want
## udf's that are fighting to be on the active roster
ranking_data = pl.read_parquet('data/ffverse-data-2025-08-12.parquet').filter(
    (pl.col('player_owned_avg') >= 25))


check = ranking_data.filter(
    pl.col('player').is_in(['Tyler Warren', 'Colston Loveland', 'Quinshon Judkins'])
)


join_stats_ranking = ranking_data.join(fill_null_values, on = 'player', how = 'left')





