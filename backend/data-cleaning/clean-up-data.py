import polars as pl 
import polars.selectors as cs 
import preliz as pz
import arviz as az
import pymc as pm 

## we do have some problems with the existing data 
## IN a long format we get a ton of redundancies 

big_stats = pl.read_parquet('data/fantasy-pros-stats.parquet').filter(
    pl.col('player').is_not_null()
)

# handle the 2022 class 

add_rookie_flags = big_stats.sort(['player', 'year']).with_columns(
    pl.col('player').count().over('player').alias('appearances'),
    pl.col('player').cum_count().over('player').alias('first_appearance')).with_columns(
        pl.when((pl.col('appearances') <= 2) & (pl.col('first_appearance') == 1)).then(pl.lit('rookie_year')).when((pl.col('player').is_in(['Brock Purdy', 'Desmond Ridder', 'Malik Willis', 'Matt Coral', 'Sam Howell', 'Bailey Zappe', 'Kenny Pickett', 'Skylar Thompson', 'Chris Olave', 'Garrett Wilson', 'Kenneth Walker III', 'Christian Watson', 'Dameon Pierce','Drake London', 'George Pickens', 'Jameson Williams', 'Jahan Dotson', 'Breece Hall', 'Skyy Moore', 'Trey McBride', 'Sam LaPorta', 'James Cook', 'Rachaad White', 'Brian Robinson Jr.', 'David Bell', 'Danny Gray', 'Cade Otton', 'Zamir White', 'Erik Ezukanma', 'Charlie Kolar', 'Romeo Doubs', 'Isaiah Likely', 'Khalil Shakir', 'Jerome Ford', 'Kyren Williams', 'Jalen Nailor' ])) & (pl.col('first_appearance') == 1)).then(pl.lit('rookie_year')).otherwise(pl.lit('veteran')).alias('rookie_flag')).select(
        pl.col('player', 'year', 'appearances', 'first_appearance', 'rookie_flag')
    )



