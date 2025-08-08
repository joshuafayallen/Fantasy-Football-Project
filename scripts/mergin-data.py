import polars as pl 
import polars.selectors as cs 
import re

espn_data = pl.scan_parquet('data/espn_fantasy*.parquet').collect()


replacement_table = {
    'Denver': 'Broncos D/ST', 
    'Washington': 'Commanders D/ST',
    'Philadelphia': 'Eagles D/ST',
    'Pittsburgh': 'Steelers D/ST',
    'Baltimore': 'Ravens D/ST', 
    'Minnesota': 'Vikings D/ST',
    'Kansas City': 'Chiefs D/ST',
    'Houston' : 'Texans D/ST',
    'Detroit': 'Lions D/ST',
    'Buffalo': 'Bills D/ST',
    'Chicago': 'Bears D/ST', 
    'San Francisco': '49ers D/ST', 
    'Tampa Bay': 'Buccaneers D/ST', 
    'Arizona': 'Cardinals D/ST',
    'Seattle': 'Seahawks D/ST', 
    'Cleveland': 'Browns D/ST', 
    'Dallas': 'Cowboys D/ST', 
    'New England': 'Patriots D/ST', 
    'New Orleans': 'Saints D/ST', 
    'Cincinnati': 'Bengals D/ST', 
    'Indianapolis': 'Colts D/ST',
    'Atlanta': 'Falcons D/ST', 
    'Las Vegas': 'Raiders D/ST', 
    'Carolina': 'Panthers D/ST', 
    'Green Bay': 'Packers D/ST',
    'Miami': 'Dolphins D/ST',
    'Jacksonville': 'Jaguars D/ST', 
    'Tennessee': 'Titans D/ST'
}


yahoo_data = pl.scan_parquet('data/yahoo_fantasy*.parquet').with_columns(
    full_name = pl.col('full_name').replace_strict(replacement_table, default = pl.col('full_name'))
).with_columns(
      pl.when(
        (pl.col('full_name') == 'Los Angeles') & (pl.col('team_abr') == 'LAR')
    ).then(pl.lit('Rams D/ST'))
    .when(
        (pl.col('full_name') == 'Los Angeles') & (pl.col('team_abr') == 'LAC')
    ).then(pl.lit('Chargers D/ST'))
    .when(
        (pl.col('full_name') == 'New York') & (pl.col('team_abr') == 'NYG')
    ).then(pl.lit('Giants D/ST'))
    .when(
        (pl.col('full_name') == 'New York') & (pl.col('team_abr') == 'NYJ')
    ).then(pl.lit('Jets D/ST')).when(
        pl.col('full_name') == "Tre' Harris"
    ).then(pl.lit('Tre Harris'))
    .otherwise(pl.col('full_name'))
    .alias('full_name')
)

merged_data = yahoo_data.join(espn_data, on = 'full_name', how = 'left')


