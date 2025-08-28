from backend.config.settings import Config
from backend.collectors.main import FantasyDataCollector
import polars as pl
import polars.selectors as cs
from datetime import date 
import janitor.polars

cfg = Config()

all_data = FantasyDataCollector(cfg)


collected_data = all_data.collect_all()

espn_data = collected_data['espn']

yahoo_data = collected_data['yahoo'].with_columns(
    pl.col('average_pick_yahoo', 'average_cost_yahoo', 'projected_auction_value').str.replace_all(r'(-)','0')
).with_columns(
    pl.col('average_pick_yahoo', 'average_cost_yahoo', 'projected_auction_value').cast(pl.Float64)
)


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
    'Tennessee': 'Titans D/ST',
    'New York Jets': 'Jets D/ST',
    'New York Giants': 'Giants D/ST',
    'Los Angeles Rams': 'Rams D/ST',
    'Los Angeles Chargers': 'Chargers D/ST'
}

reversed_replacement_table = {value: key for key, value in replacement_table.
items()}


clean_espn_data = espn_data.with_columns(
    pl.col('player').replace_strict(reversed_replacement_table, default = pl.col('player')))


ffverse = pl.scan_parquet('data/ffverse-data*.parquet').collect()


added_adp = ffverse.join(yahoo_data, on = ['player'], how = 'left').join(clean_espn_data, on =['player'], 
how = 'left')

added_adp.write_parquet('data/add-adp-to-ffverse.parquet')

