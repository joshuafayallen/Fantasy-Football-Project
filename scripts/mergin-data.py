import polars as pl
import datetime
from scripts.fantasycollector import FantasyDataCollector

ff_collector = FantasyDataCollector()

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

reversed_replacement_table = {value: key for key, value in replacement_table.items()}

espn_data = ff_collector.espn_draft()


clean_espn_data = espn_data.with_columns(
    pl.col('full_name').replace_strict(reversed_replacement_table, default = pl.col('full_name')))

yahoo_data= ff_collector.get_yahoo_data()

merged_data = yahoo_data.join(espn_data, on = 'full_name')


ffverse = pl.scan_parquet('data/ffverse-data*.parquet').collect()


add_adp = ffverse.join(merged_data, left_on=['player'], right_on=['full_name'])

date = datetime.date.today()

add_adp.write_parquet(f'data/ffverse-add-adp-{date}.parquet')
