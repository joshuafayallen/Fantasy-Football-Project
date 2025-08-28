import nfl_data_py as nfl
import polars as pl
from pathlib import Path
import sys
import sys 


pbp_folder = Path('pbp-data')
pbp_folder.mkdir(exist_ok=True)

roster_folder = Path('roster-data')
roster_folder.mkdir(exist_ok=True)

year = int(sys.argv[1])

pbp_df = nfl.import_pbp_data(years = [year])

pbp_file = pbp_folder / f"pbp-season-{year}.parquet"

roster_file = roster_folder / f"roster-season-{year}.parquet"

pbp_df.to_parquet(pbp_file)

roster_df = nfl.import_seasonal_rosters(years = [year])
roster_df.to_parquet(roster_file)

print(f"Downloaded data for {year}")