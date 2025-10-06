import nflreadpy as nfl
import polars as pl
from pathlib import Path
import sys


pbp_folder = Path('pbp-data')
pbp_folder.mkdir(exist_ok=True)

roster_folder = Path('roster-data')
roster_folder.mkdir(exist_ok=True)

year = int(sys.argv[1])

pbp_df = nfl.load_pbp(seasons = year)

pbp_file = pbp_folder / f"pbp-season-{year}.parquet"

roster_file = roster_folder / f"roster-season-{year}.parquet"

pbp_df.write_parquet(pbp_file)

roster_df = nfl.load_rosters(seasons = [year])
roster_df.write_parquet(roster_file)

print(f"Downloaded data for {year}")