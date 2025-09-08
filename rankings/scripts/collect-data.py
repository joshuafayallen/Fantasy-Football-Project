import polars as pl 
import nfl_data_py as nfl

seasons_game = pl.from_pandas(nfl.import_schedules(years = [2022,2023,2024,2025]))

seasons_game.write_parquet('data', partition_by='season')


