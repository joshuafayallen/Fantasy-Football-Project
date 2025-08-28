from pathlib import Path
import sys
import polars as pl 

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import ep_process, passers, rush, common

pbp_folder = Path('pbp-data')
roster_folder = Path('roster-data')

pbp_files = sorted(pbp_folder.glob('pbp-season-*.parquet'))
roster_files = sorted(roster_folder.glob('roster-season-*.parquet'))


if len(pbp_files) != len(roster_files):
    raise ValueError('Mismatch in Number of PBP files and roster files')

def build_year_map(files, prefix="pbp-season-"):
    year_map = {}
    for f in files:
        # extract year from filename
        year = f.stem.split(prefix)[-1]
        year_map[year] = f
    return year_map

pbp_map = build_year_map(pbp_files, prefix="pbp-season-")
roster_map = build_year_map(roster_files, prefix="roster-season-")

years = sorted(set(pbp_map.keys()) & set(roster_map.keys()))

all_processed_data = {}

for year in years:
    print(f"processing rosters and play by play data for {year}")

    pbp_df = pl.read_parquet(pbp_map[year])
    roster_df = pl.read_parquet(roster_map[year])

    processor = ep_process(pbp_data = pbp_df, rosters_data= roster_df)

    processed_dict = processor.process_all_data()

    all_processed_data[year] = processed_dict

processed_folder = Path('processed_data')

processed_folder.mkdir(exist_ok=True)


for year, processed_dict in all_processed_data.items():
    for key, df, in processed_dict.items():

        out_file = processed_folder / f"{key}_{year}.parquet"
        df.write_parquet(out_file)