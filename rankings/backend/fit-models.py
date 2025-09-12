import polars as pl 
import pymc as pm 
import arviz as az
import numpy as np
from backend.models.bradleyterry import BradleyTerryModel
import nflreadpy as nfl
import json


seasons = range(1999,2025)


seasons_data = nfl.load_schedules(seasons=True)

team_stats = nfl.load_pbp(seasons = True)

team_stats.select(pl.col('season').min())


logos_data = pl.read_csv("https://raw.githubusercontent.com/shoenot/NFL-Team-Logos-Transparent-Squared/refs/heads/main/logo_urls.csv").rename(
        {'team_abbr': 'team'}

        
    )

bt_model = BradleyTerryModel()

seasons = seasons_data.select(pl.col('season').unique()).to_series().to_list()

seasons_dict = {}





for i in seasons:
    df = seasons_data.filter(
        pl.col('season') == i
    )
    stats = team_stats.filter((pl.col('season') == i) & (pl.col('epa').is_not_null()) & (pl.col('play_type').is_in(['pass', 'run'])) & (pl.col('down').is_in([1,2,3,4]))).select(
    pl.col('season', 'week', 'epa', 'play_id', 'posteam', 'defteam')
)
    off_epa = stats.group_by(['posteam']).agg(pl.col('epa').mean().alias('off_epa_per_play'))

    def_epa = stats.group_by(['defteam']).agg(pl.col('epa').mean().alias('def_epa_per_play'))
    epa = off_epa.join(def_epa, left_on=['posteam'], right_on=['defteam'])
  

    bt_model.build_model(X = df, season = i)

    with bt_model.model:
        out = pm.sample(random_seed=1994, nuts_sampler='nutpie')
    

    skills = pl.from_pandas(az.summary(out, round_to = None).reset_index()).filter(
    pl.col('index').str.contains(r'(team_skills)')).with_columns(
    pl.col('index').str.extract(r"\[(\w+)\]", group_index=1),
    pl.col('mean').rank(descending=True).alias('Team Rank')).rename(
    {'index': 'team'}).sort('Team Rank').rename(
        {'hdi_3%': 'hdi_lower',
        'hdi_97%': 'hdi_upper'}
    )

    skills = skills.join(logos_data, on = ['team'])
    skills = skills.join(epa, left_on=['team'], right_on=['posteam'])
    


    season_key = f"{i} Season"
    seasons_dict[season_key] = skills.to_dict()


serializable_dict = {
    season: {k: list(v) for k, v in df.items()}
    for season, df in seasons_dict.items()
}

# Save to a JSON file
with open("./frontend/public/seasons_rankings.json", "w") as f:
    json.dump(serializable_dict, f, indent=2)

## random wal

