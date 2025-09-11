import polars as pl 
import pymc as pm 
import arviz as az
import numpy as np
from backend.models.bradleyterry import BradleyTerryModel
import nfl_data_py as nfl
import json

seasons = range(1999,2025)


seasons_data = pl.from_pandas(nfl.import_schedules(years = seasons))


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

    bt_model.build_model(X = df, season = i)

    with bt_model.model:
        out = pm.sample(random_seed=1994, nuts_sampler='nutpie')
    

    skills = pl.from_pandas(az.summary(out).reset_index()).filter(
    pl.col('index').str.contains(r'(team_skills)')).with_columns(
    pl.col('index').str.extract(r"\[(\w+)\]", group_index=1),
    pl.col('mean').rank(descending=True).alias('Team Rank')).rename(
    {'index': 'team'}).sort('Team Rank').rename(
        {'hdi_3%': 'hdi_lower',
        'hdi_97%': 'hdi_upper'}
    )

    skills = skills.join(logos_data, on = ['team'])
    


    season_key = f"{i} Season"
    seasons_dict[season_key] = skills.to_dict()


serializable_dict = {
    season: {k: list(v) for k, v in df.items()}
    for season, df in seasons_dict.items()
}

# Save to a JSON file
with open("season_rankings.json", "w") as f:
    json.dump(serializable_dict, f, indent=2)

## random wal

