import polars as pl 
import pymc as pm 
import arviz as az
import numpy as np
from backend.models.bradleyterry import BradleyTerryModel
from backend.models.davidsonmodel import DavidsonModel
import nflreadpy as nfl
import json


seasons = range(1999,2025)


seasons_data = nfl.load_schedules(seasons=True)

team_stats = nfl.load_pbp(seasons = True)


bt_model = BradleyTerryModel()

davidson_mod = DavidsonModel()


logos_data = pl.read_csv("https://raw.githubusercontent.com/shoenot/NFL-Team-Logos-Transparent-Squared/refs/heads/main/logo_urls.csv").rename(
        {'team_abbr': 'team'}        
    )


seasons_dict = {}


for i in seasons:
    df = seasons_data.filter(
        pl.col('season') == i
    )

    ties_exist = (df['home_score'] == df['away_score']).any()

    stats = team_stats.filter((pl.col('season') == i) & (pl.col('epa').is_not_null()) & (pl.col('play_type').is_in(['pass', 'run'])) & (pl.col('down').is_in([1,2,3,4]))).select(
    pl.col('season', 'week', 'epa', 'play_id', 'posteam', 'defteam')
)
    off_epa = stats.group_by(['posteam']).agg(pl.col('epa').mean().alias('off_epa_per_play'))

    def_epa = stats.group_by(['defteam']).agg(pl.col('epa').mean().alias('def_epa_per_play'))
    epa = off_epa.join(def_epa, left_on=['posteam'], right_on=['defteam'])

    if ties_exist:
        model_to_use = davidson_mod
    else: 
        model_to_use = bt_model

  

    model_to_use.build_model(X = df, season = i)

    with model_to_use.model:
        out = pm.sample(random_seed=1994, draws=5000, nuts_sampler='nutpie')

    
    if model_to_use._model_type == 'Davidson Model':
        skills = pl.from_pandas(az.summary(out, round_to = None).reset_index()).filter(
    pl.col('index').str.contains(r'(team_mu)')).with_columns(
    pl.col('index').str.extract(r"\[(\w+)\]", group_index=1),
    pl.col('mean').rank(descending=True, method = 'ordinal').alias('Team Rank')).rename(
    {'index': 'team'}).sort('Team Rank').rename(
        {'hdi_3%': 'hdi_lower',
        'hdi_97%': 'hdi_upper'}
    )
    else: 
        skills = pl.from_pandas(az.summary(out, round_to = None).reset_index()).filter(
    pl.col('index').str.contains(r'(team_skills)')).with_columns(
    pl.col('index').str.extract(r"\[(\w+)\]", group_index=1),
    pl.col('mean').rank(descending=True, method = 'ordinal').alias('Team Rank')).rename(
    {'index': 'team'}).sort('Team Rank').rename(
        {'hdi_3%': 'hdi_lower',
        'hdi_97%': 'hdi_upper'}
    ).with_columns(
        ((pl.col('mean').exp()/ (pl.lit(1) + pl.col('mean').exp())) * 100).alias('prob_beat_avg')
    )



    wins = model_to_use._cleaned.group_by(['winner_id']).agg(
        pl.len().alias('wins'))
    losses = model_to_use._cleaned.group_by(['loser_id']).agg(
        pl.len().alias('losses'))

    team_lookup = pl.DataFrame( {
        'team_id': range(len(model_to_use.teams)),
        'team': model_to_use.teams

    })


   
    record = wins.join(losses, left_on='winner_id', right_on='loser_id', how='full').with_columns(
    pl.col('winner_id').fill_null(pl.col('loser_id')).alias('team_id')).select(
    'team_id', 'wins', 'losses').join(team_lookup, on='team_id', how='left').with_columns(
    pl.col('wins').fill_null(0),
    pl.col('losses').fill_null(0)).select(
    'team', 'wins', 'losses').sort('wins', descending=True).with_columns(
        (pl.col('wins') + pl.col('losses')).alias('games_played')
    ).with_columns(
    pl.when(pl.col('games_played') == pl.col('games_played').max())
    .then(pl.concat_str([
        pl.col('wins'),
        pl.lit('-'),
        pl.col('losses')
    ]))
    .otherwise(pl.concat_str([
        pl.col("wins"),
        pl.lit('-'),
        pl.col('losses'),
        pl.lit('-'),
        pl.lit('1')]
    )).alias('record')
    ).select(pl.col('team', 'record'))



    skills = skills.join(logos_data, on = ['team'])
    skills = skills.join(epa, left_on=['team'], right_on=['posteam'])
    skills = skills.join(record, on = 'team')
    


    season_key = f"{i} Season"
    seasons_dict[season_key] = skills.to_dict()


serializable_dict = {
    season: {k: list(v) for k, v in df.items()}
    for season, df in seasons_dict.items()
}

# Save to a JSON file
with open("./frontend/public/seasons_rankings.json", "w") as f:
    json.dump(serializable_dict, f, indent=2)


with open('backend/seasons_rankings.json', 'w') as f:
    json.dump(serializable_dict, f, indent=2)
## random wal


