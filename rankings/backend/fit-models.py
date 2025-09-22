import polars as pl 
import pymc as pm 
import arviz as az
import numpy as np
from backend.models.btmag import BradleyTerryMag
import nflreadpy as nfl
import json


seasons = range(1999,2025)


seasons_data = nfl.load_schedules(seasons=True)

team_stats = nfl.load_pbp(seasons = True)


logos_data = pl.read_csv("https://raw.githubusercontent.com/shoenot/NFL-Team-Logos-Transparent-Squared/refs/heads/main/logo_urls.csv").rename(
        {'team_abbr': 'team'}        
    )


bt_mag_mod = BradleyTerryMag()


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

    bt_mag_mod.build_model(X = df, season = i)

    with bt_mag_mod.model:
        out = pm.sample(random_seed=1994)
    
    
    
    df = pl.from_pandas(az.summary(out, var_names=['p_win_neutral', 'team_mu']).reset_index()).with_columns(
    pl.col('index').str.extract(r"\[(\w+)\]", group_index=1).alias('team'))
    ability = df.filter(
    pl.col('index').str.contains(r"team_mu")
     ).rename(
    {'mean': 'team_ability',
    'hdi_3%': 'hdi_low_ability',
    'hdi_97%': 'hdi_high_ability'}).select(pl.exclude(
    'index')).with_columns(pl.col('team_ability').rank(descending=True, method = 'ordinal').alias('Team Rank'))


    win_prob = df.filter(
    pl.col("index").str.contains(r"p_win_neutral")).rename(
    {
        'mean': 'team_win_prob',
        'hdi_3%': 'hdi_low_prob',
        'hdi_97%': 'hdi_high_prob'
    }).with_columns(
    (pl.col('team_win_prob') * 100)).select(pl.exclude('index'))

    skills = ability.join(win_prob, on = 'team')
    ability = df.filter(
    pl.col('index').str.contains(r"team_mu")
     ).rename(
    {'mean': 'team_ability',
    'hdi_3%': 'hdi_low_ability',
    'hdi_97%': 'hdi_high_ability'}).select(pl.exclude(
    'index')).with_columns(pl.col('team_ability').rank(descending=True, method = 'ordinal').alias('Team Rank'))

    
    win_prob = df.filter(
    pl.col("index").str.contains(r"p_win_neutral")).rename(
    {
        'mean': 'team_win_prob',
        'hdi_3%': 'hdi_low_prob',
        'hdi_97%': 'hdi_high_prob'
    }).with_columns(
    (pl.col('team_win_prob') * 100),
     (pl.col('hdi_low_prob') * 100).alias('hdi_low_prob'),
     (pl.col('hdi_high_prob') * 100).alias('hdi_high_prob')).select(pl.exclude('index'))

    skills = ability.join(win_prob, on = 'team').with_columns(
        pl.when(pl.col('team') == 'STL')
        .then(pl.lit('LA'))
        .when(pl.col('team') == 'SD')
        .then(pl.lit('LAC'))
        .when(pl.col('team') == 'OAK')
        .then(pl.lit('LV'))
        .otherwise(pl.col('team'))
        .alias('logo_team') )
    


    wins = bt_mag_mod._cleaned.group_by(['winner_id']).agg(
        pl.len().alias('wins'))
    losses = bt_mag_mod._cleaned.group_by(['loser_id']).agg(
        pl.len().alias('losses'))

    team_lookup = pl.DataFrame( {
        'team_id': range(len(bt_mag_mod.teams)),
        'team': bt_mag_mod.teams})


   
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



    skills = skills.join(logos_data, left_on = ['logo_team'], right_on=['team'], how = 'full')
    skills = skills.join(epa, left_on=['logo_team'], right_on=['posteam']).select(pl.exclude('logo_team'))
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


