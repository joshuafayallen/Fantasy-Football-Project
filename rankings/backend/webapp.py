from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import polars as pl
import pymc as pm
import arviz as az
from pathlib import Path
from models.btmag import BradleyTerryMag
import nflreadpy as nfl
import json

app = FastAPI(title = 'NFL Power Rankings')


app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

class SeasonRequest(BaseModel):
    season: int
DATA_FILE = Path('seasons_rankings.json')
if DATA_FILE.exists():
    with open(DATA_FILE, 'r') as f: 
        precomputed_results: dict[str, dict[str, list]] = json.load(f)
else:
    precomputed_results = {}
@app.post('/fit')

def fit_model(request:SeasonRequest):
    print(f"=== API CALL START ===")
    print(f"Received request for season: {request.season}")
    print(f"Season type: {type(request.season)}")
    print(f"Season value repr: {repr(request.season)}")
    
    season_key = f"{request.season} Season"
    print(f"Looking for season_key: '{season_key}'")
    print(f"Available keys in precomputed_results: {list(precomputed_results.keys())}")
    
    if season_key in precomputed_results:
        print("✓ Found precomputed results, returning cached data")
        return JSONResponse(content=precomputed_results[season_key])
    
    print("✗ No precomputed results found, will call nfl.import_schedules")

    df = nfl.load_schedules(seasons=request.season).filter(
        pl.col('home_score').is_not_null()
    )
    stats = nfl.load_pbp(seasons=request.season).filter(
        (pl.col('epa').is_not_null()) & (pl.col('play_type').is_in(['pass', 'run'])) & (pl.col('down').is_in([1,2,3,4]))
    )

    off_epa = stats.group_by(['posteam']).agg(pl.col('epa').mean().alias('off_epa_per_play'))

    def_epa = stats.group_by(['defteam']).agg(pl.col('epa').mean().alias('def_epa_per_play'))
    epa = off_epa.join(def_epa, left_on=['posteam'], right_on=['defteam'])

    bt_mag = BradleyTerryMag()

    bt_mag.build_model(X = df, season = request.season)

    with bt_mag.model:
        out = pm.sample(random_seed=1994, nuts_sampler='nutpie')

    
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







    wins = bt_mag._cleaned.group_by(['winner_id']).agg(
        pl.len().alias('wins'))
    losses = bt_mag._cleaned.group_by(['loser_id']).agg(
        pl.len().alias('losses'))

    team_lookup = pl.DataFrame( {
        'team_id': range(len(bt_mag.teams)),
        'team': bt_mag.teams

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

    logos_data = pl.read_csv("https://raw.githubusercontent.com/shoenot/NFL-Team-Logos-Transparent-Squared/refs/heads/main/logo_urls.csv").rename(
        {'team_abbr': 'team'}
    )



    skills = skills.join(logos_data, on = ['team'])
    skills = skills.join(epa, left_on=['team'], right_on=['posteam'])
    skills = skills.join(record, on = 'team')
    skills_list = skills.to_dicts()

    precomputed_results[season_key] = skills_list

    with open(DATA_FILE, 'w') as f:
        json.dump(precomputed_results, f, indent=2)

    return JSONResponse(content = skills_list)
