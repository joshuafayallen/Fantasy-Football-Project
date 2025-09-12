from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import polars as pl
import pymc as pm
import arviz as az
from pathlib import Path
from models.bradleyterry import BradleyTerryModel
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

    df = nfl.load_schedules(seasons=request.season)
    stats = nfl.load_team_stats(seasons=request.season).filter(
        (pl.col('epa').is_not_null()) & (pl.col('play_type').is_in(['pass', 'run'])) & (pl.col('down').is_in([1,2,3,4]))
    )

    off_epa = stats.group_by(['posteam']).agg(pl.col('epa').mean().alias('off_epa_per_play'))

    def_epa = stats.group_by(['defteam']).agg(pl.col('epa').mean().alias('def_epa_per_play'))
    epa = off_epa.join(def_epa, left_on=['posteam'], right_on=['defteam'])

    bt_model = BradleyTerryModel()
    bt_model.build_model(X = df, season = request.season)

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

    logos_data = pl.read_csv("https://raw.githubusercontent.com/shoenot/NFL-Team-Logos-Transparent-Squared/refs/heads/main/logo_urls.csv").rename(
        {'team_abbr': 'team'}
    )

    skills = skills.join(logos_data, on =['team']).join(epa, left_on=['team'], right_on=['posteam'])
    skills_dict = {k: list(v) for k, v in skills.to_dict().items()}
    precomputed_results[season_key] = skills_dict
    with open(DATA_FILE, 'w') as f:
        json.dump(precomputed_results, f, indent=2)

    return JSONResponse(content = skills_dict)
