from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import polars as pl
import pymc as pm
import arviz as az
from .models.bradleyterry import BradleyTerryModel
import nfl_data_py as nfl

app = FastAPI(title = 'NFL Power Rankings')


app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

class SeasonRequest(BaseModel):
    season: int
@app.post('/fit')

def fit_model(request:SeasonRequest):
    df = nfl.import_schedules([request.season])

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

    skills = skills.join(logos_data, on =['team'])
    skills_dict = skills.to_dicts()

    return JSONResponse(content = skills_dict)
