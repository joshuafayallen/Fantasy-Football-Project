from models.bradleyterry import BradleyTerryModel
import polars as pl
from pathlib import Path
import pymc as pm
import arviz as az 

season = 2022

x = Path('data/')

bt_model = BradleyTerryModel()

bt_model.build_model(X = x, season = season)


with bt_model.model:
    out = pm.sample()


check = az.summary(out).reset_index()
