from typing import Dict, List, Optional, Tuple, Union
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
from pymc_extras.model_builder import ModelBuilder
from pathlib import Path




class BradleyTerryModel(ModelBuilder):
    _model_type = 'Bradley-Terry Model'
    version = '0.1.1'

    def build_model(self, X: Union[str, Path, pl.DataFrame, pd.DataFrame ,None] = None, season: int = None , **kwargs):
        season = kwargs.get('season', None)
            

        self._generate_and_preprocess_model_data(X, season = season)

        with pm.Model(coords = self.model_coords) as self.model:
            team_sd = pm.HalfNormal('team_sd', sigma =  self.model_config.get('team_sd',  1.0))
            team_mu = pm.Normal('team_mu', mu = self.model_config.get('team_mu_mu_prior', 0), sigma = self.model_config.get('team_mu_sd_prior', 1),
            shape = 32) # no need to be cute

            team_skills = pm.Deterministic('team_skills', team_mu * team_sd, dims = 'teams')

            logit_skills = team_skills[self.winner_ids] - team_skills[self.loser_ids]

            pm.Bernoulli('win_lik', logit_p = logit_skills, observed = np.ones(self.winner_ids.shape[0]))

    @staticmethod
    def get_default_model_config() -> Dict:
        model_config: Dict = {
            'team_sd': 1.0,
            'team_mu_mu_prior': 0.0,
            'team_mu_sd_prior': 1.0
        }
        return model_config
    @staticmethod
    def get_default_sampler_config() -> Dict:
        return {
            "draws": 1000,
            "tune": 1000,
            "chains": 3,
            "target_accept": 0.9,
            "random_seed": 1994
        }    




    def _generate_and_preprocess_model_data(self, input_data: Union[str, Path,pd.DataFrame, pl.DataFrame], season: int = None):

        if isinstance(input_data, (str, Path)):

            path = Path(input_data)

        # CSV file
            if path.suffix == ".csv":
                df = pl.scan_csv(path).collect()

        # Parquet file or directory
            elif path.suffix in [".parquet", ".pq"] or path.is_dir():
                if path.is_dir():
                # Read all parquet files in the directory and concatenate
                    files = list(path.rglob("*.parquet"))
                if not files:
                    raise ValueError(f"No parquet files found in directory {path}")
                df = pl.concat([pl.scan_parquet(f).collect() for f in files])
            else:
                df = pl.scan_parquet(path).collect()
        elif isinstance(input_data, pd.DataFrame):
            df = pl.from_pandas(input_data)
        elif isinstance(input_data, pl.DataFrame):
            df = input_data

        else:
            raise ValueError(f"input data must be a path Polars data frame or Pandas Dataframe")

        teams = np.array(df.select(pl.col('home_team').unique()).to_series().to_list())
        if season is not None:

            df = df.filter(pl.col("season").is_in([season]))

        team_ids = pl.Enum(teams)
        cleaned_data = df.with_columns(
        (pl.col('home_score') - pl.col('away_score')).alias('result'),
        pl.col('home_team').cast(team_ids).to_physical().alias('home_id'),
        pl.col('away_team').cast(team_ids).to_physical().alias('away_id')).with_columns(
                pl.when(
                pl.col('result') > 0)
                .then(pl.lit('Home Team Win'))
                .otherwise(pl.lit('Away Team Win')).alias('score_cat')).with_columns(
                    pl.when(
                    pl.col('score_cat') == 'Home Team Win')
                    .then(pl.col('home_id'))
                    .otherwise(pl.col('away_id'))
                    .alias('winner_id'),
                    pl.when(
                    pl.col('score_cat') == 'Home Team Win')
                    .then(pl.col('away_id'))
                    .otherwise(pl.col('home_id'))
                    .alias('loser_id'))
        self.teams = teams
        self.winner_ids = cleaned_data['winner_id'].to_numpy()
        self._cleaned = cleaned_data
        self.loser_ids = cleaned_data['loser_id'].to_numpy()
        self.model_coords = {'teams': self.teams}









