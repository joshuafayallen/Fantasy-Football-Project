from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
from pymc_extras.model_builder import ModelBuilder
import pytensor.tensor as pt
from pathlib import Path




class DavidsonModel(ModelBuilder):
    _model_type = 'Davidson Model'
    version = '0.1.1'

    def build_model(self, X: Union[str, Path, pl.DataFrame, pd.DataFrame ,None] = None, season: int = None , **kwargs):
        season = kwargs.get('season', None)
            

        self._generate_and_preprocess_model_data(X, season = season)



        with pm.Model(coords = self.model_coords) as self.model:
            
            team_mu = pm.Normal('team_mu', mu = self.model_config.get('team_mu_mu_prior', 0), sigma = self.model_config.get('team_mu_sd_prior', 1),
            dims = 'teams') 

            home_advantage = pm.Normal('home_adv', mu = self.model_config.get(
                'home_adv_mu_prior', 0), sigma = self.model_config.get(
                    'home_adv_sd_prior', 1
                ))
            
            prob_of_ties = pm.Normal('tie_prob', mu = self.model_config.get('prior_ties_mu', 0), sigma = self.model_config.get(
                'prior_ties_std',1 
            ))
            

            home_strength = team_mu[self.home_ids]
            away_strength = team_mu[self.away_ids]

            home_strength_adjusted = home_strength + home_advantage
            home_exp = pt.exp(home_strength_adjusted)
            away_exp = pt.exp(away_strength)

            tie_term = pt.exp(prob_of_ties + 0.5 * (home_strength_adjusted + away_strength))

            denom = away_exp + home_exp + tie_term

            p_away_win = away_exp/denom
            p_home_win = home_exp/denom
            p_tie = tie_term/denom

            probs = pt.stack([p_away_win, p_home_win, p_tie], axis = 1)

            pm.Categorical('outcomes', p = probs, observed = self.outcomes)

    @staticmethod
    def get_default_model_config() -> Dict:
        model_config: Dict = {
            'team_sd': 1.0,
            'team_mu_mu_prior': 0.0,
            'team_mu_sd_prior': 1.0,
            'home_adv_mu_prior': 0.0,
            'home_adv_sd_prior': 1.0, 
            'prior_ties_mu': 0,
            'prior_ties_std': 1
        }
        return model_config
    @staticmethod
    def get_default_sampler_config() -> Dict:
        return {
            "draws": 2000,
            "tune": 2000,
            "chains": 4,
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
            raise ValueError("input data must be a path Polars data frame or Pandas Dataframe")
        if season is not None:

            df = df.filter((pl.col("season").is_in([season])) ).filter(
                pl.col('game_type') == 'REG')

        
        home_teams = df.select(pl.col("home_team")).to_series()
        away_teams = df.select(pl.col('away_team')).to_series()
        all_teams = pl.concat([home_teams, away_teams]).unique().sort()
        teams = all_teams.to_numpy()
        team_ids = pl.Enum(teams)

        cleaned_data = df.with_columns(
        (pl.col('home_score') - pl.col('away_score')).alias('result'),
        pl.col('home_team').cast(team_ids).to_physical().alias('home_id'),
        pl.col('away_team').cast(team_ids).to_physical().alias('away_id')).with_columns(
                pl.when(
                pl.col('result') > 0)
                .then(pl.lit('Home Team Win'))
                .when(pl.col('result') == 0)
                .then(pl.lit('Tie'))
                .otherwise(pl.lit('Away Team Win')).alias('score_cat')).with_columns(
                    pl.when(
                        pl.col('score_cat') == 'Home Team Win')
                    .then(pl.lit(1)) # win
                    .when(pl.col('score_cat') == 'Tie')
                    .then(pl.lit(2)) # tie
                    .otherwise(pl.lit(0)) # loss
                    .alias('outcome'),
                    pl.when(pl.col('score_cat') == 'Home Team Win')
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0)).alias('is_home_game')
                   ).with_columns(
                    pl.when(pl.col('score_cat') == 'Home Team Win')
                    .then(pl.col('home_id'))
                    .when(pl.col('score_cat') == 'Away Team Win')
                    .then(pl.col('away_id'))
                    .otherwise(None)
                    .alias('winner_id'),
                    pl.when(pl.col('score_cat') == 'Home Team Win')
                    .then(pl.col('away_id'))
                    .when(pl.col('score_cat') == 'Away Team Win')
                    .then(pl.col('home_id'))
                    .otherwise(None)
                    .alias('loser_id')
                   ).filter(
                pl.col('game_type') == 'REG')
        self.teams = teams
        self.outcomes = cleaned_data['outcome'].to_numpy()
        self._cleaned = cleaned_data
        self.home_ids = cleaned_data['home_id'].to_numpy()
        self.away_ids = cleaned_data['away_id'].to_numpy()
        self.model_coords = {'teams': self.teams}
        self.is_home_game = cleaned_data['is_home_game'].to_numpy()










