from typing import Dict, List, Optional, Tuple, Union
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

        n_teams = len(self.teams)
        c = self.cats
        def ordered_logistic_probs(eta,cutpoints):
            p_le = pm.math.sigmoid(cutpoints -eta[:,None])
            p_le = pm.math.concatenate(
            [pm.math.zeros((p_le.shape[0], 1)), p_le, pm.math.ones((p_le.shape[0], 1))], axis = 1)
            return p_le[:, 1:] - p_le[:, :-1]

        with pm.Model(coords = self.model_coords) as self.model:
            team_mu = pm.Normal('team_mu', mu = self.model_config.get('team_mu_mu_prior', 0), sigma = self.model_config.get('team_mu_sd_prior', 1),
            dims = 'teams') # no need to be cute

            home_advantage = pm.Normal('home_adv', mu = self.model_config.get(
                'home_adv_mu_prior', 0), sigma = self.model_config.get(
                    'home_adv_sd_prior', 1.0
                ))

            c = pm.Normal(
                mu = self.model_config.get(
                    'cutpoint_mu', 0
                ),
                sigma = self.model_config.get(
                    'cutpoint_sd', 2
                ),
                transform=pm.distributions.transforms.ordered,
                initval= np.linspace(-3, 3, c - 1)
            )
            
            theta = team_mu[self.home_ids] - team_mu[self.away_ids]
            mean_alpha = pm.Deterministic('mean_alpha', team_mu.mean())
            alpha_n = team_mu - mean_alpha
            probs = ordered_logistic_probs(alpha_n, c)
            p_win = probs[:, 4:7].sum(axis = 1)
            p_win = pm.Deterministic('p_win_neutral', p_win, dims = 'teams')
            yl = pm.OrderedLogistic('y', eta = theta, cutpoints=c, observed = self.outcome)

    @staticmethod
    def get_default_model_config() -> Dict:
        model_config: Dict = {
            'team_mu_mu_prior': 0.0,
            'team_mu_sd_prior': 1.0,
            'home_adv_mu_prior': 0.0,
            'home_adv_sd_prior': 1.0,
            'cutpoint_mu': 0,
            'cutpoind_sd': 2
        }
        return model_config
    @staticmethod
    def get_default_sampler_config() -> Dict:
        return {
            "draws": 1000,
            "tune": 1000,
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

        home_teams = df.select(pl.col("home_team")).to_series()
        away_teams = df.select(pl.col('away_team')).to_series()
        all_teams = pl.concat([home_teams, away_teams]).unique().sort()
        teams = all_teams.to_numpy()
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
                .when(pl.col('result') == 0)
                .then(pl.lit('tie'))
                .otherwise(pl.lit('Away Team Win')).alias('score_cat')).with_columns(
                    pl.when(
                    pl.col('score_cat') == 'Home Team Win')
                    .then(pl.col('home_id'))
                    .when(
                        pl.col('score_cat') == 'Away Team Win'
                    )
                    .then(pl.col('away_id'))
                    .otherwise(None)
                    .alias('winner_id'),
                    pl.when(
                    pl.col('score_cat') == 'Home Team Win')
                    .then(pl.col('away_id'))
                    .when(
                        pl.col('score_cat') == 'Away Team Win'
                    )
                    .then(pl.col('home_id'))
                    .otherwise(None)
                    .alias('loser_id'),
                pl.when(
                pl.col('home_team') == 'Home')
              .then(1)
            .otherwise(0)
            .alias('is_home_team') ).filter(pl.col('game_type') == 'REG')
        
        
        self.teams = teams
        self.winner_ids = cleaned_data['winner_id'].to_numpy()
        self._cleaned = cleaned_data
        self.home_ids = cleaned_data['home_id'].to_numpy()
        self.away_ids  = cleaned_data['away_id'].to_numpy()
        self.loser_ids = cleaned_data['loser_id'].to_numpy()
        #self.outcome = cleaned_data['outcome'].to_numpy()
        self.model_coords = {'teams': self.teams}
        self.cats = len(np.unique(cleaned_data['outcome'].to_numpy()))
        self.is_home_game = cleaned_data['is_home_game'].to_numpy()









