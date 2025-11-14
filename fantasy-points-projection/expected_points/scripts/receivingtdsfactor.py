from typing import Dict, Optional, Any, Union
import polars as pl
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc_extras.model_builder import ModelBuilder
from scipy.stats import norm


class OrderedLogit(ModelBuilder):
    _model_type = "Ordered Logit"
    version = 0.1

    def get_default_model_config() -> Dict:
        model_config: Dict = {
            "numeric_factors": [
                "player_rest_diff",
                "def_epa_diff",
                "wind",
                "temp",
                "total_line",
                "air_yards_per_pass_attempt",
            ],
            "categorical_factors": ["div_game", "home_game", "is_indoors", "era"],
            "within_m": 27,
            "within_c": 1.7083333333333333,
            "seasons_m": 108,
            "seasons_c": 4.555555555555555,
            "intercept_sigma": 4.0,
            "touchdown_sd": 1.0,
            "cutpoints_off": 4.0,
            "delta_mean_sigma": 1.0,
            "delta_sig_lam": 1.0,
            "alpha_scale": 0.03,
            "upper_scale": 2.0,
            "short_term_form_alpha": 18.12068352072794,
            "short_term_form_beta": 58.50005431282928,
            "seasons_gp_alpha": 4.024251897372992,
            "seasons_gp_beta": 14.661508040684918,
            "slope_sigma": 1.0,
        }

        return model_config

    def __init__(
        self,
        rng: int = 61164170,
        model_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        self.rng = rng

        self.numeric_factors = model_config["numeric_factors"]
        self.categorical_factors = model_config["categorical_factors"]
        self.factors = self.numeric_factors + self.categorical_factors
        self.within_m = model_config["within_m"]
        self.within_c = model_config["within_c"]
        self.seasons_m = model_config["seasons_m"]
        self.seasons_c = model_config["seasons_c"]

        self.intercept_sigma = model_config["intercept_sigma"]
        self.touchdown_sd = model_config["touchdown_sd"]
        self.cutpoints_off = model_config["cutpoints_off"]
        self.delta_mean_sigma = model_config["delta_mean_sigma"]
        self.delta_sig_lambda = model_config["delta_sig_lambda"]
        self.alpha_scale = model_config["alpha_scale"]
        self.upper_scale = model_config["upper_scale"]
        self.short_term_form_alpha = model_config["short_term_form_alpha"]
        self.short_term_form_beta = model_config["short_term_form_beta"]
        self.seasons_gp_alpha = model_config["seasons_gp_alpha"]
        self.seasons_gp_beta = model_config["seasons_gp_beta"]
        self.slope_sigma = model_config["slope_sigma"]

        self.model_config = model_config
        self.sampler_config = sampler_config
        self.means_ = None
        self.sds = None

        super().__init__(model_config=model_config, sampler_config=sampler_config)

    def _standardize_factors(self, X: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)

        factors_data = X.select(pl.col(self.factors))

        self.means_ = factors_data.select(
            [pl.col(c).mean().alias(c) for c in self.numeric_factors]
        )
        self.sds_ = factors_data.select(
            [pl.col(c).mean().alias(c) for c in self.numeric_factors]
        )

        factors_sdz = factors_data(
            [
                ((pl.col(c) - self.means_[0, c]) / self.sds_[0, c]).alias(c)
                for c in self.numeric_factors
            ]
        ).with_columns(
            pl.Series("home_game", factors_data["home_game"]),
            pl.Series("div_game", factors_data["div_game"]),
            pl.Series("is_indoors", factors_data["is_indoors"]),
            pl.Series("era", factors_data["era"]),
        )

        return factors_sdz

    def _calculate_delta_prior(df: pl.DataFrame | pd.DataFrame, y: str) -> np.ndarray:
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        empirical_probs = df[y].values_counts(normalize=True).to_numpy()

        cumulative_probs = empirical_probs.cumsum()[:-1]

        cutpoints_standard = norm.ppf(cumulative_probs)

        delta_prior = np.diff(cutpoints_standard)

        return delta_prior

    def build_model(
        self,
        X: pd.DataFrame | pl.DataFrame,
        y: pd.DataFrame | pl.DataFrame,
        coords: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        factors_data = self._standardize_factors(X)

        if isinstance(X, pl.DataFrame):
            x_pd = X.to_pandas()
        else:
            x_pd = X

        self.unique_players = x_pd["receiver_full_name"].sort_values().unique()
        self.unique_games = x_pd["games_played"].sort_values().unique()
        self.unique_seasons = x_pd["number_of_seasons_played"].sort_values.unique()

        player_idx = pd.Categorical(
            x_pd["receiver_full_name"], categories=self.unique_players
        ).codes

        games_idx = pd.Categorical(
            x_pd["games_played"], categories=self.unique_games
        ).codes

        seasons_idx = pd.Categorical(
            x_pd["number_of_seasons_played"], categories=self.unique_seasons
        ).codes

        coords = {
            "factors": self.factors,
            "gameday": self.unique_games,
            "seasons": self.unique_seasons,
            "player": self.unique_players,
            "obs_id": x_pd.index,
            "time_scale": ["games", "season"],
        }

        self.delta_prior = self._calculate_delta_prior

        with pm.Model(coords=coords) as self.model:
            factor_data = pm.Data(
                "factor_data", factors_data, dims=("obs_id", "factors")
            )
            games_id = pm.Data("games_id", games_idx, dims="obs_id")
            seasons_id = pm.Data("seasons_id", seasons_idx, dims="obs_id")

            player_id = pm.Data("player_id", player_idx, dims="obs_id")

            rec_tds_obs = pm.Data("rec_tds_obs", y["rec_tds"], dims="obs_id")
            x_gamedays = pm.Data("x_gamedays", self.unique_games, dims="gameday")[
                :, None
            ]
            x_seasons = pm.Data("x_seasons", self.unique_seasons, dims="seasons")[
                :, None
            ]

            baseline_sigma = pt.sqrt(
                (self.intercept_sigma**2 + self.touchdown_sd**2)
                / len(self.unique_players)
            )

            baseline = baseline_sigma + pm.Normal("baseline")

            player_effect = pm.Deterministic(
                "player_effect",
                baseline
                + pm.ZeroSumNormal(
                    "player_effect_raw", sigma=self.touchdown_sd, dims="player"
                ),
                dims="player",
            )

            alpha_scale, upper_scale = self.alpha_scale, self.upper_scale

            gps_sigma = pm.Exponential(
                "gps_sigma", lam=-np.log(alpha_scale) / upper_scale, dims="time_scale"
            )

            ls = pm.InverseGamma(
                "ls",
                alpha=np.array([self.short_term_form_alpha, self.seasons_gp_alpha]),
                beta=np.array([self.short_term_form_beta, self.seasons_gp_beta]),
                dims="time_scale",
            )

            cov_games = gps_sigma[0] ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ls[0])
            cov_seasons = gps_sigma[0] ** 2 * pm.gp.cov.Matern52(input_dim=1, ls=ls[0])

            gp_games = pm.gp.HSGP(m=[self.within_m], c=self.within_c, cov=cov_games)

            gp_season = pm.gp.HSGP(
                m=[self.seasons_m], c=self.seasons_c, cov=cov_seasons
            )

            basis_vectors_season, sqrt_psd_season = gp_season.prior_linearized(
                X=x_seasons
            )

            basis_vectors_game, sqrt_psd_game = gp_games.prior_linearized(X=x_gamedays)

            basis_coeffs_games = pm.Normal(
                "basis_coeffs_games", shape=gp_games.n_basis_vectors
            )

            f_games = pm.Deterministic(
                "f_games",
                basis_vectors_game @ (basis_coeffs_games * sqrt_psd_game),
                dims="gameday",
            )
            basis_vectors_season, sqrt_psd_season = gp_season.prior_linearized(
                X=x_seasons
            )

            basis_coeffs_season = pm.Normal(
                "basis_coeffs_season", shape=gp_season.n_basis_vectors
            )

            f_season = pm.Deterministic(
                "f_season",
                basis_vectors_season @ (basis_coeffs_season * sqrt_psd_season),
                dims="seasons",
            )

            alpha = pm.Deterministic(
                "alpha",
                player_effect[player_id] + f_season[seasons_id] + f_games[games_id],
                dims="obs_id",
            )

            slope = pm.Normal("slope", sigma=self.slope_sigma, dims="factors")

            eta = pm.Deterministic(
                "eta", alpha + pm.math.dot(factor_data, slope), dims="obs_id"
            )

            cutpoints_off = self.cutpoints_off

            delta_mean = pm.Normal(
                "delta_mean",
                mu=self.delta_prior * cutpoints_off,
                sigma=self.delta_mean_sigma,
                shape=2,
            )

            delta_sig = pm.Exponential("delta_sig", self.delta_sig_lambda, shape=2)

            player_delta = delta_mean + delta_sig * pm.Normal(
                "player_delta", shape=(len(coords["player"]), 2)
            )

            cutpoints = pm.Deterministic(
                "cutpoints",
                pt.concatenate(
                    [
                        pt.full((player_effect.shape[0], 1), cutpoints_off),
                        pt.cumsum(pt.softplus(player_delta), axis=-1) + cutpoints_off,
                    ],
                    axis=-1,
                ),
            )
            pm.OrderedLogistic(
                "tds_scored",
                cutpoints=cutpoints[player_id],
                eta=eta,
                observed=rec_tds_obs,
                dims="obs_id",
            )
        return self.model

    @property
    def output_var(self):
        return "tds_scored"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        return self.model_config

    def get_default_sampler_config(self, rng: int) -> Dict:
        sampler_config: Dict = {
            "target_accept": 0.99,
            "random": self.rng,
            "nuts_sampler": "numpyro",
        }
        return sampler_config
