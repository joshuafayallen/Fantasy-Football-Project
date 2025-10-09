import polars as pl 
import polars.selectors as cs
import xarray as xr
import pandas as pd
from patsy import dmatrix
import pymc as pm 
import os
import pytensor.tensor as pt 
import nutpie
import preliz as pz
import seaborn as sns
import numpy as np
from scipy.special import logit
import janitor.polars
import matplotlib.pyplot as plt
import arviz as az
import nflreadpy as nfl 


seed = sum(map(ord, 'receivingyardsproject'))
rng = np.random.default_rng(seed)

full_pass_data = pl.scan_parquet('processed_data/processed_passers_*.parquet').collect()

full_scores = nfl.load_schedules()

player_exp = nfl.load_players().select(
    pl.col('gsis_id', 'display_name' , 'birth_date', 'rookie_season')
)


clean_full_scores = full_scores.select(
    pl.col('game_id','home_rest','week' ,'away_rest' ,'home_score', 'away_score' ,'home_team', 'away_team', 'result', 'total', 'total_line', 'div_game')
)


rec_predictors = ['posteam','off_play_caller','receiver_full_name','receiver_player_id','receiving_yards', 'week', 'air_yards','epa' ,'receiver_position', 'surface', 'no_huddle', 'game_id', 'yards_after_catch','roof', 'game_id', 'complete_pass', 'targeted', 'defteam', 'wind', 'temp', 'def_play_caller', 'season', 'total_pass_attempts', 'pass_touchdown']

rec_data_full = full_pass_data.with_columns(
    pl.col('pass_attempt').sum().over(['receiver_full_name', 'game_id']).alias('targeted'),
    pl.col('pass_attempt').sum().over(['posteam', 'game_id']).alias('total_pass_attempts')
).filter((pl.col('week') <= 18)).select(pl.col(rec_predictors)).filter(
    (pl.col('yards_after_catch').is_not_null()) & 
    (pl.col('receiver_position').is_in(['RB', 'TE', 'WR']))
).with_columns(
    pl.col('complete_pass').str.to_integer().count().over('receiver_player_id', 'season').alias('receptions_season'),
    pl.col('complete_pass').str.to_integer().count().over(['receiver_player_id', 'game_id', 'season']).alias('receptions_per_game'),
    (pl.col('epa') * -1).alias('defensive_epa')
)

rec_data_full = full_pass_data.with_columns(
    pl.col('pass_attempt').sum().over(['receiver_full_name', 'game_id']).alias('targeted'),
    pl.col('pass_attempt').sum().over(['posteam', 'game_id']).alias('total_pass_attempts')
).filter((pl.col('complete_pass') == '1') & (pl.col('week') <= 18)).select(pl.col(rec_predictors)).filter(
    (pl.col('yards_after_catch').is_not_null()) & 
    (pl.col('receiver_position').is_in(['RB', 'TE', 'WR']))
).with_columns(
    pl.col('complete_pass').str.to_integer().count().over('receiver_player_id', 'season').alias('receptions_season'),
    pl.col('complete_pass').str.to_integer().count().over(['receiver_player_id', 'game_id', 'season']).alias('receptions_per_game'),
    (pl.col('epa') * -1).alias('defensive_epa')
)



agg_full_seasons = rec_data_full.with_columns(
    pl.col('yards_after_catch').sum().over(['receiver_full_name', 'game_id', 'season']).alias("yac_per_game"),
    pl.col('receiving_yards').sum().over(['receiver_full_name', 'game_id', 'season']), 
    (pl.col('air_yards')/pl.col('total_pass_attempts')).alias('avg_depth_of_target').over(['posteam', 'game_id', 'season']),
    (pl.col('receiving_yards')/ pl.col('receptions_per_game')).alias('yards_per_catch'),
    # how efficient was the offense in the game
    pl.col('epa').mean().over(['game_id', 'posteam', 'season']).alias('pass_epa_per_play'),
    # how efficient was the defense 
    pl.col('defensive_epa').mean().over(['game_id', 'defteam', 'season']).alias('def_epa_per_play'),
    pl.col('pass_touchdown').str.to_integer()).with_columns(
    pl.col('pass_touchdown').sum().over(['receiver_full_name', 'game_id', 'season']).alias('rec_tds_game')
).unique(subset = ['game_id', 'receiver_full_name', 'season']).select(
    # get rid of the per play to not have any confusion
    pl.exclude('epa', 'defensive_epa')
).with_columns(
    pl.when(pl.col('rec_tds_game') > 0)
    .then(1)
    .otherwise(0)
    .alias('rec_tds')
)


joined_scores = agg_full_seasons.join(clean_full_scores, on = ['game_id'], how = 'left').with_columns(
     pl.when( pl.col('posteam') == pl.col('home_team'))
    .then(pl.col('home_score'))
    .otherwise(pl.col('away_score'))
    .alias('player_team_score'),
    pl.when(pl.col('defteam') == pl.col('away_team'))
    .then(pl.col("away_score"))
    .otherwise(pl.col('home_score'))
    .alias('opponent_score'),
    pl.when( pl.col('posteam') == pl.col('home_team'))
    .then(pl.col('home_rest'))
    .otherwise(pl.col('away_rest'))
    .alias('player_rest'),
    pl.when(pl.col('defteam') == pl.col('away_team'))
    .then(pl.col("away_rest"))
    .otherwise(pl.col('home_rest'))
    .alias('opponent_rest'),
    pl.when(
        pl.col('posteam') == pl.col('home_team')
    )
    .then(1)
    .otherwise(0)
    .alias('home_game')
    
).sort(['posteam','season', 'week' ,'receiver_full_name']).with_columns(
    (pl.col('player_team_score') - pl.col("opponent_score")).alias('player_team_score_diff'),
    (pl.col('opponent_score') - pl.col('player_team_score')).alias('opponent_score_diff'),
    (pl.col('player_rest') - pl.col('opponent_rest')).alias('player_rest_diff'),
    (pl.col('opponent_rest') - pl.col('player_rest')).alias('opponent_rest_diff')
)



# we are goinng to effectively do games played 

construct_games_played = joined_scores.with_columns(
    pl.col('game_id').cum_count().over(['receiver_full_name', 'season']).alias('games_played'),
    ((pl.col('receiving_yards') - pl.col("receiving_yards").mean())).alias('receiving_yards_c')).join(player_exp, left_on=['receiver_player_id'], right_on='gsis_id', how = 'left').with_columns(
        (pl.col('season') - pl.col('rookie_season')).alias('number_of_seasons_played'),
        pl.col('birth_date').str.to_date().dt.year().alias('birth_year')
    ).with_columns(
        (pl.col('season') - pl.col('birth_year')).alias('age')
    ).sort(['receiver_full_name', 'season', 'game_id'])

# For whatever reason constructing idx natively in polars
# and then feeding it to pymc is kind of a pain 
construct_games_played_pd = construct_games_played.to_pandas()



unique_games = construct_games_played_pd['games_played'].sort_values().unique()
unique_seasons = construct_games_played_pd['number_of_seasons_played'].sort_values().unique()

off_play_caller = construct_games_played_pd['off_play_caller'].sort_values().unique()
def_play_caller = construct_games_played_pd['def_play_caller'].sort_values().unique()

unique_players =  construct_games_played_pd['receiver_full_name'].sort_values().unique()


player_idx = pd.Categorical(
    construct_games_played_pd['receiver_full_name'], categories=unique_players
).codes

games_idx = pd.Categorical(
    construct_games_played_pd['games_played'], categories=unique_games
).codes

off_play_caller_idx = pd.Categorical(
    construct_games_played_pd['off_play_caller'], categories=off_play_caller
).codes

def_play_caller_idx = pd.Categorical(
    construct_games_played_pd['def_play_caller'], categories=def_play_caller
).codes




factors_numeric = ['player_team_score_diff', 'opponent_score_diff' ,  'player_rest_diff', 'opponent_rest_diff',  'pass_epa_per_play', 'def_epa_per_play', 'total_pass_attempts', 'wind', 'temp']

factors = factors_numeric + ['div_game', 'home_game']

factors_numeric_train = construct_games_played.select(
    factors_numeric
)


factors
means = factors_numeric_train.select([pl.col(c).mean().alias(c) for c in factors_numeric])
sds = factors_numeric_train.select([pl.col(c).std().alias(c) for c in factors_numeric])

factors_numeric_sdz = factors_numeric_train.with_columns(
    [
        ((pl.col(c)  - means[0,c])/ sds[0,c]).alias(c)
        for c in factors_numeric
    ]
).with_columns(
    pl.Series("home_game", construct_games_played['home_game']),
    pl.Series('div_game', construct_games_played['div_game'])
)

coords = {
    'factors': factors,
    'gameday': unique_games,
    'seasons': unique_seasons,
    'obs_id': construct_games_played_pd.index,
    'player': unique_players,
    'off_play_caller': off_play_caller,
    'def_play_caller': def_play_caller, 
    
}


seasons_gp_prior, ax = pz.maxent(pz.InverseGamma(), lower = 2, upper = 8)


seasons_m, seasons_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range = [0, construct_games_played.select(pl.col('number_of_seasons_played').max()).to_series()[0]],
    lengthscale_range = [2,8],
    cov_func='matern52'
)

# 
short_term_form, ax  = pz.maxent(pz.InverseGamma(), lower=2, upper = 5)

med_form, ax = pz.maxent(pz.InverseGamma(), lower = 10, upper = 18)


within_m, within_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[0, construct_games_played.select(pl.col('games_played').max()).to_series()[0]],
    lengthscale_range=[2,18],
    cov_func='matern52'
)


touchdown_dist, ax = pz.maxent(pz.Exponential(), 0.1, 2)



with pm.Model(coords = coords) as receiving_mod_long:

    gameday_id = pm.Data('gameday_id', games_idx, dims = 'obs_id')
    seasons_id = pm.Data('season_id', construct_games_played_pd['number_of_seasons_played'], dims = 'obs_id')
    
    off_id = pm.Data('off_play_caller_id', off_play_caller_idx, dims = 'obs_id')

    def_id = pm.Data('def_play_caller_id', def_play_caller_idx, dims = 'obs_id')


    x_gamedays = pm.Data("X_gamedays", unique_games, dims = 'gameday')[:,None]
    x_season = pm.Data('x_season', unique_seasons, dims = 'seasons')[:,None]

    fct_data = pm.Data('factor_num_data', factors_numeric_sdz.to_numpy(), dims = ('obs_id', 'factors_num'))
    
    
    player_id = pm.Data('player_id', player_idx, dims = 'obs_id')
    
    td_obs = pm.Data('rec_obs', construct_games_played_pd['rec_tds'].to_numpy(), dims = 'obs_id')

    sigma_player = touchdown_dist.to_pymc('player_sigma')
    
    # setting this at the mean
    player_effects = pm.Normal('player_z', mu = logit(construct_games_played_pd['rec_tds'].mean()) , sigma =  sigma_player, dims='player')
    
    ls_games = short_term_form.to_pymc('games_lengthscale_prior')

    # the upper scale 
    alpha_scale, upper_scale = 0.01, 2.0

    sigma_games = pm.Exponential('sigma_game', -np.log(alpha_scale)/upper_scale)

    cov_games = sigma_games**2 * pm.gp.cov.Matern52(input_dim=1, ls = ls_games)


    gp_within = pm.gp.HSGP(
        m = [within_m],
        c = within_c,
        cov_func=cov_games
    )
    f_within = gp_within.prior(
        'f_within',
        X = x_gamedays,
        hsgp_coeffs_dims = 'basis_coeffs_within',
        dims = 'gameday'
    )

    # just getting crazy to see how this affects the sampling
    sigma_season = pm.Exponential('sigma_season',  -np.log(alpha_scale)/upper_scale)

    ls_season = seasons_gp_prior.to_pymc(name = 'seasons_lengthscale_prior')

    cov_season = sigma_season**2 * pm.gp.cov.Matern52(1, ls = ls_season)

    

    gp_season = pm.gp.HSGP(
        m = [seasons_m],
        c = seasons_c,
        cov_func=cov_season
    )
    
    f_season = gp_season.prior(
        'f_season',
        X = x_season,
        hsgp_coeffs_dims='basis_coeffs_seasons',
        dims = 'seasons'
    )

    slope_num = pm.Normal('slope_num', sigma = 0.25, dims = 'factors')


    alpha = pm.Deterministic(
        'alpha',
        player_effects[player_id] 
        + f_within[gameday_id]
        + f_season[seasons_id]
        , dims = 'obs_id'
    )
    
    mu_player = pm.Deterministic(
        'mu_player', pm.math.sigmoid(alpha + pm.math.dot(fct_data, slope_num)) , dims = 'obs_id')

    p = pm.Bernoulli(
        'tds_scored',
        p = mu_player,
        observed = td_obs,
        dims = 'obs_id'
    )

    trace = pm.sample(nuts_sampler='nutpie', random_seed=rng)



trace.sample_stats['diverging'].values.sum()

az.plot_trace(
    trace,
    var_names=['slope_num', 'sigma_season', 'sigma_game', 'games_lengthscale_prior', 'seasons_lengthscale_prior', 'player_z', 'player_sigma']
)
az.plot_ess(
    trace,
    kind = 'evolution',
    var_names=[RV.name for RV in receiving_mod_long.free_RVs if RV.size.eval() <=3],
    grid = (5,2),
    textsize=25
)






