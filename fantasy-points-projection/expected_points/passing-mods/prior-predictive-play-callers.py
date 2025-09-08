import polars as pl 
import polars.selectors as cs
from patsy import dmatrix
import pymc as pm 
import nutpie
import preliz as pz
import seaborn as sns
import numpy as np
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import plotnine as gg

pass_completion = pl.read_parquet('processed_data/processed_passers_2020.parquet')

def get_ig_params(x_vals, l_b=None, u_b=None, mass=0.95, plot=False):
    """
    Returns a weakly informative prior for the length-scale parameter of the GP kernel.
    """

    differences = np.abs(np.subtract.outer(x_vals, x_vals))
    if l_b is None:
        l_b = np.min(differences[differences != 0]) * 2
    if u_b is None:
        u_b = np.max(differences) / 1.5

    dist = pz.InverseGamma()
    pz.maxent(dist, l_b, u_b, mass, plot=plot)

    return dict(zip(dist.param_names, dist.params))


n_qtrs = 5
n_downs = 4
n_positions = 3

yac_predictors = ['off_play_caller', 'receiver_full_name','relative_to_endzone', 'wind', 'score_differential', 'qtr', 'down', 'game_seconds_remaining' ,'pass_location', 'ydstogo', 'temp', 'air_yards', 'yardline_100', 'ep', 'receiver_position', 'vegas_wp', 'xpass', 'surface', 'no_huddle', 'fixed_drive', 'game_id', 'yards_after_catch', 'posteam_type', 'roof', 'desc', 'team', 'posteam']

yac_data = pass_completion.filter(pl.col('complete_pass') == '1').select(pl.col(yac_predictors)).filter(
    (pl.col('yards_after_catch').is_not_null()) & 
    (pl.col('receiver_position').is_in(['RB', 'TE', 'WR']))
)

n_players = len(yac_data.select(pl.col('receiver_full_name').unique()).to_series().to_list())



## lets look at it by pass locations 
## We aren't that surprised that passes to the right and left 
## have a much higher density of zero YAC. Wheareas iddle has a little bit less 

# when we look at the simple descriptives
# the middle is right around 5 
# with a spread of about 6 yards 
yac_data.select(pl.col('yards_after_catch').median().alias('yac_median'),
                pl.col('yards_after_catch').mean().alias('yac_mean'),
                pl.col('yards_after_catch').std().alias('yac_sd'))


pos_codes = yac_data.select(pl.col('receiver_position').unique()).to_series().to_list()

player_codes = yac_data.select(pl.col('receiver_full_name').unique()).to_series().to_list()

location_codes = yac_data.select(pl.col('pass_location').unique()).to_series().to_list()

play_caller_codes = yac_data.select(pl.col('off_play_caller').unique()).to_series().to_list()

n_locations = len(location_codes)

n_play_callers = len(play_caller_codes)


location_codes_array = np.array(location_codes)
player_codes_array = np.array(player_codes)
pos_codes_array = np.array(pos_codes)
play_caller_array = np.array(play_caller_codes)

pos_enum = pl.Enum(pos_codes_array)
player_enum = pl.Enum(player_codes_array)
location_enum = pl.Enum(location_codes_array)
play_caller_enum = pl.Enum(play_caller_array)

converted_pos = yac_data.with_columns(
    pl.col('receiver_position').cast(pos_enum).alias('positions_enum'),
    pl.col('receiver_full_name').cast(player_enum).alias('player_enum'),
    pl.col('pass_location').cast(location_enum).alias('location_enum'),
    pl.col('off_play_caller').cast(play_caller_enum).alias('play_caller_enum')
).with_columns(
    pl.col('location_enum').to_physical().alias('loc_num'),
    pl.col('player_enum').to_physical().alias('player_num'),
    pl.col('positions_enum').to_physical().alias('pos_num'),
    pl.col('play_caller_enum').to_physical().alias('play_caller_num')
    
)

# okay this gets the correct play caller indexs. The problem is that in the 
# data if SF is playing the cardinals then george kittle gets two play callers! 
pos_idx = converted_pos['pos_num'].to_numpy()
player_idx = converted_pos['player_num'].to_numpy()
loc_idx = converted_pos['loc_num'].to_numpy()
play_caller_idx = converted_pos['play_caller_num'].to_numpy()

player_to_pos = (converted_pos.rename(
    {'team': 'play_caller_team'}
).filter(pl.col('posteam') == pl.col('play_caller_team'))
  .select(['player_num', 'play_caller_num', 'play_caller_team', 'receiver_full_name', 'off_play_caller'])
  .unique()
  .sort('player_num'))

player_coach_idx = player_to_pos['play_caller_num'].to_numpy()


std_air_yards = converted_pos.with_columns(
    ((pl.col('air_yards') - pl.col('air_yards').mean())/ pl.col('air_yards').std()).alias('air_yards_std'), 
    ((pl.col('vegas_wp') - pl.col('vegas_wp').mean())/ pl.col('vegas_wp').std()).alias('vegas_wp_std'), 
    # center and scale by touchdowns 
    ((pl.col('score_differential') - pl.col('score_differential').mean())/ 7).alias('score_differential_c_scaled')
)

with pm.Model() as yac_play_caller_simple:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    mu_play_caller = pm.Normal('play_caller_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            shape = n_play_callers)


    player_level_sigma = pm.HalfNormal('player_sigma', 1)

    player_effects = pm.Normal('player_effects_raw', 0, player_level_sigma,
                                shape = n_players)
    

    mu_player =  mu_play_caller[play_caller_idx] + player_effects[player_idx] 


    nu = pm.Gamma('nu', 2, 0.1)

    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 2)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = std_air_yards['yards_after_catch'].to_numpy()
    )
    out = pm.sample_prior_predictive(random_seed=1994)


az.plot_ppc(out, group = 'prior', num_pp_samples=100)

plt.xlim(-80, 100)
