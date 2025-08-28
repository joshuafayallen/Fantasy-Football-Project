import polars as pl 
import polars.selectors as cs
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


# no need to be to cute
# Overtime is recorded as 5 
# there is some sillyness in the receiver positions category
n_qtrs = 5
n_downs = 4
n_positions = 7

yac_predictors = ['receiver_full_name','relative_to_endzone', 'wind', 'score_differential', 'qtr', 'down', 'game_seconds_remaining' ,'pass_location', 'ydstogo', 'temp', 'air_yards', 'yardline_100', 'ep', 'receiver_position', 'vegas_wp', 'xpass', 'surface', 'no_huddle', 'fixed_drive', 'game_id', 'yards_after_catch', 'posteam_type', 'roof', 'desc']

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


## we also have some candidates for gps
## relative to endzone, game seconds remaining, and score differential
## game seconds is obviously a time component and 
## relative to endzone and score differential are both, in effect distance metrics 

# does it make sense to not have a global effect of down? 
# I don't really thing it does because at a baseline we would think
# that passes are more likely later in the downs if the offense is on scheducle 


pos_codes = yac_data.select(pl.col('receiver_position').unique()).to_series().to_list()

player_codes = yac_data.select(pl.col('receiver_full_name').unique()).to_series().to_list()

location_codes = yac_data.select(pl.col('pass_location').unique()).to_series().to_list()

n_locations = len(location_codes)

location_codes_array = np.array(location_codes)
player_codes_array = np.array(player_codes)
pos_codes_array = np.array(pos_codes)

pos_enum = pl.Enum(pos_codes_array)
player_enum = pl.Enum(player_codes_array)
location_enum = pl.Enum(location_codes_array)

converted_pos = yac_data.with_columns(
    pl.col('receiver_position').cast(pos_enum).alias('positions_enum'),
    pl.col('receiver_full_name').cast(player_enum).alias('player_enum'),
    pl.col('pass_location').cast(location_enum).alias('location_enum')
)

pos_idx = converted_pos['positions_enum'].to_physical().to_numpy()
player_idx = converted_pos['player_enum'].to_physical().to_numpy()
loc_idx = converted_pos['location_enum'].to_physical().to_numpy()



with pm.Model() as yac_model_small:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 3)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions = pm.Normal('position_mu', mu = yac_mean, sigma = yac_sigma, shape = n_positions)


    player_level_sigma = pm.HalfNormal('player_sigma', 2)
    player_effects = pm.Normal('player_effects', 0, player_level_sigma, shape = n_players)

    mu_player = mu_positions[pos_idx] + player_effects[player_idx]



    nu = pm.Gamma('nu', 2, 0.1)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 4)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = yac_data['yards_after_catch'].to_numpy()
    )

    out = pm.sample(random_seed=1994)

    
    
std_air_yards = yac_data.with_columns(
    ((pl.col('air_yards') - pl.col('air_yards'))/ pl.col('air_yards').std()).alias('air_yards_std')
)


with pm.Model() as yac_model_ay_added:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions= pm.Normal('position_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            shape = n_positions)


    player_level_sigma = pm.HalfCauchy('player_sigma', 1.5)
    player_effects = pm.Normal('player_effects', 0,
                                player_level_sigma,
                                shape = n_players)

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)                         
    
    
    mu_player = pm.Deterministic('mu', mu_positions[pos_idx] + player_effects[player_idx] + air_yards_coef * std_air_yards['air_yards_std'].to_numpy())


    nu = pm.Gamma('nu', 1.68, 0.0426)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 2)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = std_air_yards['yards_after_catch'].to_numpy()
    )
    out = pm.sample(random_seed=1994)


az.plot_trace(out)