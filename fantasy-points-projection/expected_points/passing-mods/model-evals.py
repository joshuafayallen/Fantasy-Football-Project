## lets compare the models 
import polars as pl 
import polars.selectors as cs
import xarray as xr
from patsy import dmatrix
import pymc as pm 
import pytensor.tensor as pt 
import nutpie
import preliz as pz
import seaborn as sns
import numpy as np
import janitor.polars
import matplotlib.pyplot as plt
import arviz as az


pass_completion = pl.read_parquet('processed_data/processed_passers_2020.parquet')

# stolen for bayesian data analysis in python 3
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

yac_predictors = ['off_play_caller','receiver_full_name','relative_to_endzone', 'wind', 'score_differential', 'qtr', 'down', 'game_seconds_remaining' ,'pass_location', 'ydstogo', 'temp', 'air_yards', 'yardline_100', 'ep', 'receiver_position', 'vegas_wp', 'xpass', 'surface', 'no_huddle', 'fixed_drive', 'game_id', 'yards_after_catch', 'posteam_type', 'roof', 'desc']

yac_data = pass_completion.filter(pl.col('complete_pass') == '1').select(pl.col(yac_predictors)).filter(
    (pl.col('yards_after_catch').is_not_null()) & 
    (pl.col('receiver_position').is_in(['RB', 'TE', 'WR']))
)

n_players = len(yac_data.select(pl.col('receiver_full_name').unique()).to_series().to_list())


pos_codes = yac_data.select(pl.col('receiver_position').unique()).to_series().to_list()

player_codes = yac_data.select(pl.col('receiver_full_name').unique()).to_series().to_list()

location_codes = yac_data.select(pl.col('pass_location').unique()).to_series().to_list()

play_caller_codes = yac_data.select(pl.col('off_play_caller').unique()).to_series().to_list()


n_locations = len(location_codes)
n_play_callers = len(play_caller_codes)

location_codes_array = np.array(location_codes)
player_codes_array = np.array(player_codes)
play_caller_array = np.array(play_caller_codes)
pos_codes_array = np.array(pos_codes)
play_caller_enum = pl.Enum(play_caller_array)


pos_enum = pl.Enum(pos_codes_array)
player_enum = pl.Enum(player_codes_array)
location_enum = pl.Enum(location_codes_array)

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



pos_idx = converted_pos['pos_num'].to_numpy()
player_idx = converted_pos['player_num'].to_numpy()
loc_idx = converted_pos['loc_num'].to_numpy()
play_caller_idx = converted_pos['play_caller_num'].to_numpy()


coords = {
    'players': player_codes_array,
    'players_flat': player_codes_array[player_idx],
    'positions': pos_codes_array,
    'positions_flat': pos_codes_array[pos_idx]
}


player_to_pos = (
    converted_pos
    .select(['player_num', 'pos_num'])
    .unique()
    .sort('player_num')  # align with enumeration order
)

player_pos_idx = player_to_pos['pos_num'].to_numpy()


std_air_yards = converted_pos.with_columns(
    ((pl.col('air_yards') - pl.col('air_yards').mean())/ pl.col('air_yards').std()).alias('air_yards_std'), 
    ((pl.col('vegas_wp') - pl.col('vegas_wp').mean())/ pl.col('vegas_wp').std()).alias('vegas_wp_std'), 
    # center and scale by touchdowns 
    ((pl.col('score_differential') - pl.col('score_differential').mean())/ 7).alias('score_differential_c_scaled')
)


with pm.Model(coords = coords) as yac_yards_added_final:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions= pm.Normal('position_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            dims = 'positions')


    player_level_sigma = pm.HalfNormal('player_sigma', 1, dims = 'positions')
    player_effects_raw = pm.Normal('player_effects_raw', 0, 1, dims = 'players')
    

    pass_location_mu = pm.Normal('pass_loc_mu', 0,1) 

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)  


    mu_wp = pm.Normal('vegas_wp', 0, 1)

    score_diff = std_air_yards['score_differential'].to_numpy()

    score_spline = dmatrix("bs(score_diff, df=3, degree=3, include_intercept=False)", {"score_diff": score_diff}, return_type='dataframe').to_numpy()
    time_rem = std_air_yards['game_seconds_remaining'].to_numpy()

    time_spline = dmatrix("bs(time_rem, df=3, degree=3, include_intercept=False)", {"time": time_rem}, return_type='dataframe').to_numpy()

    time_spline_coefs = pm.Normal('time_spline_coef', 0, 0.25, shape = time_spline.shape[1] )

    score_coefs = pm.Normal('spline_coefs', 0, 0.25, shape = score_spline.shape[1])

    length_prior = pm.InverseGamma('length_prior', **get_ig_params(std_air_yards['ydstogo'].to_numpy()))
    
    cov = pm.gp.cov.ExpQuad(1, ls = length_prior)
    gp = pm.gp.HSGP(m = [8], c = 1.5, cov_func=cov)
    gp_prior = gp.prior('gp_prior', X = std_air_yards['ydstogo'].to_numpy()[:, None])
    gp_prior_flat = pt.flatten(gp_prior)

    player_effects = pm.Deterministic('player_effects', player_effects_raw * player_level_sigma[player_pos_idx])
    mu_player = pm.Deterministic('player_mu', mu_positions[pos_idx] + player_effects[player_idx]
        + air_yards_coef * std_air_yards['air_yards_std'].to_numpy()
        + pass_location_mu * std_air_yards['loc_num'].to_numpy()
        + mu_wp * std_air_yards['vegas_wp_std'].to_numpy()
        + pt.dot(score_spline, score_coefs)
        + pt.dot(time_spline, time_spline_coefs)
        +  gp_prior_flat,
        dims = 'players_flat')

    nu = pm.Gamma('nu', 2, 0.1)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 1)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = std_air_yards['yards_after_catch'].to_numpy(),
        dims = 'players_flat'

    )
    player_position_trace= pm.sample(nuts_sampler='nutpie', random_seed=1994,
                    progressbar=True)


with pm.Model(coords=coords) as yac_play_caller:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    mu_play_caller_raw = pm.Normal('play_caller_mu_raw',
                            mu = 0,
                            sigma = 1,
                            shape = n_play_callers)

    mu_play_caller = pm.Deterministic(
        'mu_play_caller',
         yac_mean + yac_sigma * mu_play_caller_raw 
    )


    player_level_sigma = pm.HalfNormal('player_sigma', 1.0)

    player_effects = pm.Normal('player_effects_raw', 0, player_level_sigma,
                                shape = n_players)

    pass_location_mu = pm.Normal('pass_loc_mu', 0,1) 

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)   
    mu_wp = pm.Normal('vegas_wp', 0, 1) 
    score_diff = std_air_yards['score_differential'].to_numpy() 
    score_spline = dmatrix("bs(score_diff, df=3, degree=3,  include_intercept=False)", {"score_diff": score_diff},  return_type='dataframe').to_numpy()
    time_rem = std_air_yards['game_seconds_remaining'].to_numpy() 
    time_spline = dmatrix("bs(time_rem, df=5, degree=3, include_intercept=False) ", {"time": time_rem}, return_type='dataframe').to_numpy()

    time_spline_coefs = pm.Normal('time_spline_coef', 0, 0.1, shape =  time_spline.shape[1] )

    score_coefs = pm.Normal('spline_coefs', 0, 0.1, shape = score_spline.shape[1])
    length_prior = pm.InverseGamma('length_prior', **get_ig_params(std_air_yards['ydstogo'].to_numpy()))
    cov = pm.gp.cov.ExpQuad(1, ls = length_prior)
    gp = pm.gp.HSGP(m = [10], c = 1.5, cov_func=cov)
    gp_prior = gp.prior('gp_prior', X = std_air_yards['ydstogo'].to_numpy()[:, None])


    mu_player =  pm.Deterministic('mu_player', mu_play_caller[play_caller_idx]
                + player_effects[player_idx]
                + air_yards_coef * std_air_yards['air_yards_std'].to_numpy()
                + pass_location_mu * std_air_yards['loc_num'].to_numpy()
                + mu_wp * std_air_yards['vegas_wp_std'].to_numpy()
                + pm.math.dot(score_spline, score_coefs)
                + pm.math.dot(time_spline, time_spline_coefs)
                + gp_prior
               ) 


    nu = pm.Gamma('nu', alpha = 2.0, beta  = 0.1)

    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 1)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = std_air_yards['yards_after_catch'].to_numpy(),
        dims='players_flat'
    )
    coaching_model_trace = pm.sample(random_seed = 1994, nuts_sampler='nutpie')

## since there is still an outstanding nutpie pr we do this 



with yac_play_caller:
    pm.compute_log_likelihood(coaching_model_trace)

with yac_yards_added_final:
    pm.compute_log_likelihood(player_position_trace)

## the player position model ended up being the best 
# one thing that is kind of interesting is that I think the player postion model is a bit more pessimistic
# while the play caller model is a bit more optimistic

pm.sample_posterior_predictive(coaching_model_trace, yac_play_caller, extend_inferencedata=True)
pm.sample_posterior_predictive(player_position_trace, yac_yards_added_final, extend_inferencedata=True)



model_dict = dict(zip(['Play Caller Model', 'Positions Model'], [coaching_model_trace, player_position_trace]))

comp = az.compare(model_dict)

comp

ppc_w = az.weight_predictions(
    [model_dict[name] for name in comp.index],
    weights=comp.weight,
)

az.plot_kde(
    coaching_model_trace.posterior_predictive["y_obs"].values,
    plot_kwargs={"color": "C0", "linestyle": "--"},
    label="Coaching Mod",
)
az.plot_kde(
    player_position_trace.posterior_predictive["y_obs"].values,
    plot_kwargs={"color": "C0", "linestyle": "-"},
    label="Positions Model",
)
az.plot_kde(
    ppc_w.posterior_predictive["y_obs"].values,
    plot_kwargs={"color": "C3", "linewidth": 2},
    label="average_model",
)



with pm.Model(coords = coords) as  combo_mod:
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions= pm.Normal('position_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            dims = 'positions')

    mu_play_caller = pm.Normal('mu_play_caller',
                            mu = 0,
                            sigma = 1,
                            shape = n_play_callers)
    
    player_level_sigma = pm.HalfNormal('player_sigma', 1, dims = 'positions')
    player_effects_raw = pm.Normal('player_effects_raw', 0, 1, dims = 'players')
    

    pass_location_mu = pm.Normal('pass_loc_mu', 0,1) 

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)  


    mu_wp = pm.Normal('vegas_wp', 0, 1)

    score_diff = std_air_yards['score_differential'].to_numpy()

    score_spline = dmatrix("bs(score_diff, df=3, degree=3, include_intercept=False)", {"score_diff": score_diff}, return_type='dataframe').to_numpy()
    time_rem = std_air_yards['game_seconds_remaining'].to_numpy()

    time_spline = dmatrix("bs(time_rem, df=3, degree=3, include_intercept=False)", {"time": time_rem}, return_type='dataframe').to_numpy()

    time_spline_coefs = pm.Normal('time_spline_coef', 0, 0.25, shape = time_spline.shape[1] )

    score_coefs = pm.Normal('spline_coefs', 0, 0.25, shape = score_spline.shape[1])

    length_prior = pm.InverseGamma('length_prior', **get_ig_params(std_air_yards['ydstogo'].to_numpy()))
    
    cov = pm.gp.cov.ExpQuad(1, ls = length_prior)
    gp = pm.gp.HSGP(m = [8], c = 1.5, cov_func=cov)
    gp_prior = gp.prior('gp_prior', X = std_air_yards['ydstogo'].to_numpy()[:, None])
    gp_prior_flat = pt.flatten(gp_prior)

    player_effects = pm.Deterministic('player_effects', player_effects_raw * player_level_sigma[player_pos_idx])
    mu_player = pm.Deterministic('player_mu', mu_play_caller[play_caller_idx] + mu_positions[pos_idx] + player_effects[player_idx]
        + air_yards_coef * std_air_yards['air_yards_std'].to_numpy()
        + pass_location_mu * std_air_yards['loc_num'].to_numpy()
        + mu_wp * std_air_yards['vegas_wp_std'].to_numpy()
        + pt.dot(score_spline, score_coefs)
        + pt.dot(time_spline, time_spline_coefs)
        +  gp_prior_flat,
        dims = 'players_flat')

    nu = pm.Gamma('nu', 2, 0.1)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 1)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = std_air_yards['yards_after_catch'].to_numpy(),
        dims = 'players_flat'

    )

    out = pm.sample(random_seed=1994, nuts_sampler='nutpie')




with combo_mod:
    pm.compute_log_likelihood(out)


pm.sample_posterior_predictive(out, combo_mod, extend_inferencedata=True)

az.plot_ppc(out, num_pp_samples=100)

plt.xlim(-80,100)

model_dict = dict(zip(['Play Caller Model', 'Positions Model', 'combo mod'], [coaching_model_trace, player_position_trace, out]))

az.compare(model_dict)
