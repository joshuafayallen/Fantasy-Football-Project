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
).with_columns(
    pl.col('location_enum').to_physical().alias('loc_num'),
    pl.col('player_enum').to_physical().alias('player_num'),
    pl.col('positions_enum').to_physical().alias('pos_num'),
    
)



player_pos_idx = converted_pos.select(pl.col('player_pos_idx')).to_numpy()
pos_idx = converted_pos['pos_num'].to_numpy()
player_idx = converted_pos['player_num'].to_numpy()
loc_idx = converted_pos['loc_num'].to_numpy()

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


g = sns.FacetGrid(data = yac_data.to_pandas(),col = 'receiver_position')

g.map(sns.histplot, 'yards_after_catch')

pz.Exponential(1/5).plot_pdf()
pz.Exponential(1/10).plot_pdf()
pz.Exponential(1/30).plot_pdf()
plt.xlim(0, 75)

plt.close('all')

pz.HalfCauchy(1).plot_pdf()
pz.HalfCauchy(2.5).plot_pdf()

pz.HalfNormal(3).plot_pdf()
pz.HalfNormal(2).plot_pdf()
pz.HalfNormal(1).plot_pdf()

plt.xlim(0, 75)

plt.close('all')

pz.Gamma(2,0.1).plot_pdf()

yac_data.select(pl.col('yards_after_catch').mean())



with pm.Model() as yac_model_small:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 3)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions= pm.Normal('position_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            shape = n_positions)


    player_level_sigma = pm.HalfCauchy('player_sigma', 2.5)
    player_effects = pm.Normal('player_effects', 0,
                                player_level_sigma,
                                shape = n_players)

    mu_player = mu_positions[pos_idx] + player_effects[player_idx]



    nu = pm.Gamma('nu', 2.18, 0.25)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 3)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = yac_data['yards_after_catch'].to_numpy()
    )

    out = pm.sample_prior_predictive(random_seed=1994)


az.plot_ppc(out, group='prior', num_pp_samples=100)

plt.xlim(-10, 100)
## the general machinery is in place lets now add some 

add_dummies = yac_data.to_dummies('pass_location')

g = sns.FacetGrid(yac_data.to_pandas(), col = 'pass_location', col_order = ['left', 'middle', 'right'])

# interestingly we see lot of similarities between left and right
# with some deviations
# most of the qbs are right handed so it kind of makes sense that are more 
# passes that go gor zero yac to the right
# We would probably expect that more throws could be going to the right
# just because it is a "closer" since most qb's are righties
g.map(sns.histplot, 'yards_after_catch')

# for the most part these are pretty similar so we could just set a common prior for these
# It would probably be 
yac_data.group_by('pass_location').agg(pl.col('yards_after_catch').mean())


# this model works fine in prior land but faces major problems 
# with one year of data 

with pm.Model() as yac_model_small:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions= pm.Normal('position_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            shape = n_positions)


    player_level_sigma = pm.HalfCauchy('player_sigma', 2.5, shape = n_positions)
    player_effects = pm.Normal('player_effects', 0,
                                player_level_sigma,
                                shape = n_players)

    pass_location_sigma = pm.HalfNormal('pass_loc_sigma', 1)
    pass_location_mu = pm.Normal('pass_loc_mu', 0,
                                pass_location_sigma) 


    mu_player = mu_positions[pos_idx] + player_effects[player_idx] + pass_location_mu[loc_idx]



    nu = pm.Gamma('nu', 2.18, 0.25)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 2)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = yac_data['yards_after_catch'].to_numpy()
    )

    out = pm.sample_prior_predictive(random_seed=1994)



az.plot_ppc(out, group ='prior', num_pp_samples=100)

plt.xlim(-80, 100)


## lets move onto adding air yards 



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


    nu = pm.Gamma('nu', 2.18, 0.25)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 2)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = yac_data['yards_after_catch'].to_numpy()
    )

    out = pm.sample_prior_predictive(random_seed=1994)


az.plot_ppc(out, group='prior', num_pp_samples=100)

plt.xlim(-20, 100)


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

    pass_location_sigma = pm.HalfNormal('pass_loc_sigma')                       
    
    
    mu_player =  mu_positions[pos_idx] + player_effects[player_idx] + air_yards_coef * std_air_yards['air_yards_std'].to_numpy()


    nu = pm.Gamma('nu', 1.68, 0.0426)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 2)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player,
        sigma = observed_sigma,
        observed = std_air_yards['yards_after_catch'].to_numpy()
    )
    out = pm.sample_prior_predictive(random_seed=1994)




p = (gg.ggplot(yac_data, gg.aes(x = 'vegas_wp'))
    + gg.geom_histogram(color = 'white'))

# the funny thing is that its actually pretty uniformish. This a obviously a bit more weight towards the end. It would probably just be a better idea to model it proper

p


pz.maxent(pz.Gamma(), lower = 0, upper = 1, mass = .95 )

plt.close('all')

with pm.Model() as yac_vega_wp_added:
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

    pass_location_mu = pm.Normal('pass_loc_mu', 0,1) 

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)  


    mu_wp = pm.Normal('vegas_wp', 0, 1)                      
    
    
    mu_player =  mu_positions[pos_idx] + player_effects[player_idx] + air_yards_coef * std_air_yards['air_yards_std'].to_numpy() + pass_location_mu * std_air_yards['loc_num'].to_numpy() + mu_wp * std_air_yards['vegas_wp_std'].to_numpy()


    nu = pm.Gamma('nu', 1.68, 0.0426)
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
plt.xlim(-20, 100)


# lets look at score differential against yac 


score_diff = (
    gg.ggplot(std_air_yards, gg.aes(x = 'score_differential'))
    + gg.geom_density()
    + gg.theme_minimal()
)

score_diff


# now we are getting into the new territory 
# lets first add game situation
# score differential is going to be the first one
# There are a lot of ways to think of it 
# I think what I keep coming back to is 
# score differential is in essence as distance variable 


# 

pz.Gamma(1.5, 1.5).plot_pdf()

with pm.Model() as yac_score_diff:
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

    pass_location_mu = pm.Normal('pass_loc_mu', 0,1) 

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)  


    mu_wp = pm.Normal('vegas_wp', 0, 1)

    score_min_padded = std_air_yards.select((pl.col('score_differential_c_scaled').min() - 0.5)).to_numpy().flatten()
    score_max_padded = std_air_yards.select((pl.col('score_differential_c_scaled').max()  + 0.5)).to_numpy().flatten()

    score_diff = std_air_yards['score_differential_c_scaled'].to_numpy()


    gp_lengthscale = pm.Gamma('score_diff_length', alpha = 1.5, beta = 1.5)
    gp_variance = pm.HalfNormal('score_variance', 1)
    cov_func = gp_variance * pm.gp.cov.ExpQuad(1, ls = gp_lengthscale)
    gp = pm.gp.HSGP(
        m = [20],
        L = [score_max_padded - score_min_padded],
        cov_func = cov_func
    )
    score_diff_gp = gp.prior('score_diff_gp', X = score_diff[:, None] - score_min_padded)

    mu_player =  mu_positions[pos_idx] + player_effects[player_idx] + air_yards_coef * std_air_yards['air_yards_std'].to_numpy() + pass_location_mu * std_air_yards['loc_num'].to_numpy() + mu_wp * std_air_yards['vegas_wp_std'].to_numpy() + score_diff_gp

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



## this looks really good the problem is we got kind of a big model and score diff is already a weird variable 
## so we may just need a simpler model of score differential
## this may also suggest that we may need a similar thing for time remaining 
# and yards to the sticks etc 


with pm.Model() as yac_score_diff_spline:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions= pm.Normal('position_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            shape = n_positions)


    player_level_sigma = pm.HalfCauchy('player_sigma', 1.5, shape = n_positions)
    player_effects_raw = pm.Normal('player_effects_raw', 0, 1,
                                shape = n_players)
    
    player_effects = pm.Deterministic('player_effects', player_effects_raw * player_level_sigma[player_pos_idx])

    pass_location_mu = pm.Normal('pass_loc_mu', 0,1) 

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)  


    mu_wp = pm.Normal('vegas_wp', 0, 1)

    score_diff = std_air_yards['score_differential'].to_numpy()

    score_spline = dmatrix("bs(score_diff, df=5, degree=3, include_intercept=False)", 
                   {"score_diff": score_diff}, return_type='dataframe').to_numpy()

    spline_coefs = pm.Normal('spline_coefs', 0, 0.25, shape = score_spline.shape[1])

    mu_player =  mu_positions[pos_idx] + player_effects[player_idx] + air_yards_coef * std_air_yards['air_yards_std'].to_numpy() + pass_location_mu * std_air_yards['loc_num'].to_numpy() + mu_wp * std_air_yards['vegas_wp_std'].to_numpy()

    mu_player += score_spline @ spline_coefs

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


plt.xlim(-20, 100)


std_air_yards.glimpse()



## add time splines 


with pm.Model() as yac_score_diff_spline:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions= pm.Normal('position_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            shape = n_positions)


    player_level_sigma = pm.HalfCauchy('player_sigma', 2.5, shape = n_positions)
    player_effects_raw = pm.Normal('player_effects_raw', 0, 1,
                                shape = n_players)
    
    player_effects = pm.Deterministic('player_effects', player_effects_raw * player_level_sigma[player_pos_idx])

    pass_location_mu = pm.Normal('pass_loc_mu', 0,1) 

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)  


    mu_wp = pm.Normal('vegas_wp', 0, 1)

    score_diff = std_air_yards['score_differential'].to_numpy()

    score_spline = dmatrix("bs(score_diff, df=5, degree=3, include_intercept=False)", {"score_diff": score_diff}, return_type='dataframe').to_numpy()
    time_rem = std_air_yards['game_seconds_remaining'].to_numpy()

    time_spline = dmatrix("bs(time_rem, df=5, degree=3, include_intercept=False)", {"time": time_rem}, return_type='dataframe').to_numpy()

    time_spline_coefs = pm.Normal('time_spline_coef', 0, 0.5, shape = time_spline.shape[1] )

    spline_coefs = pm.Normal('spline_coefs', 0, 0.25, shape = score_spline.shape[1])

    mu_player =  mu_positions[pos_idx] + player_effects[player_idx] + air_yards_coef * std_air_yards['air_yards_std'].to_numpy() + pass_location_mu * std_air_yards['loc_num'].to_numpy() + mu_wp * std_air_yards['vegas_wp_std'].to_numpy()

    mu_player += score_spline @ spline_coefs + time_spline @ time_spline_coefs

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


plt.xlim(-80,100)



std_air_yards.select(pl.col('ydstogo').min().alias('min'), pl.col('ydstogo').max().alias('max'))



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

get_ig_params(x_vals = std_air_yards['ydstogo'].to_numpy())

with pm.Model() as yac_yards_added:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions= pm.Normal('position_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            shape = n_positions)


    player_level_sigma = pm.HalfNormal('player_sigma', 1, shape = n_positions)
    player_effects_raw = pm.Normal('player_effects_raw', 0, 1,
                                shape = n_players)
    
    player_effects = pm.Deterministic('player_effects', player_effects_raw * player_level_sigma[player_pos_idx])

    pass_location_mu = pm.Normal('pass_loc_mu', 0,1) 

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)  


    mu_wp = pm.Normal('vegas_wp', 0, 1)

    score_diff = std_air_yards['score_differential'].to_numpy()

    score_spline = dmatrix("bs(score_diff, df=5, degree=3, include_intercept=False)", {"score_diff": score_diff}, return_type='dataframe').to_numpy()
    time_rem = std_air_yards['game_seconds_remaining'].to_numpy()

    time_spline = dmatrix("bs(time_rem, df=5, degree=3, include_intercept=False)", {"time": time_rem}, return_type='dataframe').to_numpy()

    time_spline_coefs = pm.Normal('time_spline_coef', 0, 0.25, shape = time_spline.shape[1] )

    spline_coefs = pm.Normal('spline_coefs', 0, 0.25, shape = score_spline.shape[1])


    ## 

    length_prior = pm.InverseGamma('length_prior', **get_ig_params(std_air_yards['ydstogo'].to_numpy()))
    
    cov = pm.gp.cov.Matern32(1, ls = length_prior)
    gp = pm.gp.HSGP(m = [10], c = 1.5, cov_func=cov)
    gp_prior = gp.prior('gp_prior', X = std_air_yards['ydstogo'].to_numpy()[:, None])

    mu_player =  mu_positions[pos_idx] + player_effects[player_idx] + air_yards_coef * std_air_yards['air_yards_std'].to_numpy() + pass_location_mu * std_air_yards['loc_num'].to_numpy() + mu_wp * std_air_yards['vegas_wp_std'].to_numpy()

    mu_player += score_spline @ spline_coefs + time_spline @ time_spline_coefs

    mu_player_hsgp = mu_player + gp_prior

    nu = pm.Gamma('nu', 2, 0.1)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 2)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player_hsgp,
        sigma = observed_sigma,
        observed = std_air_yards['yards_after_catch'].to_numpy()
    )

    out = pm.sample_prior_predictive(random_seed=1994)



az.plot_ppc(out, group = 'prior', num_pp_samples=100)

plt.xlim(-80, 100)



with pm.Model() as yac_yards_added:
    # hyper priors 
    # for each play we would expect a deviation of about 5 yards or so 
    yac_sigma = pm.HalfNormal('yac_sigma', 1)
    yac_mean = pm.Normal('yac_mean', 5, 1)

    # positon parameters 

    mu_positions= pm.Normal('position_mu',
                            mu = yac_mean,
                            sigma = yac_sigma,
                            shape = n_positions)


    player_level_sigma = pm.HalfNormal('player_sigma', 1, shape = n_positions)
    player_effects_raw = pm.Normal('player_effects_raw', 0, 1,
                                shape = n_players)
    
    player_effects = pm.Deterministic('player_effects', player_effects_raw * player_level_sigma[player_pos_idx])

    pass_location_mu = pm.Normal('pass_loc_mu', 0,1) 

    air_yards_coef = pm.Normal('air_yards_beta', 0, 1)  


    mu_wp = pm.Normal('vegas_wp', 0, 1)

    score_diff = std_air_yards['score_differential'].to_numpy()

    score_spline = dmatrix("bs(score_diff, df=5, degree=3, include_intercept=False)", {"score_diff": score_diff}, return_type='dataframe').to_numpy()
    time_rem = std_air_yards['game_seconds_remaining'].to_numpy()

    time_spline = dmatrix("bs(time_rem, df=5, degree=3, include_intercept=False)", {"time": time_rem}, return_type='dataframe').to_numpy()

    time_spline_coefs = pm.Normal('time_spline_coef', 0, 0.25, shape = time_spline.shape[1] )

    spline_coefs = pm.Normal('spline_coefs', 0, 0.25, shape = score_spline.shape[1])


    ## 

    length_prior = pm.InverseGamma('length_prior', **get_ig_params(std_air_yards['ydstogo'].to_numpy()))
    
    cov = pm.gp.cov.ExpQuad(1, ls = length_prior)
    gp = pm.gp.HSGP(m = [10], c = 1.5, cov_func=cov)
    gp_prior = gp.prior('gp_prior', X = std_air_yards['ydstogo'].to_numpy()[:, None])

    mu_player =  mu_positions[pos_idx] + player_effects[player_idx] + air_yards_coef * std_air_yards['air_yards_std'].to_numpy() + pass_location_mu * std_air_yards['loc_num'].to_numpy() + mu_wp * std_air_yards['vegas_wp_std'].to_numpy()

    mu_player += score_spline @ spline_coefs + time_spline @ time_spline_coefs

    mu_player_hsgp = mu_player + gp_prior

    nu = pm.Gamma('nu', 2, 0.1)
    observed_sigma = pm.HalfNormal('obs_sigma', sigma = 2)

    y_obs = pm.StudentT(
        'y_obs', 
        nu = nu,
        mu = mu_player_hsgp,
        sigma = observed_sigma,
        observed = std_air_yards['yards_after_catch'].to_numpy()
    )


compiled_yac_yards_added = nutpie.compile_pymc_model(yac_yards_added)

sampled_yac_yards = nutpie.sample(compiled_yac_yards_added, seed = 1994)

az.plot_trace(sampled_yac_yards)

pm.sample_posterior_predictive(sampled_yac_yards, yac_yards_added, extend_inferencedata=True)

az.plot_ppc(sampled_yac_yards, num_pp_samples=100)

plt.xlim(-80, 100)

# okay everything is looking good 
## now we need to go ahead and grab the real full posterior 
# so we are going to have to rewrite the model a bitg 