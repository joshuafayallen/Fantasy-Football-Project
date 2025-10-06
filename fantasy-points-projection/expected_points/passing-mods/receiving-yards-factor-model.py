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
).with_columns(
    # ideally we would have adjusted games lost 
    (pl.col('home_rest') - pl.col('away_rest')).alias('rest_diff'),
    # how far off was vegas?
    (pl.col('total') - pl.col('total_line')).alias('diff_between_line_and_result')
    )

rec_predictors = ['posteam','off_play_caller','receiver_full_name','receiver_player_id','receiving_yards', 'week', 'air_yards','epa' ,'receiver_position', 'surface', 'no_huddle', 'game_id', 'yards_after_catch','roof', 'game_id', 'complete_pass', 'targeted', 'defteam', 'wind', 'temp', 'def_play_caller', 'season', 'total_pass_attempts']



rec_data_full = full_pass_data.with_columns(
    pl.col('pass_attempt').sum().over(['receiver_full_name', 'game_id', 
    'season']).alias('targeted'),
    pl.col('pass_attempt').sum().over(['posteam', 'game_id', 'season']).alias('total_pass_attempts')
).filter((pl.col('complete_pass') == '1') & (pl.col('week') <= 18)).select(pl.col(rec_predictors)).filter(
    (pl.col('yards_after_catch').is_not_null()) & 
    (pl.col('receiver_position').is_in(['RB', 'TE', 'WR']))
).with_columns(
    pl.col('complete_pass').str.to_integer().count().over('receiver_player_id', 'season').alias('receptions_season'),
    pl.col('complete_pass').str.to_integer().count().over(['receiver_player_id', 'game_id', 'season']).alias('receptions_per_game'),
    (pl.col('epa') * -1).alias('defensive_epa')
).filter(
    pl.col('receptions_season') >= 19 
)


agg_full_seasons = rec_data_full.with_columns(
    pl.col('yards_after_catch').sum().over(['receiver_full_name', 'game_id', 'season']),
    pl.col('receiving_yards').sum().over(['receiver_full_name', 'game_id', 'season']), 
    (pl.col('air_yards')/pl.col('total_pass_attempts')).alias('avg_depth_of_target'),
    (pl.col('receiving_yards')/ pl.col('receptions_per_game')).alias('yards_per_catch'),
    # how efficient was the offense in the game
    pl.col('epa').mean().over(['game_id', 'posteam', 'season']).alias('off_epa_per_play'),
    # how efficient was the defense 
    pl.col('defensive_epa').mean().over(['game_id', 'defteam', 'season']).alias('def_epa_per_play')
).unique(subset = ['game_id', 'receiver_full_name', 'season']).select(
    # get rid of the per play to not have any confusion
    pl.exclude('epa', 'defensive_epa')
)


joined_scores = agg_full_seasons.join(clean_full_scores, on = ['game_id'], how = 'left').sort(['posteam','week', 'receiver_full_name'])


off_play_caller = joined_scores['off_play_caller'].unique().to_numpy()
def_play_caller = joined_scores['def_play_caller'].unique().to_numpy() 
unique_players =  joined_scores.select(pl.col('receiver_full_name').unique()).to_series().to_list()

player_array = np.array(unique_players)
def_play_caller_array = np.array(def_play_caller)
off_play_caller_array = np.array(off_play_caller)

player_codes = pl.Enum(player_array)
def_codes = pl.Enum(def_play_caller_array)
off_codes = pl.Enum(off_play_caller_array)

# we do have to construct the week indexes a bit different 
# since we have to construct indices for each player 
# we are goinng to effectively do games played 

construct_games_played = joined_scores.with_columns(
    pl.col('game_id').cum_count().over(['receiver_full_name', 'season']).alias('games_played'),
    pl.col('off_play_caller').cast(off_codes).to_physical().alias('off_play_caller_id'),
    pl.col('def_play_caller').cast(def_codes).to_physical().alias('def_play_caller_id'),
    pl.col('receiver_full_name').cast(player_codes).to_physical().alias('rec_player_id'),
    (pl.col('receiving_yards') - pl.col("receiving_yards").mean()).alias('receiving_yards_c')).join(player_exp, left_on=['receiver_player_id'], right_on='gsis_id', how = 'left').with_columns(
        (pl.col('season') - pl.col('rookie_season')).alias('number_of_seasons_played'),
        pl.col('birth_date').str.to_date().dt.year().alias('birth_year')
    ).with_columns(
        (pl.col('season') - pl.col('birth_year')).alias('age')
    ).sort(['receiver_full_name', 'season', 'game_id'])
# zero-based index
unique_games = np.sort(construct_games_played['games_played'].unique().to_numpy())
week_index = {week: i for i, week in enumerate(unique_games)}
games_idx = np.array([week_index[w] for w in construct_games_played["games_played"]])

unique_seasons = np.sort(construct_games_played['number_of_seasons_played'].unique().to_numpy())
season_index = {season: i for i, season in enumerate(unique_seasons)}
season_idx = np.array([season_index[s] for s in construct_games_played['number_of_seasons_played']])

unique_players =  construct_games_played['receiver_full_name'].unique().to_numpy()

off_play_caller_idx = construct_games_played['off_play_caller_id'].to_numpy()
def_play_caller_idx = construct_games_played['def_play_caller_id'].to_numpy()


factors_numeric = ['home_rest', 'away_rest','total', 'home_score', 'avg_depth_of_target',  'off_epa_per_play', 'def_epa_per_play', 'away_score',  'wind', 'temp', 'total_pass_attempts']

factor_data = construct_games_played.select(pl.col(factors_numeric)).with_columns(
    [
        ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std())
        for col in construct_games_played.select(factors_numeric).columns
    ]
)


players_ordered = np.array(construct_games_played['receiver_full_name'].unique().to_list())

player_idx = construct_games_played['rec_player_id'].to_numpy()


construct_games_played.group_by('season').agg(
    pl.col('receiving_yards').mean().alias('avg_receiving'),
    pl.col('receiving_yards').std().alias('std_receiving')
)

construct_games_played.select(
    pl.col('receiving_yards').mean().alias('avg_receiving'),
    pl.col('receiving_yards').std().alias('std_receiving')
)
coords = {
    'factors_num': factors_numeric,
    'gameday': unique_games,
    'seasons': unique_seasons,
    'obs_id': np.arange(len(construct_games_played)),  
    'player': unique_players,
    'off_play_caller': off_play_caller,
    'def_play_caller': def_play_caller, 
    
}



seasons_gp_prior, ax = pz.maxent(pz.Gamma(), lower = 2, upper = 6, mass = 0.99)


short_term_form, ax  = pz.maxent(pz.Gamma(), lower=1, upper = 3, mass = .95)

plt.xlim(-10, 10)

dist_nu_new, ax  = pz.maxent(pz.Gamma(), lower = 20, upper = 100, mass = 0.95)


seasons_sd = construct_games_played.group_by('season').agg(
    pl.col('receiving_yards_c').mean().alias('avg')
).select(pl.col('avg').std().alias('sd'))

seasons_sd = seasons_sd['sd'][0]

games_sd = construct_games_played.group_by('game_id').agg(
    pl.col('receiving_yards_c').mean().alias('avg')
).select(pl.col('avg').std().alias('sd'))

games_sd = games_sd['sd'][0]

player_sds = construct_games_played.group_by('receiver_full_name').agg(
    pl.col('receiving_yards_c').mean().alias('avg')
).select(pl.col('avg').std().alias('sd'))

player_sds = player_sds['sd'][0]


obs_sd = construct_games_played.select(pl.col('receiving_yards_c').std()).to_series()[0]


with pm.Model(coords = coords) as receiving_mod_long:

    gameday_id = pm.Data('gameday_id', games_idx, dims = 'obs_id')
    seasons_id = pm.Data('season_id', season_idx, dims = 'obs_id')
    
    off_id = pm.Data('off_play_caller_id', off_play_caller_idx, dims = 'obs_id')

    def_id = pm.Data('def_play_caller_id', def_play_caller_idx, dims = 'obs_id')


    x_gamedays = pm.Data("X_gamedays", unique_games, dims = 'gameday')[:,None]
    x_season = pm.Data('x_season', unique_seasons, dims = 'seasons')[:,None]

    fct_data = pm.Data('factor_num_data', factor_data.to_numpy(), dims = ('obs_id', 'factors_num'))
    
    
    player_id = pm.Data('player_id', player_idx, dims = 'obs_id')
    
    rec_obs = pm.Data('rec_obs', construct_games_played['receiving_yards_c'].to_numpy(), dims = 'obs_id')
    
    # we would expect the differnce between say 
    # a league avg wr and 
    sigma_player = pm.HalfNormal('player_sigma', sigma = player_sds * 0.5 )
    
    player_z = pm.Normal('player_z', 0 , 1, dims='player')
    
    player_effects = pm.Deterministic(
    'player_effects',
    player_z * sigma_player,
    dims='player')


    ls_games = pm.Gamma('ls_games', alpha = short_term_form.alpha, beta = short_term_form.beta)

    # Between games we may expect a difference of 60ish yards
    sigma_games = pm.HalfNormal('sigma_game', 15)

    cov_games = sigma_games **2 * pm.gp.cov.Matern52(input_dim=1, ls = ls_games)

    gp_games = pm.gp.HSGP(
        m = [12], 
        c = 1.5,
        cov_func=cov_games
        , parametrization='centered'

    )

    f_games = gp_games.prior(
        'f_games',
        X = x_gamedays,
        hsgp_coeffs_dims='basis_coeffs_games',
        dims = 'gameday'
    )

    ls_season = pm.Gamma('ls_season', **seasons_gp_prior.params_dict)
   
    # Basically for every seasons we may expect a difference of about 300 yards
    sigma_season = pm.HalfNormal('sigma_season', 80)

    cov_season = sigma_season**2 * pm.gp.cov.Matern52(1, ls = ls_season)

    gp_season = pm.gp.HSGP(
        m = [10],
        c = 1.5,
        cov_func=cov_season
        , parametrization='centered'
    )
    
    f_season = gp_season.prior(
        'f_season',
        X = x_season,
        hsgp_coeffs_dims='basis_coeffs_seasons',
        dims = 'seasons'
    )

    slope_num = pm.Normal('slope_num', sigma = 0.5, dims = 'factors_num')


    off_coach_effect = pm.Normal("slope_off", mu = 0, sigma = 1.0, dims = 'off_play_caller')
    


    alpha = pm.Deterministic(
        'alpha',
        player_effects[player_id] 
        + off_coach_effect [off_id] 
        + f_games[gameday_id]
        + f_season[seasons_id]
        , dims = 'obs_id'
    )
    
    mu_player = pm.Deterministic(
        'mu_player', alpha + pm.math.dot(fct_data, slope_num) , dims = 'obs_id'
    )
    nu = dist_nu_new.to_pymc(name = 'nu')

    # we may expect some big swings due to matchups
    # health
    # and football stuff 
    sigma_obs = pm.HalfNormal("sigma_obs", sigma= obs_sd * 1.5)

    rec_yards = pm.StudentT(
        'receiving_yards',
        nu = nu,
        mu = mu_player,
        sigma = sigma_obs,
        observed= rec_obs,
        dims = 'obs_id'
    )
    trace = pm.sample(nuts_sampler='nutpie',
                        random_seed=rng)
    p = pm.sample_prior_predictive()
    pm.sample_posterior_predictive(trace, receiving_mod_long, extend_inferencedata=True)




trace.sample_stats['diverging'].values.sum()

az.plot_trace(trace, var_names=['sigma_season', 'sigma_game', 'sigma_obs', 'ls_season', 'ls_games', 'player_sigma'])


az.plot_ppc(trace, num_pp_samples=100)
plt.xlim(-100, 250)

index = pd.MultiIndex.from_product(
    [unique_seasons,unique_games],
    names = ['number_of_seasons_played', 'gameday']
)

unique_combos = pd.DataFrame(index = index).reset_index()

f_long_post = trace.posterior['f_season']
f_games_post = trace.posterior['f_games']


f_long_post_aligned = f_long_post.sel(
    seasons = unique_combos['number_of_seasons_played'].to_numpy()
).rename({'seasons': 'timestamp'})

f_long_post_aligned['timestamp'] = unique_combos.index

f_games_post_aligned = f_games_post.sel(
    gameday = unique_combos['gameday'].to_numpy()
).rename({'gameday': 'timestamp'})

some_samps = rng.choice(4000, size = 20, replace = True)

_,axes = plt.subplot_mosaic(
    """AB""",
    figsize = (12,7.5),
    layout = 'constrained'
)


axes['A'].plot(
    f_long_post.seasons,
    az.extract(f_long_post)['f_season'].isel(sample = some_samps),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
    label="random draws",

)
az.plot_hdi(
    x = f_long_post.seasons,
    y = f_long_post,
    hdi_prob = 0.87,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9, "label": r"$87\%$ HDI"},
    ax=axes["A"],
    smooth=False,

)
axes['A'].plot(
    f_long_post.seasons,
    f_long_post.mean(('chain', 'draw')),
    color="#FBE64D",
    lw=2.5,
    label="Mean",
)
axes['A'].set(
    xlabel = 'Season',
    ylabel = 'Receiving Yards',
    title ='Between Sesason Varition'
)

axes['B'].plot(
    f_games_post.gameday,
    az.extract(f_games_post)['f_games'].isel(sample = some_samps),
    color="#70133A",
    alpha=0.3,
    lw=1.5,
)



az.plot_hdi(
    x = f_games_post.gameday,
    y = f_games_post,
    hdi_prob=0.87,
    color="#AAC4E6",
    fill_kwargs={"alpha": 0.9},
    ax=axes["B"],
    smooth=False
)

axes['B'].plot(
    f_games_post.gameday,
    f_games_post.mean(('chain', 'draw')),
    color="#FBE64D",
    lw=2.5,
)
axes['B'].set(
    xlabel = 'Games',
    ylabel='Receiving Yards',
    title = 'Between Game Variation'
)



az.plot_ess(
    trace, 
    kind='evolution',
    var_names=[RV.name for RV in receiving_mod_long.free_RVs if RV.size.eval() <=3],
    grid = (5,2),
    textsize=25
)


observed_std = construct_games_played['receiving_yards_c'].std()
ppc_samples = trace.posterior_predictive['receiving_yards'].values
ppc_std = ppc_samples.std()

print("VARIANCE CHECK:")
print(f"Observed std:  {observed_std:.1f}")
print(f"PPC std:       {ppc_std:.1f}")
print(f"Ratio:         {ppc_std/observed_std:.2f}")
print(f"{'TOO WIDE' if ppc_std > observed_std * 1.1 else 'TOO NARROW' if ppc_std < observed_std * 0.9 else 'GOOD'}")

post = trace.posterior
player_contrib = post['player_effects'].sel(player=unique_players).var(dim=['chain', 'draw']).mean().item()
games_contrib = post['f_games'].sel(gameday=unique_games).var(dim=['chain', 'draw']).mean().item()
season_contrib = post['f_season'].sel(seasons=unique_seasons).var(dim=['chain', 'draw']).mean().item()

# Get coach contributions
off_contrib = post['slope_off'].sel(off_play_caller=off_play_caller).var(dim=['chain', 'draw']).mean().item()

# Factor contributions
mu_full = post['mu_player'].var(dim=['chain', 'draw']).mean().item()
alpha_var = post['alpha'].var(dim=['chain', 'draw']).mean().item()
factor_contrib = mu_full - alpha_var

sigma_obs_posterior = post['sigma_obs'].mean().item()

print("\nVARIANCE DECOMPOSITION:")
print(f"Player effects:    {player_contrib:>6.1f}")
print(f"Games GP:          {games_contrib:>6.1f}")
print(f"Season GP:         {season_contrib:>6.1f}")
print(f"Off coach:         {off_contrib:>6.1f}")
print(f"Factors:           {factor_contrib:>6.1f}")
print(f"Sigma_obs:         {sigma_obs_posterior**2:>6.1f}")
print(f"{'─'*30}")
print(f"Total systematic:  {player_contrib + games_contrib + season_contrib + off_contrib + factor_contrib:>6.1f}")
print(f"Observation noise: {sigma_obs_posterior**2:>6.1f}")
print(f"Data variance:     {observed_std**2:>6.1f}")

# 3. Check nu (Student-t degrees of freedom)
nu_posterior = post['nu'].mean().item()
print(f"\nStudent-t nu:      {nu_posterior:.1f}")
if nu_posterior < 10:
    print("⚠ Very heavy tails - might be TOO flexible")
elif nu_posterior > 50:
    print("⚠ Almost Normal - Student-t not needed")

# 4. Check if sigma_obs is hitting its prior boundary
print(f"\nSigma_obs prior:   {obs_sd * 0.95:.1f}")
print(f"Sigma_obs post:    {sigma_obs_posterior:.1f}")
if sigma_obs_posterior > obs_sd * 0.9:
    print("⚠ Sigma_obs at prior boundary - INCREASE THE PRIOR!")