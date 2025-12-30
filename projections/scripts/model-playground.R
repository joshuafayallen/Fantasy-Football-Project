library(arrow)
library(modeltime)
library(nflreadr)
library(MetBrewer)
library(mirai)
library(bonsai)
library(timetk)
library(tidymodels)
library(tidyverse)

# instead of worrying to much about the time stuff
# Looking at this model https://github.com/richjand/nfl-run-pass/blob/master/pass-prob.Rmd
# it looks like richard is just ignoring the time dynamics from play to play

raw_data = load_player_stats(seasons = c(1999:2024)) |>
  filter(position_group %in% c('WR', 'TE', 'RB', 'QB'), season_type == 'REG') |>
  select(
    -starts_with('def'),
    -c(penalties, penalty_yards),
    -starts_with("fg"),
    -starts_with('gwfg'),
    -starts_with('pat')
  ) |>
  mutate(era = ifelse(season >= 2018, 'Post-2018', 'Pre-2018'))

# dir.create('model_data')

write_parquet(
  raw_data,
  'model_data/fantasy-points-data.parquet'
)

vars = raw_data |>
  select(
    -c(
      fantasy_points,
      fantasy_points_ppr,
      player_id,
      season,
      season_type,
      team,
      opponent_team,
      week,
      season,
      headshot_url,
      position,
      sacks_suffered,
      fumble_recovery_yards_own,
      fumble_recovery_opp,
      fumble_recovery_yards_opp,
      fumble_recovery_tds,
      rushing_first_downs,
      passing_first_downs,
      receiving_first_downs,
      rushing_first_downs
    ),
    -ends_with('name'),
    -c(air_yards_share, wopr)
  ) |>
  colnames()

pts_form = reformulate(vars, response = 'fantasy_points')


init_split = initial_split(raw_data, strata = position_group)

fantasy_train = training(init_split)
fantasy_test = testing(init_split)
fantasy_folds = vfold_cv(fantasy_train)


fantasy_rec = recipe(
  pts_form,
  data = fantasy_train
) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE)


fantasy_mod = boost_tree(
  mtry = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = tune()
) |>
  set_engine('xgboost') |>
  set_mode('regression')


xgb_grid = grid_space_filling(
  finalize(mtry(), fantasy_train),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  sample_size = sample_prop(),
  stop_iter(),
  min_n(),
  size = 40,
  type = 'latin_hypercube'
)


fantasy_wf = workflow() |>
  add_recipe(fantasy_rec) |>
  add_model(fantasy_mod)


xgb_res = tune_grid(
  fantasy_wf,
  resamples = fantasy_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)


best_mod = xgb_res |>
  select_best()


fantasy_mod = finalize_workflow(fantasy_wf, best_mod) |>
  last_fit(init_split)


preds = fantasy_mod |>
  collect_predictions()

preds |>
  pivot_longer(c(.pred, fantasy_points)) |>
  ggplot(aes(x = value, fill = name)) +
  geom_density(alpha = 0.5) +
  scale_fill_met_d(name = 'Lakota')


ggplot(preds, aes(x = .pred, y = fantasy_points)) +
  geom_point()


## lets train two more models
## lets do a lstm neural net
## a lightgbm  and a random forest

rand_forest_fantasy = rand_forest(
  trees = tune(),
  min_n = tune()
) |>
  set_engine('ranger') |>
  set_mode('regression')


rand_forest_wf = workflow() |>
  add_recipe(fantasy_rec) |>
  add_model(rand_forest_fantasy)


rand_forest_grid = grid_space_filling(
  trees(),
  min_n(),
  size = 30,
  type = 'latin_hypercube'
)

rand_forest_tune = tune_grid(
  rand_forest_wf,
  grid = rand_forest_grid,
  resamples = fantasy_folds,
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

best_rand_forest = rand_forest_tune |>
  select_best()

final_rand_forest_fantasy = finalize_workflow(
  rand_forest_wf,
  best_rand_forest
) |>
  last_fit(init_split)


preds = final_rand_forest_fantasy |>
  collect_predictions()

preds |>
  pivot_longer(c(.pred, fantasy_points)) |>
  ggplot(aes(x = value, fill = name)) +
  geom_density(alpha = 0.5) +
  scale_fill_met_d(name = 'Lakota')


ggplot(preds, aes(x = .pred, y = fantasy_points)) +
  geom_point()


fantasy_mod_lgbm = boost_tree(
  mtry = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = tune()
) |>
  set_engine("lightgbm") |>
  set_mode('regression')


lgbm_grid = grid_space_filling(
  finalize(mtry(), fantasy_train),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  sample_size = sample_prop(),
  stop_iter(),
  min_n(),
  size = 40,
  type = 'latin_hypercube'
)


fantasy_wf_lgbm = workflow() |>
  add_recipe(fantasy_rec) |>
  add_model(fantasy_mod_lgbm)

light_gbm_tune = tune_grid(
  fantasy_wf_lgbm,
  grid = lgbm_grid,
  resamples = fantasy_folds,
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)


best_lgbm = light_gbm_tune |>
  select_best()

final_lgbm = finalize_workflow(fantasy_wf_lgbm, best_lgbm) |>
  last_fit(init_split)


final_lgbm |>
  collect_predictions() |>
  pivot_longer(c(.pred, fantasy_points)) |>
  ggplot(aes(x = value, fill = name)) +
  geom_density(alpha = 0.5) +
  scale_fill_met_d(name = 'Lakota')


## we

model_list = list(
  'Light GBM' = final_lgbm,
  'Random Forest' = final_rand_forest_fantasy,
  'XGBoost' = fantasy_mod
)


model_comparision = map_df(model_list, collect_metrics, .id = 'model')

# It looks like the LightGBM model is the best model by RMSE
ggplot(
  model_comparision,
  aes(x = model, y = .estimate, color = .metric)
) +
  geom_point()


check_predictions = map_df(model_list, collect_predictions, .id = 'model') |>
  mutate(residual = fantasy_points - .pred)


ggplot(check_predictions, aes(x = residual, fill = model)) +
  geom_density() +
  facet_wrap(vars(model))

ggplot(check_predictions, aes(x = .pred, y = residual, color = model)) +
  geom_point() +
  geom_smooth() +
  facet_wrap(vars(model))
