library(AllenMisc)
library(arrow)
library(ggdist)
library(nflreadr)
library(nflplotR)
library(tidybayes)
library(tidyverse)

## this is super big so i think it will just be easier to use it a database
## we can do a lot of the plotting in ggplot
## the problem is that we ar going to have to move back and forth
pymc_post = open_dataset(here::here(
  'rec-tds-posterior',
  'rec-tds-posterior.parquet'
))

implied_probs = open_dataset(here::here(
  'rec-tds-posterior',
  'implied_probs_posterior.parquet'
))

player_stats = load_player_stats(
  seasons = c(2014:2024),
  summary_level = 'reg'
) |>
  filter(
    position_group %in% c('RB', 'TE', 'WR')
  ) |>
  select(
    receiver_full_name = player_display_name,
    gsis_id = player_id,
    season,
    receiving_tds,
    position_group
  )

replacement_level = player_stats |>
  mutate(
    ranks = rank(receiving_tds, ties.method = 'first'),
    .by = c(season, position_group)
  ) |>
  # this is probably a bit eager
  filter(ranks %in% c(1:5), season == 2023) |>
  pull(receiver_full_name)


elite = player_stats |>
  mutate(
    ranks = rank(-receiving_tds, ties.method = 'first'),
    .by = c(season, position_group)
  ) |>
  filter(ranks %in% c(1:5), season == 2023) |>
  pull(receiver_full_name)

elite_df = pymc_post |>
  filter(receiver_full_name %in% elite, season == 2023) |>
  summarise(
    rec_tds_probs = mean(tds_scored),
    .by = c(receiver_full_name, draw)
  ) |>
  collect()


replacement_df = pymc_post |>
  filter(receiver_full_name %in% replacement_level, season == 2023) |>
  group_by(draw) |>
  summarise(rep_tds_probs = mean(tds_scored), .groups = 'drop') |>
  collect()


par_df = elite_df |>
  left_join(replacement_df, join_by(draw)) |>
  mutate(par = rec_tds_probs - rep_tds_probs) |>
  # this looks gross
  left_join(
    player_stats |>
      filter(season == 2023) |>
      select(receiver_full_name, position_group),
    join_by(receiver_full_name)
  )

# right now this looks pretty nice in a full screen context
ggplot(
  par_df,
  aes(
    x = par,
    y = receiver_full_name,
    fill = position_group,
    color = position_group
  )
) +
  stat_halfeye(alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = 'dashed') +
  facet_wrap(vars(position_group), scales = 'free_x') +
  MetBrewer::scale_color_met_d(name = 'Lakota') +
  MetBrewer::scale_fill_met_d(name = 'Lakota') +
  scale_y_discrete(labels = scales::label_wrap(10)) +
  labs(y = NULL, x = 'Touchdowns') +
  theme_allen_minimal(base_size = 20) +
  theme(legend.position = 'none')


just_calvin_manual = implied_probs |>
  select(-contains('index')) |>
  filter(receiver_full_name == 'Calvin Johnson') |>
  collect() |>
  group_by(event) |>
  median_qi(
    tds_scored_probs,
    .width = c(.80, .95)
  )

ggplot(just_calvin_manual, aes(x = event, y = tds_scored_probs)) +
  geom_pointrange(
    data = filter(just_calvin, .width == 0.8),
    aes(ymin = .lower, ymax = .upper),
    linewidth = 1.5
  ) +
  geom_pointrange(
    data = filter(just_calvin, .width == 0.95),
    aes(ymin = .lower, ymax = .upper),
    linewidth = 0.5
  )

check = implied_probs |>
  collect()


implied_probs |>
  select(-contains('index')) |>
  filter(
    receiver_full_name %in%
      c('Calvin Johnson', 'Rob Gronkowski', 'Christian McCaffrey')
  ) |>
  collect() |>
  ggplot(aes(x = event, y = tds_scored_probs)) +
  stat_pointinterval() +
  labs(x = "Touchdowns", y = 'Estimated Probabilities') +
  facet_wrap(vars(receiver_full_name)) +
  theme_allen_minimal()


## lets get an eyechekc on Gronk versus a good tightend
## and a blocking tightend

implied_probs |>
  filter(
    receiver_full_name %in%
      c("Rob Gronkowski", 'Mark Andrews', 'Luke Farrell')
  ) |>
  collect() |>
  mutate(
    receiver_full_name = as_factor(receiver_full_name),
    receiver_full_name = fct_relevel(
      receiver_full_name,
      'Rob Gronkowski',
      'Mark Andrews',
      'Luke Farrell'
    )
  ) |>
  ggplot(aes(x = event, y = tds_scored_probs)) +
  stat_pointinterval() +
  labs(x = "Touchdowns", y = 'Estimated Probabilities') +
  facet_wrap(vars(receiver_full_name)) +
  theme_allen_minimal()
