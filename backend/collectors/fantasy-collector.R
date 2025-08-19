## this is really just a wrapper around fantasy pros
##

library(nflverse)
library(arrow)
library(tidyverse)
library(ffanalytics)


fantasy_data_raw = load_ff_rankings(type = 'draft')

just_redraft = fantasy_data_raw |>
  filter(
    ecr_type %in%
      c('ro') &
      !pos %in% c('DB', 'DL', 'LB') &
      tm != 'FA'
  ) |>
  select(
    id,
    player,
    pos,
    tm,
    ecr,
    sd,
    best,
    worst,
    bye,
    rank_delta,
    scrape_date,
    player_owned_avg
  ) |>
  # this is just to make it more readable to me
  rename(avg_expert_ranking = ecr, sd_expert_ranking = sd) |>
  mutate(scrape_date = ymd(Sys.Date()))


arrow::write_parquet(
  just_redraft,
  glue::glue('data/ffverse-data-{Sys.Date()}.parquet')
)


fantasy_data_raw
