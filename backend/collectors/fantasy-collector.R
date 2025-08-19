## this is really just a wrapper around fantasy pros
##

library(nflverse)
library(arrow)
library(ffanalytics)
library(tidyverse)

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

roster_data = load_players()

clean_up_roster_data = roster_data |>
  filter(position_group %in% c('QB', 'TE', 'RB', "WR")) |>
  select(display_name, years_of_experience, rookie_season, birth_date) |>
  mutate(birth_date = ymd(birth_date))

# this is getting mad
add_roster_info = just_redraft |>
  left_join(clean_up_roster_data, join_by(player == display_name))

arrow::write_parquet(
  add_roster_info,
  glue::glue('data/ffverse-data-{Sys.Date()}.parquet')
)
