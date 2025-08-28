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

roster_data$position |> table()

roster_data |>
  filter(str_detect(display_name, "Tre")) -> check


clean_up_roster_data = roster_data |>
  filter(
    position %in%
      c('QB', 'TE', 'RB', "WR", 'K') |
      display_name == 'Travis Hunter'
  ) |>
  select(display_name, years_of_experience, rookie_season, birth_date) |>
  mutate(birth_date = ymd(birth_date), ) |>
  distinct(display_name, .keep_all = TRUE) |>
  mutate(
    display_name = case_when(
      display_name == 'Patrick Mahomes' ~ 'Patrick Mahomes II',
      display_name == 'Aaron Jones' ~ 'Aaron Jones Sr.',
      display_name == 'Travis Etienne' ~ 'Travis Etienne Jr.',
      display_name == 'Michael Pittman' ~ 'Michael Pittman Jr.',
      display_name == 'Brian Robinson' ~ 'Brian Robinson Jr.',
      display_name == 'Anthony Richardson' ~ 'Anthony Richardson Sr.',
      display_name == 'Chris Godwin Jr.' ~ 'Chris Godwin',
      display_name == 'Josh Palmer' ~ 'Joshua Palmer',
      display_name == 'Kyle Pitts' ~ 'Kyle Pitts Sr.',
      display_name == 'Audric Estimé' ~ 'Audric Estime',
      display_name == 'Tre Harris' ~ "Tre' Harris",
      .default = display_name
    )
  )

add_roster_info = just_redraft |>
  left_join(clean_up_roster_data, join_by(player == display_name))


arrow::write_parquet(
  add_roster_info,
  glue::glue('data/ffverse-data-{Sys.Date()}.parquet')
)
