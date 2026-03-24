library(nflreadr)
library(tidyverse)


df = load_participation(
  seasons = c(2017:2024),
  include_pbp = TRUE
) |>
  filter(play_type_nfl %in% c('PASS', 'RUSH'), season_type == 'REG')


table(df$season_type)

play_callers = read_csv(
  "https://raw.githubusercontent.com/samhoppen/NFL_public/2f60ca7a84880f63c4349e5e05d3990a66d13a30/data/all_playcallers.csv"
) |>
  select(
    season,
    team,
    week,
    game_id,
    off_play_caller
  )

add_play_callers = df |>
  inner_join(
    play_callers,
    join_by(nflverse_game_id == game_id, season, week, possession_team == team)
  ) |>
  filter(
    off_play_caller %in%
      c(
        'Kyle Shanahan',
        'Sean McVay',
        'Ben Johnson',
        'Josh McDaniels',
        "Matt LaFleur",
        'Mike McDaniel',
        "Kevin O'Conell",
        'Todd Monken',
        'Andy Reid',
        'Joe Brady',
        'Brian Daboll',
        "Joe Brady",
        'Ken Dorsey',
        'Mike McCarthy'
      )
  )

write_csv(add_play_callers, 'raw_data/participation-small.csv')


make_outcomes = add_play_callers |>
  filter(nchar(offense_personnel) > 1) |>
  mutate(
    n_tes = str_extract(offense_personnel, "\\d+ TE"),
    n_rbs = str_extract(offense_personnel, "\\d+ RB"),
    across(c(n_tes, n_rbs), \(x) ifelse(is.na(x), 0, x)),
    across(c(n_tes, n_rbs), \(x) str_remove_all(x, "TE|RB| ")),
    personnel_grouping = glue::glue("{n_rbs}{n_tes}_personnel"),
    is_explosive = case_when(
      play_type_nfl == 'RUN' & rushing_yards >= 10 ~ 1,
      play_type_nfl == 'PASS' & receiving_yards >= 20 ~ 1,
      .default = 0
    ),
    offense = ifelse(possession_team == home_team, home_team, away_team),
    defense = ifelse(defteam == home_team, home_team, away_team)
  ) |>
  group_by(season, nflverse_game_id, off_play_caller, play_type_nfl) |>
  mutate(
    success_rate = mean(success, na.rm = TRUE),
    explosive_play_rate = mean(is_explosive, na.rm = TRUE),
    avg_epa = mean(epa, na.rm = TRUE),
    avg_defenders_in_box = mean(defenders_in_box, na.rm = TRUE),
    total_game_snaps = n()
  ) |>

  pivot_wider(
    names_from = personnel_grouping,
    values_from = personnel_grouping,
    values_fn = length,
    values_fill = 0
  ) |>
  summarise(across(everything(), first), .groups = "drop") |>
  select(-c(offense_personnel, defense_personnel)) |>
  mutate(
    across(
      ends_with("_personnel"),
      \(x) x / total_game_snaps,
      .names = "share_{.col}"
    ),
    is_off_home_game = ifelse(offense == home_team, 1, 0)
  )


arrow::write_parquet(make_outcomes, 'processed-data/processed-dat.parquet')

## lets look at the distribution of offensive formations
## generally I know that kyle uses 21 a ton and sean uses 11 a lot.
