library(dplyr)

t = nflreadr::load_teams() |>
  purrr::pluck('team_abbr')

cols = nflplotR:::primary_colors

color_tib = data.frame(
  team = names(cols),
  color = unname(cols)
) |>
  mutate(
    cols = case_match(
      team,
      'LAR' ~ 'LA',
      'STL' ~ 'LA',
      'SD' ~ 'LAC',
      'OAK' ~ 'LV',
      .default = team
    ),
    combos = glue::glue("'{cols}':'{color}',")
  ) |>
  distinct(team, .keep_all = TRUE)


color_tib$combos
