library(rvest)
library(tidyverse)

url = 'https://theringer.com/fantasy-football/2025'


session = read_html_live(url)

session$view()


rank = session |>
  html_elements(
    xpath = '//*[@id="main-content"]/div/div[2]/div/div/div/div/div/button/div[1]'
  ) |>
  html_text()

player = session |>
  html_elements(
    xpath = '//*[@id="main-content"]/div/div[2]/div/div/div/div/div/button/div[3]'
  ) |>
  html_text()

ringer_rankings = bind_cols(player, rank) |>
  rename(
    player = `...1`,
    ringer_ranking = `...2`
  ) |>
  mutate(ringer_ranking = as.numeric(ringer_ranking))

write_csv(ringer_rankings, 'data/ringer-ff-rankings.csv')
