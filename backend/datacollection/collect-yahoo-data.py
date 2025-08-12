import requests
import polars as pl
import polars.selectors as cs


url = "https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2/league/461.l.public;out=settings/players;position=ALL;start=0;count=400;sort=average_pick;search=;out=auction_values,ranks;ranks=season;ranks_by_position=season;out=expert_ranks;expert_ranks.rank_type=projected_season_remaining/draft_analysis;cut_types=diamond;slices=last7days?format=json_f"

headers = {
    "Accept": "*/*",
    "Host": "pub-api-ro.fantasysports.yahoo.com",
    "Origin": "https://football.fantasysports.yahoo.com",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "none",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "Josh Allen PhD"
}


response = requests.get(url, headers = headers)


response.raise_for_status()


data = response.json()

players_data = data['fantasy_content']['league']['players']

player_list = []

for player_wrapper in players_data:
    player = player_wrapper.get("player")
    if not player:
        continue  # skip empty entries

    try:

        player_info = {
            'full_name': player['name']['full'],
            'first_name': player['name']['first'],
            'last_name': player['name']['last'],
            'team_abr': player.get('editorial_team_abbr'),
            'bye_week': player.get('bye_weeks', {}).get('week'),
            'position': player['display_position'],
            'projected_auction_value': player.get('projected_auction_value'),
            'average_pick_yahoo': player['draft_analysis'].get('average_pick'),
            'average_cost_yahoo': player['draft_analysis'].get('average_cost'),
            'yahoo_rank_2025': player['player_ranks'][0]['player_rank']['rank_value']
        }
        player_list.append(player_info)
    except Exception as e:
        print(f'Skipping a player due to missing data: {e}')



yahoo_fantasy_2025 = pl.from_dicts(player_list)

fp24 = pl.read_csv('data/fpros-scoring-2024.csv').select(pl.col('Player'), pl.col('AVG').alias('avg_2024'), pl.col('TTL').alias('total_2024'))

fp23 = pl.read_csv('data/fpros-scoring-2023.csv').select(pl.col('Player'), pl.col('AVG').alias('avg_2023'), pl.col('TTL').alias('total_2023'))


add_points = yahoo_fantasy_2025.join(fp24, left_on=['full_name'], right_on=['Player'], how='left').join(fp23, left_on=['full_name'], right_on=['Player'], how = 'left')

yahoo_fantasy_2025.write_parquet('data/yahoo_fantasy.parquet')
