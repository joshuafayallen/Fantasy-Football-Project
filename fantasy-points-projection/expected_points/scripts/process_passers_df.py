import polars as pl

def process_passers(df, rosters):
    d = df.filter(
    (pl.col('play_type') == 'pass') | (pl.col('play_type') == 'qb_spike')).join(rosters, left_on = ['passer_player_id', 'season'], right_on = ['gsis_id', 'season'], how = 'left').rename(
    { 
        "position": 'passer_position',
        'birth_date': 'passer_birth_date',
        'full_name': 'passer_full_name',
        'jersey_number_right': 'qb_jersey_number',
        'week_right': 'qb_week'
    }).join(rosters, left_on = ['receiver_player_id', 'season'], right_on = ['gsis_id', 'season'], how = 'left').rename(
    {
        'position': 'receiver_position',
        'birth_date': 'receiver_birth_date',
        'full_name': 'receiver_full_name'
    }).with_columns(
    (pl.col('air_yards') - pl.col('ydstogo')).alias('relative_to_sticks'),
    (pl.col('air_yards')- pl.col('yardline_100')).alias('relative_to_endzone'),
    pl.when(
        pl.col('complete_pass') == 1
    )
    .then(pl.lit('1'))
    .otherwise(pl.lit('0'))
    .alias('complete_pass'),
    pl.when(
        pl.col('pass_touchdown') == 1
    )
    .then(pl.lit('1'))
    .otherwise(pl.lit('0'))
    .alias('pass_touchdown'), 
    pl.when(pl.col('first_down') == 1)
    .then(pl.lit('1'))
    .otherwise(pl.lit('0'))
    .alias('first_down'),
    pl.when(
        pl.col('interception') == 1
    )
    .then(pl.lit('1'))
    .otherwise(pl.lit('0'))
    .alias('interception')).with_columns(
    pl.col('air_yards').fill_null(0)).filter(
    pl.col('sack') == 0).select(
    pl.col(
        'game_id',
        'play_id',
        'desc',
        'passer_player_id',
        'passer_full_name',
        'passer_position',
        'receiver_player_id',
        'receiver_full_name',
        'receiver_position',
        'posteam',
        'defteam',
        'two_point_attempt',
        'two_point_converted',
        'pass_attempt',
        'receiving_yards',
        'first_down_pass',
        'fumble_lost',
        'season',
        'week',
        'season',
        'complete_pass',
        'yards_after_catch',
        'pass_touchdown',
        'first_down',
        'interception',
        'relative_to_endzone',
        'wind',
        'score_differential',
        'xpass',
        'vegas_wp',
        'total_line',
        'implied_total',
        'relative_to_sticks',
        'air_yards',
        'yardline_100',
        'half_seconds_remaining',
        'game_seconds_remaining',
        'epa',
        'fixed_drive',
        'ydstogo',
        'temp',
        'era',
        'qb_hit',
        'posteam_type',
        'pass_location',
        'surface',
        'roof',
        'passer_position',
        'receiver_position',
        'qtr',
        'down',
        'goal_to_go',
        'shotgun',
        'no_huddle'))
    return d 
