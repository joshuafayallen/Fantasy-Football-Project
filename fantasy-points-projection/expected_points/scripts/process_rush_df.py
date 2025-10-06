import polars as pl

def process_rush_df(df, rosters):
    d = df.filter((pl.col('play_type') == 'run') | (pl.col('play_type') == 'qb_kneel')).join(rosters, left_on = ['rusher_player_id', 'season'], right_on=['gsis_id', 'season'], nulls_equal=False, how = 'left').with_columns(
    pl.when(
        (pl.col('run_location').is_not_null())
    )
    .then(pl.col('run_location'))
    .when(pl.col('desc').str.contains(r" left"))
    .then(pl.lit('left'))
    .when(pl.col('desc').str.contains(r' right'))
    .then(pl.lit('right'))
    .when(pl.col('desc').str.contains(r' middle'))
    .then(pl.lit('middle'))
    .otherwise(pl.lit('unk'))
    .alias('run_location'),
    pl.when(
        pl.col('run_gap').is_not_null()
    )
    .then(pl.col('run_gap'))
    .when(pl.col('desc').str.contains(r" end"))
    .then(pl.lit('end'))
    .when(pl.col('desc').str.contains(r" tackle"))
    .then(pl.lit('tackle'))
    .when(pl.col('desc').str.contains(r" guard"))
    .then(pl.lit('guard'))
    .when(pl.col('desc').str.contains(r" middle"))
    .then(pl.lit('guard'))
    .otherwise(pl.lit('unk'))
    .alias('run_gap')).with_columns(
    pl.concat_str(
        [
            pl.col('run_location'),
            pl.lit('_'),
            pl.col('run_gap')
        ]
    ).alias('run_gap_dir'),
    pl.when(
        pl.col('rush_touchdown') == 1
    )
    .then(pl.lit('1'))
    .otherwise(pl.lit('0'))
    .alias('rush_touchdown'),
    pl.when(
        pl.col('first_down') == 1
    )
    .then(pl.lit('0'))
    .otherwise(pl.lit('0'))
    .alias('first_down')).select(
    pl.col(
        'game_id',
        'play_id',
        'desc',
        'rusher_player_id',
        'full_name',
        'posteam',
        'defteam',
        'two_point_attempt',
        'two_point_converted',
        'rush_attempt',
        'first_down_rush',
        'fumble_lost',
        'season',
        'week',
        'rushing_yards',
        'rush_touchdown',
        'first_down',
        'posteam_type',
        'run_location',
        'run_gap',
        'run_gap_dir',
        'surface',
        'wind',
        'temp',
        'roof',
        'position',
        'yardline_100',
        'half_seconds_remaining',
        'game_seconds_remaining',
        'fixed_drive',
        'era',
        'xpass',
        'qtr',
        'down',
        'goal_to_go',
        'ydstogo',
        'shotgun',
        'no_huddle',
        'qb_dropback',
        'qb_scramble',
        'score_differential',
        'epa',
        'vegas_wp',
        'implied_total'
    ))
    return d