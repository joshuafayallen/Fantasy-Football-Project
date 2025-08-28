import polars as pl 


def process_common_fields(df):
    processed = df.with_columns(
    pl.when(
        (pl.col("posteam_type") == "away") & (pl.col("spread_line") <= 0)
    )
    .then((pl.col("total_line") + pl.col("spread_line")) / 2 - pl.col("spread_line"))
    .when(
        (pl.col("posteam_type") == "away") & (pl.col("spread_line") > 0)
    )
    .then((pl.col("total_line") - pl.col("spread_line")) / 2)
    .when(
        (pl.col("posteam") == "home") & (pl.col("spread_line") <= 0)
    )
    .then((pl.col("total_line") - pl.col("spread_line")) / 2)
    .when(
        (pl.col("posteam") == "home") & (pl.col("spread_line") > 0)
    )
    .then((pl.col("total_line") + pl.col("spread_line")) / 2 - pl.col("spread_line"))
    .otherwise(None)
    .alias("implied_total")).with_columns(
    pl.when(
        (pl.col('surface') == 'grass')
    )
    .then(pl.lit('grass'))
    .otherwise(pl.lit('turf'))
    .alias('surface'),
    pl.when(
        (pl.col('roof').is_in(['closed', 'dome']))
    )
    .then(pl.lit('indoors'))
    .otherwise(pl.lit('outdoors'))
    .alias('roof'),
    pl.when(
        (pl.col('roof').is_in(['closed', 'dome']))
    )
    .then(pl.lit(68))
    .when(pl.col('temp').is_null())
    .then(pl.lit(60))
    .otherwise(pl.col('temp'))
    .alias('temp'),
    pl.when(
        (pl.col('roof').is_in(['closed', 'dome']))
    )
    .then(pl.lit(0))
    .when(pl.col('wind').is_null())
    .then(pl.lit(8))
    .otherwise(pl.col('wind'))
    .alias('wind'),
    pl.when(
        pl.col('season') >= 2018
    )
    .then(pl.lit('post_2018'))
    .otherwise(pl.lit('pre_2018'))
    .alias('era'),
    pl.when(
        pl.col('two_point_attempt') == 1
    )
    .then(pl.lit(4))
    .otherwise(pl.col('down'))
    .alias('down'),
    pl.when(
        pl.col('two_point_attempt') == 1
    )
    .then(pl.lit(0))
    .otherwise(pl.col('rushing_yards'))
    .alias('rushing_yards'),
    pl.when(
        pl.col('two_point_attempt') == 1
    )
    .then(pl.lit(0.75))
    .otherwise(pl.col('xpass'))
    .alias('xpass'),
    pl.when(
        pl.col('pass_location').is_not_null()
    )
    .then(pl.col('pass_location'))
    .when(pl.col('desc').str.contains(r' left'))
    .then(pl.lit('left'))
    .when(pl.col('desc').str.contains(r' right'))
    .then(pl.lit('right'))
    .when(pl.col('desc').str.contains(r" middle"))
    .then(pl.lit('middle'))
    .otherwise(pl.lit('unk'))
    .alias('pass_location'),
    pl.when(
        pl.col('two_point_attempt') == 1 
    )
    .then(pl.lit(0))
    .otherwise(pl.col('yards_after_catch'))
    .alias('yards_after_catch'),
    pl.when(
        pl.col('two_point_attempt') == 1
    )
    .then(pl.col('yardline_100'))
    .otherwise(pl.col('air_yards'))
    .alias('air_yards'),
    pl.when(
        pl.col('two_point_conv_result') == 'success'
    )
    .then(pl.lit(1))
    .when((pl.col('two_point_conv_result').is_null()) & (pl.col('desc').str.contains(r"ATTEMPT SUCCEEDS")))
    .then(1)
    .otherwise(pl.lit(0))
    .alias('two_point_converted'))
    return processed