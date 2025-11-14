import polars as pl
import polars.selectors as cs


posteriors = pl.scan_parquet("writeup-dat/model_stats.parquet")


long_data = (
    posteriors.filter(pl.int_range(pl.len()).shuffle().over(["chain"]) < 500)
    .rename({"chain": ".chain", "draw": ".iteration"})
    .unpivot(
        cs.numeric().exclude([".iteration", ".chain"]), index=[".iteration", ".chain"]
    )
    .with_columns(
        pl.col("variable").str.replace_all(r"_posterior_", ""),
        pl.row_index().alias(".draw"),
    )
    .with_columns(
        pl.col("variable")
        .str.extract(
            r"(eta|player_effect|baseline|slope|delta_mean|sd|player_delta|cutpoint|basis_coeffs_game|basis_coeffs_season|baseline|alpha|tds_scored_probs|eta|player_effects_raw|gps_sigma|f_games|f_season|delta_sig|ls)"
        )
        .alias("param"),
        pl.col("variable").str.extract(r"(\d+)").alias("obs_id"),
    )
    .collect()
)


long_data.write_parquet(
    "trace-plot-data", use_pyarrow=True, pyarrow_options={"partition_cols": ["param"]}
)
