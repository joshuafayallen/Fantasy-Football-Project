import scripts.process_common_fields as common
import scripts.process_passers_df as passers
import scripts.process_rush_df as rush
import polars as pl 


class ep_process:
    """
    Mimics https://github.com/ffverse/ffopportunity/blob/main/R/ep_preprocess.R
    using Polars
    """

    def __init__(
        self,
        pbp_data: pl.DataFrame | None = None,
        rosters_data: pl.DataFrame | None = None,
    ):
        self.pbp_data = pbp_data
        self.rosters_data = rosters_data

    def process_all_data(
        self,
        pbp_data: pl.DataFrame | None = None,
        rosters_data: pl.DataFrame | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Runs the full preprocessing pipeline and returns
        the processed dataframes as a dict.
        """

        pbp_data = pbp_data if pbp_data is not None else self.pbp_data
        rosters_data = rosters_data if rosters_data is not None else self.rosters_data
        

        if pbp_data is None or rosters_data is None:
            raise ValueError(
                "Must provide the results of nfl_data_py.import_pbp_data() "
                "or nfl_data_py.import_seasonal/weekly_rosters()"
            )

        processed_common_fields = common.process_common_fields(df=pbp_data)
        processed_rush = rush.process_rush_df(
            df=processed_common_fields,
            rosters=rosters_data
        )
        processed_passers = passers.process_passers(
            df=processed_common_fields,
            rosters=rosters_data
        )

        return {
            "processed_common_fields": processed_common_fields,
            "processed_rushers": processed_rush, 
            "processed_passers": processed_passers,
        }

