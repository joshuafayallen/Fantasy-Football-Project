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
        play_caller_data: pl.DataFrame | str = (
            "https://raw.githubusercontent.com/samhoppen/NFL_public/"
            "2f60ca7a84880f63c4349e5e05d3990a66d13a30/data/all_playcallers.csv"
        )
    ):
        self.pbp_data = pbp_data
        self.rosters_data = rosters_data
        self.play_caller_data = play_caller_data

    def process_all_data(
        self,
        pbp_data: pl.DataFrame | None = None,
        rosters_data: pl.DataFrame | None = None,
        play_caller_data: pl.DataFrame | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Runs the full preprocessing pipeline and returns
        the processed dataframes as a dict.
        """

        pbp_data = pbp_data if pbp_data is not None else self.pbp_data
        rosters_data = rosters_data if rosters_data is not None else self.rosters_data
        play_caller_data = play_caller_data if play_caller_data is not None else self.play_caller_data

        play_caller_data = pl.read_csv(play_caller_data)

        off_play_caller = play_caller_data.select(
            pl.col("team", 'off_play_caller', 'game_id')
        )
        def_play_caller = play_caller_data.select(
            pl.col('team', 'def_play_caller', 'game_id')
        )
        

        if pbp_data is None or rosters_data is None:
            raise ValueError(
                "Must provide the results of nfl_data_py.import_pbp_data() "
                "or nfl_data_py.import_seasonal/weekly_rosters()"
            )

        processed_common_fields = common.process_common_fields(df=pbp_data)

        processed_rush = rush.process_rush_df(
            df=processed_common_fields,
            rosters=rosters_data
        ).join(off_play_caller, left_on=['game_id', 'posteam'], right_on=['game_id', 'team']).join(def_play_caller, left_on = ['game_id', 'defteam'], right_on=['game_id', 'team'])

        processed_passers = passers.process_passers(
            df=processed_common_fields,
            rosters=rosters_data
        ).join(off_play_caller, left_on=['game_id', 'posteam'], right_on=['game_id', 'team']).join(def_play_caller, left_on = ['game_id', 'defteam'], right_on=['game_id', 'team'])


        return {
            "processed_common_fields": processed_common_fields,
            "processed_rushers": processed_rush, 
            "processed_passers": processed_passers,
        }

