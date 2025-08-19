import polars as pl
import polars.selectors as cs
import datetime
import janitor.polars
import os


wr_advc_stats_raw = pl.scan_csv('data/fpros_advanced_stats_wr-*.csv', include_file_paths='year').collect().clean_names()

te_advc_stats_raw = pl.scan_csv('data/fpros_advanced_stats_te-*.csv', include_file_paths='year').collect().clean_names()

rb_advc_stats_raw = pl.scan_csv('data/fpros_advanced_stats_rb-*.csv', include_file_paths='year').collect().clean_names()

qb_advc_stats_raw = pl.scan_csv('data/fpros_advanced_stats_qb-*.csv', include_file_paths='year').collect().clean_names()



clean_up_wr_names = wr_advc_stats_raw.rename(
    {'g': 'games_played',
    "rec": 'receptions',
    'y_r': 'yards_per_reception',
    'ybc': 'yards_before_contact_rc',
    'ybc_r': 'yards_before_contact_per_reception',
    'air': 'air_yards_receiving',
    'air_r': 'air_yards_per_reception', 
    'yacon': 'yac_per_reception',
    '%_tm' : 'target_share',
    '10+_yds': 'reception_10_plus_yds',
    '20+_yds': 'reception_20_plus_yds',
    '30+_yds': 'reception_30_plus_yds', 
    '40+_yds': 'reception_40_plus_yds', 
    '50+_yds': 'reception_50_plus_yds',
    'rz_tgt' : 'red_zone_targets',
    'lng': 'longest_reception', 
    'yds': 'receiving_yards'
    }
).filter(pl.col('player').is_not_null()).with_columns(
    pl.col('target_share', 'receiving_yards', 'air_yards_receiving', 'yards_before_contact_rc').str.replace(r'(%)|(,)',''),
    pl.col('year').str.extract(r'(\d{4})').alias('year')
).with_columns(
    pl.col('target_share').cast(pl.Float64).alias('target_share'), 
    pl.col('receiving_yards').str.to_integer().alias('receiving_yards'),
    pl.col('year').str.to_integer().alias('year'),
    pl.col('player').str.replace(r"\s*\([A-Z]{2,}\)", "").alias('player'),
    pl.col('air_yards_receiving').str.to_integer().alias('air_yards_receiving'),
    pl.col('yards_before_contact_rc').str.to_integer().alias('yards_before_contact_rc')
)

clean_up_te_names = te_advc_stats_raw.rename({'g': 'games_played',
    'g': 'games_played',
    "rec": 'receptions',
    'yds': 'receiving_yards',
    'y_r': 'yards_per_reception',
    'ybc': 'yards_before_contact_rc',
    'ybc_r': 'yards_before_contact_per_reception',
    'air': 'air_yards_receiving',
    'air_r': 'air_yards_per_reception', 
    'yacon': 'yac_per_reception',
    '%_tm' : 'target_share',
    'rz_tgt': 'red_zone_targets',
    '10+_yds': 'reception_10_plus_yds',
    '20+_yds': 'reception_20_plus_yds',
    '30+_yds': 'reception_30_plus_yds', 
    '40+_yds': 'reception_40_plus_yds', 
    '50+_yds': 'reception_50_plus_yds',
    'lng': 'longest_reception'
    }
).filter(pl.col('player').is_not_null()).with_columns(
    pl.col('target_share', 'receiving_yards', 'air_yards_receiving').str.replace(r'(%)|(,)',''),
    pl.col('year').str.extract(r'(\d{4})').alias('year')
).with_columns(
    pl.col('target_share').cast(pl.Float64).alias('target_share'), 
    pl.col('receiving_yards').str.to_integer().alias('receiving_yards'),
    pl.col('year').str.to_integer().alias('year'),
    pl.col('player').str.replace(r"\s*\([A-Z]{2,}\)", "").alias('player'),
    pl.col('air_yards_receiving').str.to_integer().alias('air_yards_receiving')
)


clean_rb_names = rb_advc_stats_raw.rename(
    {
        'g': 'games_played',
        'att': 'rush_attempts',
        'yds': 'rushing_yards', 
        'y_att': 'yards_per_attempt',
        'ybcon': 'yards_before_contact_rush', 
        'ybcon_att': 'yards_before_contact_per_attempt',
        'brktkl': 'broken_tackles',
        'tk_loss': 'tackles_for_loss',
        'tk_loss_yds' : 'tackles_for_loss_yards',
        '10+_yds': 'rush_10_plus_yds',
        '20+_yds': 'rush_20_plus_yds',
        '30+_yds': 'rush_30_plus_yds', 
        '40+_yds': 'rush_40_plus_yds', 
        '50+_yds': 'rush_50_plus_yds',
        'lng': 'longest_rush', 
        'rec' : 'receptions', 
        'tgt': 'targets',
        'rz_tgt': 'red_zone_targets',
        'yacon_duplicated_0': 'yards_before_contact_rc'
    }
).with_columns(
    pl.col('rushing_yards', 'yards_before_contact_rush').str.replace_all(r',', ''),
    pl.col('year').str.extract(r'(\d{4})')).with_columns(
        pl.col('rushing_yards', 'yards_before_contact_rush').str.to_integer(),
        pl.col('year').str.to_integer(),
        pl.col('player').str.replace(r"\s*\([A-Z]{2,}\)", "").alias('player')
    )

clean_qb_names = qb_advc_stats_raw.rename(
    {
        'comp': 'completions',
        'att': 'passing_attempts',
        'pct': 'completion_pct',
        'g': 'games_played', 
        'yds': 'passing_yards',
        'y_a': 'passing_yards_per_attempt',
        'air': 'passing_air_yards',
        'air_a': 'passing_air_yards_per_attempt',
        '10+_yds': 'pass_10_plus_yds',
        '20+_yds': 'pass_20_plus_yds',
        '30+_yds': 'pass_30_plus_yds', 
        '40+_yds': 'pass_40_plus_yds', 
        '50+_yds': 'pass_50_plus_yds',
        'pkt_time': 'time_in_the_pocket',
        'poor': 'poor_passes', 
        'rz_att': 'red_zone_pass_attempts',
        'rtg': 'passer_rating'

    }
).with_columns(
    pl.col('completion_pct', 'passing_yards', 'passing_air_yards').str.replace(r'(,|%)', ''),
    pl.col('year').str.extract(r'(\d{4})')
).with_columns(
    pl.col('completion_pct', 'passing_yards', 'passing_air_yards', 'year').str.to_integer(),
    pl.col('player').str.replace(r"\s*\([A-Z]{2,}\)", "").alias('player')
)

player_name_regex = r"\s*\([A-Z]{2,}\)"



wr_scoring_raw = pl.scan_csv('data/fpros-fantasy-scoring-*wr.csv', include_file_paths='year').collect().clean_names()

rb_scoring_raw = pl.scan_csv('data/fpros-fantasy-scoring-*rb.csv', include_file_paths='year').collect().clean_names()

te_scoring_raw = pl.scan_csv('data/fpros-fantasy-scoring-*te.csv', include_file_paths='year').collect().clean_names()

qb_scoring_raw = pl.scan_csv('data/fpros-fantasy-scoring-*qb.csv', include_file_paths='year').collect().clean_names()

dst_scoring_raw = pl.scan_csv('data/fpros-fantasy-scoring-*dst.csv', include_file_paths='year').collect().clean_names()

wr_scoring_clean = wr_scoring_raw.select(
    pl.col('player'), pl.col('yds_duplicated_0').alias('rushing_yards'), pl.col('td_duplicated_0').alias('rushing_tds'), pl.col('td').alias('receiving_tds'), pl.col('year', 'fpts', 'fpts_g')).with_columns(
    pl.col('player').str.replace(player_name_regex, '').alias('player'),
    pl.col('year').str.extract(r'(\d{4})').str.to_integer().alias('year')
)

te_scoring_clean = te_scoring_raw.select(
    pl.col('player'), pl.col('yds_duplicated_0').alias('rushing_yards'), pl.col('td_duplicated_0').alias('rushing_tds'), pl.col('td').alias('receiving_tds'), pl.col('year', 'fpts', 'fpts_g')
).with_columns(
    pl.col('player').str.replace(player_name_regex, '').alias('player'),
    pl.col('year').str.extract(r'(\d{4})').str.to_integer().alias('year')
)

rb_scoring_clean = rb_scoring_raw.select(
    pl.col('td').alias('rushing_tds'), pl.col('yds_duplicated_0').alias('receiving_yards'), pl.col('y_r').alias('yards_per_reception'), pl.col('td_duplicated_0').alias('receiving_tds'), pl.col('player','year', 'fpts', 'fpts_g')
).with_columns(
   # pl.col('receiving_yds').str.replace(r'(,)', '').str.to_integer(),
    pl.col('player').str.replace(player_name_regex, ''),
    pl.col('year').str.extract(r'(\d{4})').str.to_integer()
)


qb_scoring_clean = qb_scoring_raw.select(
    pl.col('td').alias('passing_tds'), pl.col('int'), pl.col('att_duplicated_0').alias('rush_attempts'), pl.col('yds_duplicated_0').alias('rushing_yards'), pl.col('td_duplicated_0').alias('rushing_tds'),
    pl.col('player', 'year', 'fpts', 'fpts_g')
).with_columns(
    pl.col('rushing_yards').str.replace(r'(,)', '').str.to_integer(),
    pl.col('player').str.replace(player_name_regex, ''),
    pl.col('year').str.extract(r'(\d{4})').str.to_integer()
).with_columns(
     (pl.col('rushing_yards') / pl.col('rush_attempts')).alias('yards_per_attempt')
)

qb_scoring_clean.glimpse()

dst_clean = dst_scoring_raw.select(pl.exclude('rost', 'rank')).with_columns(
    pl.col('year').str.extract(r'(\d{4})').str.to_integer(),
    pl.col('player').str.replace(player_name_regex, '')
).rename({'g': 'games_played'})


joined_qb_stats = clean_qb_names.join(qb_scoring_clean, on = ['player', 'year'], how = 'left').select(pl.exclude('rank'))

joined_rb_stats = clean_rb_names.join(rb_scoring_clean, on = ['player', 'year'], how = 'left').select(pl.exclude('rank'))

joined_wr_stats = clean_up_wr_names.join(wr_scoring_clean, on = ['player', 'year'], how = 'left').select(pl.exclude('rank'))

joined_te_stats = clean_up_te_names.join(te_scoring_clean, on = ['player', 'year'], how = 'left').select(pl.exclude('rank'))


# for the most part the nones are really just 

big_stats_data = pl.concat([joined_qb_stats, joined_rb_stats,  joined_te_stats, joined_wr_stats, dst_clean], how = 'diagonal')

check_rookies = big_stats_data.filter(
    pl.col('player').is_in(['Jayden Daniels', 'C.J. Stroud', 'Malik Nabers', 'Zay Flower', 'Sam LaPorta', 'Brock Bowers', 'Bucky Irving', 'Bijan Robinson'])
)


big_stats_data.write_parquet('data/fantasy-pros-stats.parquet')




check = big_stats_data.filter(
    pl.col('player') == 'Josh Allen'
)
