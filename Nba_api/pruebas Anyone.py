import pandas as pd
import numpy as np
from unidecode import unidecode
import re
from datetime import datetime as dt

players_personal_info = pd.read_csv('nba_players_personal_info.csv')
players_career_stats = pd.read_csv('nba_players_career_stats.csv')
players_salaries = pd.read_csv('nba_players_salary.csv')
players_next_game = pd.read_csv('nba_players_next_game.csv')

# players_personal_info = players_personal_info['']

# players_personal_info = players_personal_info.set_index('PERSON_ID')
# players_career_stats = players_career_stats.set_index('PLAYER_ID')


# def personal_info_cleanse(players_personal_info):
#     players_personal_info['PLAYER_NAME'] = players_personal_info['FIRST_NAME'] + ' ' + players_personal_info['LAST_NAME']
#     players_personal_info = players_personal_info.drop(['Unnamed: 0', 'FIRST_NAME', 'LAST_NAME'], axis=1)
#     players_personal_info.set_index('PERSON_ID', inplace=True)
#     aux_player_names = list(players_personal_info.loc[:,'PLAYER_NAME'])
#     players_personal_info.insert(0, 'PLAYER_NAME', aux_player_names, allow_duplicates=True)
#     players_personal_info = players_personal_info.iloc[:, 0:-1]
#     players_personal_info['SEASON_EXP'] = players_personal_info['TO_YEAR'] - players_personal_info['FROM_YEAR']
#     players_personal_info = players_personal_info[["PLAYER_NAME", "TEAM_NAME", "POSITION", "HEIGHT", "WEIGHT", "COUNTRY",
#                                                    "BIRTHDATE", "SEASON_EXP", "DRAFT_NUMBER"]]
#     players_personal_info['WEIGHT'] = round(players_personal_info['WEIGHT'].astype('float64') / 2.20462,2)
#     players_personal_info['HEIGHT'] = (players_personal_info['HEIGHT'].str.replace('-', '.')).astype('float64')
#     players_personal_info['HEIGHT'] = players_personal_info['HEIGHT'] * 30.48
#     players_personal_info = players_personal_info.astype(
#         {'PLAYER_NAME': 'string', 'TEAM_NAME': 'string', 'POSITION': 'string', 'COUNTRY': 'string',
#          'BIRTHDATE': 'datetime64', 'SEASON_EXP': 'int64', 'HEIGHT':'int64'})
#     print(players_personal_info.info())
#     return players_personal_info
#
# def career_stats_cleanse(players_career_stats):
#     players_career_stats = players_career_stats.set_index('PLAYER_ID')
#     players_career_stats = players_career_stats.drop(['LEAGUE_ID', 'Team_ID', 'GS', 'FGM',
#                                     'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
#                                     'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV',
#                                     'PF', 'Unnamed: 0'], axis=1)
#     players_career_stats = players_career_stats[["GP", "MIN", "PTS", "REB", "AST", "STL", "BLK"]]
#     print(players_career_stats.info())
#     return players_career_stats
#
# career_stats = career_stats_cleanse(players_career_stats_d)
# personal_info = personal_info_cleanse(players_personal_info_d)

# def remove_jr_sr(dataframe, players_column_name):
#     regex = r" Sr.$"
#     regex2 = r" Jr.$"
#     regex3 = r" II$"
#     regex4 = r" III$"
#     regex5 = r" IV$"
#     regex6 = r" V$"
#     subst = ""
#     try:
#         for i in dataframe.index:
#             try:
#                 test_str = dataframe.loc[i, str(players_column_name)]
#                 result = re.sub(regex, subst, test_str, 1)
#                 dataframe.loc[i,str(players_column_name)] = result
#                 if result == test_str:
#                     result = re.sub(regex2, subst, test_str, 1)
#                     dataframe.loc[i,str(players_column_name)] = result
#                     if result == test_str:
#                         result = re.sub(regex3, subst, test_str, 1)
#                         dataframe.loc[i, str(players_column_name)] = result
#                         if result == test_str:
#                             result = re.sub(regex4, subst, test_str, 1)
#                             dataframe.loc[i, str(players_column_name)] = result
#                             if result == test_str:
#                                 result = re.sub(regex5, subst, test_str, 1)
#                                 dataframe.loc[i, str(players_column_name)] = result
#                                 if result == test_str:
#                                     result = re.sub(regex6, subst, test_str, 1)
#                                     dataframe.loc[i, str(players_column_name)] = result
#             except:
#                 pass
#     except:
#         for i in range(len(dataframe)):
#             try:
#                 test_str = dataframe.loc[i, str(players_column_name)]
#                 result = re.sub(regex, subst, test_str, 1)
#                 dataframe.loc[i,str(players_column_name)] = result
#                 if result == test_str:
#                     result = re.sub(regex2, subst, test_str, 1)
#                     dataframe.loc[i,str(players_column_name)] = result
#                     if result == test_str:
#                         result = re.sub(regex3, subst, test_str, 1)
#                         dataframe.loc[i, str(players_column_name)] = result
#                         if result == test_str:
#                             result = re.sub(regex4, subst, test_str, 1)
#                             dataframe.loc[i, str(players_column_name)] = result
#                             if result == test_str:
#                                 result = re.sub(regex5, subst, test_str, 1)
#                                 dataframe.loc[i, str(players_column_name)] = result
#                                 if result == test_str:
#                                     result = re.sub(regex6, subst, test_str, 1)
#                                     dataframe.loc[i, str(players_column_name)] = result
#             except:
#                 pass
#     return dataframe
#
#
# players_personal_info = remove_jr_sr(players_personal_info, 'PLAYER_NAME')
#
# def get_nba_players_salaries(csv_file_path):
#     salaries_f = pd.read_csv(csv_file_path, encoding= 'utf-8')
#     salaries_f = salaries_f.drop_duplicates(subset=['Unnamed: 1'])
#     salaries_f = salaries_f.reset_index(drop=True)
#     salaries_f[['Player2', 'Discard']] = salaries_f['Unnamed: 1'].str.split('\\', expand=True)
#     salaries_f = salaries_f.drop(['Unnamed: 1', 'Discard'], axis=1)
#     salaries_f.columns = salaries_f.iloc[0]
#     salaries_f = salaries_f.drop([0], axis=0)
#     valores_player = list(salaries_f.loc[:, 'Player'])
#     salaries_f.insert(1, 'Player2', valores_player, allow_duplicates=True)
#     salaries_f = salaries_f.drop(['Player', 'Rk', 'Tm', '2022-23', '2023-24', '2024-25', '2025-26', '2026-27',
#                               'Signed Using', 'Guaranteed'], axis=1)
#     salaries_f = salaries_f.rename(columns={'Player2': 'Player'})
#     salaries_f['2021-22'] = salaries_f['2021-22'].str.replace('$', '')
#     salaries_f['2021-22'] = salaries_f['2021-22'].str.replace('?', '')
#     salaries_f = salaries_f.fillna(0)
#     salaries_f = remove_jr_sr(salaries_f, 'Player')
#     salaries_f['2021-22'] = salaries_f['2021-22'].astype('int64')
#     salaries_f['Player'] = salaries_f['Player'].astype('string')
#     salaries_f['Player'] = salaries_f['Player'].apply(unidecode)
#     for i, row in salaries_f.iterrows():
#         try:
#             salaries_f.loc[i, 'PLAYER_ID'] = int(players_personal_info.index[players_personal_info['PLAYER_NAME'] == row['Player']][0])
#         except:
#             deleted_player = salaries_f.loc[i,'Player']
#             salaries_f = salaries_f.drop(index= i)
#             print(f'{deleted_player} is not active. Player deleted')
#             pass
#
#     salaries_f = salaries_f.set_index('PLAYER_ID')
#     salaries_f.to_csv('nba_players_salary.csv')
#     return salaries_f
#
# salarios = get_nba_players_salaries('contracts.csv')


def merge_dataframes(personal_info, career_stats, salaries, next_game):
    raw_players_dataset1 = pd.merge(personal_info, career_stats, left_index=True, right_index=True)
    raw_players_dataset2 = pd.merge(raw_players_dataset1, salaries, left_index=True, right_index=True)
    raw_players_dataset3 = pd.merge(raw_players_dataset2, next_game, left_index=True, right_index=True) if not next_game.empty else raw_players_dataset2
    raw_players_dataset3 = raw_players_dataset3.drop(['Player', 'PLAYER_ID_x', 'PLAYER_ID_y'], axis=1)
    raw_players_dataset3 = raw_players_dataset3.rename(columns={'2021-22':'SALARY'})
    return raw_players_dataset3


def copy_and_delete_nan(players_dataset):
    players_dataset_c = players_dataset.copy()
    players_dataset_c = players_dataset_c[(~players_dataset_c['TEAM_NAME'].isnull()) | (~players_dataset_c['SALARY'].isnull())]
    return players_dataset_c

raw_players_dataset = merge_dataframes(players_personal_info, players_career_stats, players_salaries, players_next_game)
working_df = copy_and_delete_nan(raw_players_dataset)

working_df['BIRTHDATE'] = working_df['BIRTHDATE'].astype('datetime64')


def add_age_column(working_df):
    working_df['AGE'] = (dt.now() - working_df['BIRTHDATE'])
    days_4_years = working_df['AGE'].astype('string')
    days_4_years = days_4_years.str.split('days', expand=True)[0]
    years = (days_4_years.astype('int64') / 365).apply(np.floor).astype('int64')
    months = (((days_4_years.astype('int64') / 365) - years) * 12).apply(np.floor).astype('int64')
    days = (((((days_4_years.astype('int64') / 365) - years) * 12) - months) * 30).apply(np.floor).astype('int64')
    years_months_days = pd.concat([years,months,days], axis=1).astype('string')
    working_df['AGE'] = (years_months_days.iloc[:,0] + ' years, ' + years_months_days.iloc[:,1] + ' months, '
                         + years_months_days.iloc[:,2] + ' days')
    return working_df

working_df = add_age_column(working_df)


def update_position(working_df):
    split_aux = working_df['POSITION'].str.split('-', expand=True)
    split_aux = split_aux.iloc[:,0]
    working_df['POSITION'] = split_aux
    return working_df

working_df = update_position(working_df)