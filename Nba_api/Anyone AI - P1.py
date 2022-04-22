from nba_api.stats.endpoints import commonplayerinfo, commonallplayers, playercareerstats, playernextngames
import time
from unidecode import unidecode
import re
import pandas as pd
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# LEN 503
def get_and_save_players_list():
    players_f = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
    players_f = players_f[(players_f['TEAM_NAME'] != '') & (players_f['GAMES_PLAYED_FLAG'] != 'N') & (players_f['PERSON_ID'] != 1630597)]
    players_f = players_f[['PERSON_ID', 'DISPLAY_FIRST_LAST', 'TEAM_NAME']]
    players_f.to_csv("nba_current_players_list.csv")
    return players_f


def remove_jr_sr(dataframe, players_column_name):
    regex = r" Sr.$"
    regex2 = r" Jr.$"
    regex3 = r" II$"
    regex4 = r" III$"
    regex5 = r" IV$"
    regex6 = r" V$"
    subst = ""
    try:
        for i in dataframe.index:
            try:
                test_str = dataframe.loc[i, str(players_column_name)]
                result = re.sub(regex, subst, test_str, 1)
                dataframe.loc[i, str(players_column_name)] = result
                if result == test_str:
                    result = re.sub(regex2, subst, test_str, 1)
                    dataframe.loc[i,str(players_column_name)] = result
                    if result == test_str:
                        result = re.sub(regex3, subst, test_str, 1)
                        dataframe.loc[i, str(players_column_name)] = result
                        if result == test_str:
                            result = re.sub(regex4, subst, test_str, 1)
                            dataframe.loc[i, str(players_column_name)] = result
                            if result == test_str:
                                result = re.sub(regex5, subst, test_str, 1)
                                dataframe.loc[i, str(players_column_name)] = result
                                if result == test_str:
                                    result = re.sub(regex6, subst, test_str, 1)
                                    dataframe.loc[i, str(players_column_name)] = result
            except:
                pass
    except:
        for i in range(len(dataframe)):
            try:
                test_str = dataframe.loc[i, str(players_column_name)]
                result = re.sub(regex, subst, test_str, 1)
                dataframe.loc[i, str(players_column_name)] = result
                if result == test_str:
                    result = re.sub(regex2, subst, test_str, 1)
                    dataframe.loc[i,str(players_column_name)] = result
                    if result == test_str:
                        result = re.sub(regex3, subst, test_str, 1)
                        dataframe.loc[i, str(players_column_name)] = result
                        if result == test_str:
                            result = re.sub(regex4, subst, test_str, 1)
                            dataframe.loc[i, str(players_column_name)] = result
                            if result == test_str:
                                result = re.sub(regex5, subst, test_str, 1)
                                dataframe.loc[i, str(players_column_name)] = result
                                if result == test_str:
                                    result = re.sub(regex6, subst, test_str, 1)
                                    dataframe.loc[i, str(players_column_name)] = result
            except:
                pass
    return dataframe


# LEN 503
def get_players_personal_information():
    all_players = pd.DataFrame()
    a = 0
    try:
        for player in players_list:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player).get_data_frames()
            player_info = player_info[0]
            all_players = pd.concat([all_players,player_info])
            time.sleep(0.5)
            a += 1
            print(f'Personal Info - Iteration N°:{a}')

    except:
        print('TimeOut. Incomplete file will be saved anyways')
        pass

    def personal_info_cleanse(all_players_c):
        all_players_c = all_players_c.drop(['DISPLAY_FIRST_LAST', 'DISPLAY_LAST_COMMA_FIRST', 'DISPLAY_FI_LAST', 'PLAYER_SLUG',
                            'SCHOOL', 'LAST_AFFILIATION', 'SEASON_EXP', 'JERSEY', 'ROSTERSTATUS', 'TEAM_ID',
                            'TEAM_ABBREVIATION', 'TEAM_CODE', 'TEAM_CITY', 'PLAYERCODE', 'DLEAGUE_FLAG',
                            'NBA_FLAG', 'GAMES_PLAYED_FLAG', 'DRAFT_YEAR', 'DRAFT_ROUND', 'GREATEST_75_FLAG',
                            'GAMES_PLAYED_CURRENT_SEASON_FLAG'], axis=1)
        all_players_c['PLAYER_NAME'] = all_players_c['FIRST_NAME'] + ' ' + all_players_c[
            'LAST_NAME']
        all_players_c = all_players_c.drop(['FIRST_NAME', 'LAST_NAME'], axis=1)
        all_players_c.set_index('PERSON_ID', inplace=True)
        aux_player_names = list(all_players_c.loc[:, 'PLAYER_NAME'])
        all_players_c.insert(0, 'PLAYER_NAME', aux_player_names, allow_duplicates=True)
        all_players_c = all_players_c.iloc[:, 0:-1]
        all_players_c['SEASON_EXP'] = all_players_c['TO_YEAR'] - all_players_c['FROM_YEAR']
        all_players_c = all_players_c[
            ["PLAYER_NAME", "TEAM_NAME", "POSITION", "HEIGHT", "WEIGHT", "COUNTRY",
             "BIRTHDATE", "SEASON_EXP", "DRAFT_NUMBER"]]
        # all_players_c['WEIGHT'] = round(all_players_c['WEIGHT'].astype('float64') / 2.20462, 2)
        # all_players_c['HEIGHT'] = (all_players_c['HEIGHT'].str.replace('-', '.')).astype('float64')
        # all_players_c['HEIGHT'] = round(all_players_c['HEIGHT'] * 30.48, 2)
        all_players_c = all_players_c.astype({'PLAYER_NAME': 'string', 'TEAM_NAME': 'string', 'POSITION': 'string',
                                              'COUNTRY':'string', 'BIRTHDATE':'datetime64', 'SEASON_EXP':'int64'})
        return all_players_c

    all_players = remove_jr_sr(personal_info_cleanse(all_players), 'PLAYER_NAME')
    all_players.to_csv("nba_players_personal_info.csv")
    return all_players


# LEN 503
def get_players_career_stats():
    all_players = pd.DataFrame()
    a = 0
    try:
        for player in players_list:
            player_info = playercareerstats.PlayerCareerStats(player_id=player, per_mode36= 'PerGame' ).get_data_frames()
            player_info = player_info[1]
            all_players = pd.concat([all_players,player_info])
            time.sleep(0.8)
            a += 1
            print(f'Career stats - Iteration N°:{a}')
    except:
        print('TimeOut. Incomplete file will be saved anyways')
        pass

    def career_stats_cleanse(players_career_stats_d):
        players_career_stats_d = players_career_stats_d.set_index('PLAYER_ID')
        players_career_stats_d = players_career_stats_d.drop(['LEAGUE_ID', 'Team_ID', 'GS', 'FGM',
                                                          'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
                                                          'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV',
                                                          'PF'], axis=1)
        players_career_stats_d = players_career_stats_d[["GP", "MIN", "PTS", "REB", "AST", "STL", "BLK"]]
        return players_career_stats_d

    all_players = career_stats_cleanse(all_players)

    all_players.to_csv("nba_players_career_stats.csv")
    return all_players


def get_players_next_game():
    all_next_games = pd.DataFrame()
    a = 0
    try:
        for player in players_list:
            try:
                next_games = playernextngames.PlayerNextNGames(player_id=player).get_data_frames()
                next_games = next_games[0]
                all_next_games = pd.concat([all_next_games,next_games])
                time.sleep(0.7)
                a += 1
                print(f'Next games - Iteration N°:{a}')
            except:
                print(f'Player n°: {player} not found')
                continue
    except:
        print('TimeOut. Incomplete file will be saved anyways')
        pass

    if all_next_games.empty:
        print('Season just closed. There are no next games.')
    else:
        all_next_games.to_csv("nba_players_next_game.csv")
    return all_next_games


# LEN 470
def get_nba_players_salaries(csv_file_path):
    salaries_f = pd.read_csv(csv_file_path, encoding= 'utf-8')
    salaries_f = salaries_f.drop_duplicates(subset=['Unnamed: 1'])
    salaries_f = salaries_f.reset_index(drop=True)
    salaries_f[['Player2', 'Discard']] = salaries_f['Unnamed: 1'].str.split('\\', expand=True)
    salaries_f = salaries_f.drop(['Unnamed: 1', 'Discard'], axis=1)
    salaries_f.columns = salaries_f.iloc[0]
    salaries_f = salaries_f.drop([0], axis=0)
    valores_player = list(salaries_f.loc[:, 'Player'])
    salaries_f.insert(1, 'Player2', valores_player, allow_duplicates=True)
    salaries_f = salaries_f.drop(['Player', 'Rk', 'Tm', '2022-23', '2023-24', '2024-25', '2025-26', '2026-27',
                              'Signed Using', 'Guaranteed'], axis=1)
    salaries_f = salaries_f.rename(columns={'Player2': 'Player'})
    salaries_f['2021-22'] = salaries_f['2021-22'].str.replace('$', '')
    salaries_f['2021-22'] = salaries_f['2021-22'].str.replace('?', '')
    salaries_f = salaries_f.fillna(0)
    salaries_f = remove_jr_sr(salaries_f, 'Player')
    salaries_f['2021-22'] = salaries_f['2021-22'].astype('int64')
    salaries_f['Player'] = salaries_f['Player'].astype('string')
    salaries_f['Player'] = salaries_f['Player'].apply(unidecode)

    for i, row in salaries_f.iterrows():
        try:
            salaries_f.loc[i, 'PLAYER_ID'] = int(players_personal_info.index[players_personal_info['PLAYER_NAME'] == row['Player']][0])
        except:
            deleted_player = salaries_f.loc[i,'Player']
            salaries_f = salaries_f.drop(index= i)
            print(f'{deleted_player} is not active. Player deleted')
            pass

    salaries_f = salaries_f.set_index('PLAYER_ID')
    salaries_f.to_csv('nba_players_salary.csv')
    return salaries_f


players = get_and_save_players_list()
players_list = list(players['PERSON_ID'])
players_personal_info = get_players_personal_information()
players_career_stats = get_players_career_stats()
players_next_game = get_players_next_game()
players_salaries = get_nba_players_salaries('contracts.csv')

# VER SI HAY QUE CORREGIR LEFT_ON=
def merge_dataframes(personal_info, career_stats, next_game, salaries):
    raw_players_dataset1 = pd.merge(personal_info, career_stats, left_index=True, right_index=True)
    raw_players_dataset2 = pd.merge(raw_players_dataset1, salaries, left_index=True, right_index=True)
    raw_players_dataset3 = pd.merge(raw_players_dataset2, next_game, left_index=True, right_index=True) if not next_game.empty else raw_players_dataset2
    raw_players_dataset3 = raw_players_dataset3.drop(['Player'], axis=1)
    raw_players_dataset3 = raw_players_dataset3.rename(columns={'2021-22':'SALARY'})
    return raw_players_dataset3


def copy_and_delete_nan(players_dataset):
    players_dataset_c = players_dataset.copy()
    players_dataset_c = players_dataset_c[(~players_dataset_c['TEAM_NAME'].isnull()) | (~players_dataset_c['SALARY'].isnull())]
    return players_dataset_c


def cast_columns(working_df):
    working_df['SALARY'] = working_df['SALARY'].astype('int64')
    working_df['BIRTHDATE'] = working_df['BIRTHDATE'].astype('datetime64')
    return working_df


def convert_height_column(working_df):
    working_df['HEIGHT'] = (working_df['HEIGHT'].str.replace('-', '.')).astype('float64')
    working_df['HEIGHT'] = round(working_df['HEIGHT'] * 30.48, 2)
    return working_df


def convert_weight_column(working_df):
    working_df['WEIGHT'] = round(working_df['WEIGHT'].astype('float64') / 2.20462, 2)
    return working_df


def add_age_column(working_df):
    working_df['AGE'] = (dt.now() - working_df['BIRTHDATE'])
    days_4_years = working_df['AGE'].astype('string')
    days_4_years = days_4_years.str.split('days', expand=True)[0]
    years_mod = days_4_years.astype('int64') / 365
    years = (days_4_years.astype('int64') / 365).apply(np.floor).astype('int64')
    months = (((days_4_years.astype('int64') / 365) - years) * 12).apply(np.floor).astype('int64')
    days = (((((days_4_years.astype('int64') / 365) - years) * 12) - months) * 30).apply(np.floor).astype('int64')
    years_months_days = pd.concat([years,months,days], axis=1).astype('string')
    working_df['AGE'] = (years_months_days.iloc[:,0] + ' years, ' + years_months_days.iloc[:,1] + ' months, '
                         + years_months_days.iloc[:,2] + ' days')
    return working_df, years_mod


def update_position(working_df):
    split_aux = working_df['POSITION'].str.split('-', expand=True)
    split_aux = split_aux.iloc[:,0]
    working_df['POSITION'] = split_aux
    return working_df


raw_players_dataset = merge_dataframes(players_personal_info, players_career_stats, players_next_game, players_salaries)
working_df = copy_and_delete_nan(raw_players_dataset)
working_df = cast_columns(working_df)
working_df = convert_height_column(working_df)
working_df = convert_weight_column(working_df)
working_df = add_age_column(working_df)[0]
years_mod = pd.DataFrame({'years':add_age_column(working_df)[1]})
working_df = update_position(working_df)
working_df.to_csv("nba_players_processed_dataset.csv")


def general_metrics(working_df):
    usa_players = working_df[working_df['COUNTRY']=='USA']
    per_position = working_df.groupby(by=['POSITION']).count()
    center = per_position.loc['Center', 'PERSON_ID']
    forward = per_position.loc['Forward', 'PERSON_ID']
    guard = per_position.loc['Guard', 'PERSON_ID']
    rookies = working_df[working_df['SEASON_EXP']==0]

    per_team = (working_df.groupby(by=['TEAM_NAME']).count())
    per_team = per_team['PERSON_ID'].copy()
    per_team = per_team.reset_index()
    per_team.columns = ['Team','N° Players']

    print('==========================================')
    print(f'Total amount of players: {len(working_df)}')
    print(f'Number of USA born players: {len(usa_players)}')
    print(f'Number of foreign players: {len(working_df)-len(usa_players)}')
    print(f'Number of Center Players: {center}')
    print(f'Number of Forward Players: {forward}')
    print(f'Number of Guard Players: {guard}')
    print(f'Number of Rookies: {len(rookies)}')
    print('==========================================')
    print(per_team)


def players_description(working_df, years_mod):
    avg_age = int(years_mod.mean())
    youngest_index = years_mod.index[years_mod['years'] == years_mod['years'].min()][0]
    youngest_player = working_df.iloc[youngest_index, -1]
    oldest_index = years_mod.index[years_mod['years'] == years_mod['years'].max()][0]
    oldest_player = working_df.iloc[oldest_index, -1]
    min_height = working_df['HEIGHT'].min()
    max_height = working_df['HEIGHT'].max()
    avg_height_per_pos = pd.DataFrame({'avg_height':working_df.groupby(by='POSITION').mean()['HEIGHT']})
    center = round(avg_height_per_pos.loc['Center','avg_height'], 2)
    forward = round(avg_height_per_pos.loc['Forward','avg_height'], 2)
    guard = round(avg_height_per_pos.loc['Guard','avg_height'], 2)

    print('==========================================')
    print(f'Average player age: {avg_age}')
    print(f'Youngest player age: {youngest_player}')
    print(f'Oldest player age: {oldest_player}')
    print(f'Min Height: {min_height}')
    print(f'Max Height: {max_height}')
    print(f'Average Height Center: {center}')
    print(f'Average Height Forward: {forward}')
    print(f'Average Height Guard: {guard}')
    print('==========================================')


def contracts(working_df):
    min_salary = int(working_df['SALARY'][working_df['SALARY'] != 0].min())
    max_salary = int(working_df['SALARY'].max())
    mean_salary = int(working_df['SALARY'].mean())
    median_salary = int(working_df['SALARY'].median())

    print(f'Min Salary: {min_salary}')
    print(f'Max Salary: {max_salary}')
    print(f'Mean Salary: {mean_salary}')
    print(f'Median Salary: {median_salary}')
    print('==========================================')

def graphs(working_df):
    new_df = working_df[working_df['SALARY'] != 0]
    new_df = new_df.astype({'SALARY':'int64', 'PTS':'float64'})
    new_df2 = new_df[new_df['SEASON_EXP'] > 4]

    sns.relplot(x='PTS', y='SALARY', data=new_df, hue='POSITION').set(title='Score Vs Salary')
    plt.show()

    fig, axs = plt.subplots(ncols=2, figsize=(10,6))
    sns.regplot(x='AST', y='SALARY', data=new_df, ax=axs[0]).set(title='Assists Vs Salary')
    sns.regplot(x='REB', y='SALARY', data=new_df, ax=axs[1]).set(title='Rebounds Vs Salary')
    plt.show()

    fig, axs = plt.subplots(ncols=3, figsize=(10,6))
    sns.regplot(x='PTS', y='SALARY', data=new_df2, ax=axs[0]).set(title='Score Vs Salary')
    sns.regplot(x='AST', y='SALARY', data=new_df2, ax=axs[1]).set(title='Assists Vs Salary')
    sns.regplot(x='REB', y='SALARY', data=new_df2, ax=axs[2]).set(title='Rebounds Vs Salary')
    plt.show()

    g = sns.FacetGrid(data=new_df, col='POSITION').set(title='Scoring per position')
    g.map(sns.boxplot, 'PTS')
    plt.show()

    sns.histplot(data=working_df, x='HEIGHT', bins=50, hue='POSITION', element='step').set(title='Height distribution')
    plt.show()


general_metrics(working_df)
players_description(working_df, years_mod)
contracts(working_df)
graphs(working_df)
