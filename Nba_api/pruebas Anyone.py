import pandas as pd

players_personal_info = pd.read_csv('nba_players_personal_info.csv')
players_career_stats = pd.read_csv('nba_players_career_stats.csv')
salaries = pd.read_csv('nba_players_salary.csv')

players_personal_info['PLAYER_NAME'] = players_personal_info['FIRST_NAME'] + ' ' + players_personal_info['LAST_NAME']
players_personal_info = players_personal_info.drop(['Unnamed: 0', 'FIRST_NAME', 'LAST_NAME'], axis=1)
players_personal_info.set_index('PERSON_ID', inplace=True)
aux_player_names = list(players_personal_info.loc[:,'PLAYER_NAME'])
players_personal_info.insert(0, 'PLAYER_NAME', aux_player_names, allow_duplicates=True)
players_personal_info = players_personal_info.iloc[:, 0:-1]
players_personal_info['SEASON_EXP'] = players_personal_info['TO_YEAR'] - players_personal_info['FROM_YEAR']
players_personal_info = players_personal_info[["PLAYER_NAME", "TEAM_NAME", "POSITION", "HEIGHT", "WEIGHT", "COUNTRY",
                                               "BIRTHDATE", "SEASON_EXP", "DRAFT_NUMBER"]]
players_personal_info['WEIGHT'] = round(players_personal_info['WEIGHT'] / 2.20462,2)
players_personal_info['HEIGHT'] = (players_personal_info['HEIGHT'].str.replace('-', '.')).astype('float')
players_personal_info['HEIGHT'] = round(players_personal_info['HEIGHT'] * 30.48, 2)
players_personal_info = players_personal_info.astype(
    {'PLAYER_NAME': 'string', 'TEAM_NAME': 'string', 'POSITION': 'string', 'HEIGHT': 'float64',
     'WEIGHT': 'float64', 'COUNTRY': 'string', 'BIRTHDATE': 'datetime64', 'SEASON_EXP': 'int64'})
print(players_personal_info.info())