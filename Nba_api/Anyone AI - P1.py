from nba_api.stats.endpoints import commonplayerinfo, commonallplayers, playercareerstats, playernextngames
import time
import chardet

import pandas as pd

# TODO:
    # Calculate Age.
    # Investigate players without salary.
    # Cast DF.
    # Uniform indexes.
    # Merge

def get_and_save_players_list():
    players_f = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
    players_f = players_f[(players_f['TEAM_NAME'] != '') & (players_f['GAMES_PLAYED_FLAG'] != 'N') & (players_f['PERSON_ID'] != 1630597)]
    players_f = players_f[['PERSON_ID', 'DISPLAY_FIRST_LAST', 'TEAM_NAME']]
    players_f.to_csv("nba_current_players_list.csv")
    return players_f

players = get_and_save_players_list()
players_list = list(players['PERSON_ID'])

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
            print(f'F1 - Iteración N°:{a}')

    except:
        print('TimeOut. Incomplete file will be saved anyways')
        pass

    all_players = all_players.drop(['DISPLAY_FIRST_LAST','DISPLAY_LAST_COMMA_FIRST', 'DISPLAY_FI_LAST', 'PLAYER_SLUG',
                        'SCHOOL', 'LAST_AFFILIATION', 'SEASON_EXP', 'JERSEY', 'ROSTERSTATUS', 'TEAM_ID',
                        'TEAM_ABBREVIATION', 'TEAM_CODE', 'TEAM_CITY', 'PLAYERCODE', 'DLEAGUE_FLAG',
                        'NBA_FLAG', 'GAMES_PLAYED_FLAG', 'DRAFT_YEAR', 'DRAFT_ROUND', 'GREATEST_75_FLAG',
                        'GAMES_PLAYED_CURRENT_SEASON_FLAG'], axis=1)
    all_players.to_csv("nba_players_personal_info.csv")
    return all_players

players_personal_info = get_players_personal_information()

def get_players_career_stats():
    all_players = pd.DataFrame()
    a = 0
    try:
        for player in players_list:
            player_info = playercareerstats.PlayerCareerStats(player_id=player, per_mode36= 'PerGame' ).get_data_frames()
            player_info = player_info[3]
            all_players = pd.concat([all_players,player_info])
            time.sleep(0.7)
            a += 1
            print(f'F2 - Iteración N°:{a}')
    except:
        print('TimeOut. Incomplete file will be saved anyways')
        pass


    all_players.to_csv("nba_players_career_stats.csv")
    return all_players

players_career_stats = get_players_career_stats()

def get_players_next_game():
    all_next_games = pd.DataFrame()
    a = 0
    try:
        for player in players_list:
            try:
                next_games = playernextngames.PlayerNextNGames(player_id=player).get_data_frames()
                next_games = next_games[0]
                all_next_games = pd.concat([all_next_games,next_games])
                time.sleep(0.5)
                a += 1
                print(f'F3 - Iteración N°:{a}')
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

players_next_game = get_players_next_game()

def get_nba_players_salaries(csv_file_path):
    with open(csv_file_path, 'rb') as f:
        enc = chardet.detect(f.read())

    salaries_f = pd.read_csv(csv_file_path, encoding= enc['encoding'])
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
    salaries_f['2021-22'] = salaries_f['2021-22'].astype('int64')
    salaries_f['Player'] = salaries_f['Player'].astype('string')
    # salaries = salaries.style.format(thousands='.')
    salaries_f.to_csv('nba_players_salary.csv')
    return salaries_f

salaries = get_nba_players_salaries('contracts.csv')
