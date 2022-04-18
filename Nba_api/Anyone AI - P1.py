from nba_api.stats.endpoints import commonplayerinfo, commonallplayers, playercareerstats, playernextngames
import time
import chardet

import pandas as pd

# TODO:
    # Calculate Age. # NO NEED
    # Investigate players without salary.
    # Cast DF. # DONE
    # Correct unit measures. # DONE
    # Correct columns. # DONE
    # Uniform indexes. # DONE
    # Merge.

def get_and_save_players_list():
    players_f = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
    players_f = players_f[(players_f['TEAM_NAME'] != '') & (players_f['GAMES_PLAYED_FLAG'] != 'N') & (players_f['PERSON_ID'] != 1630597)]
    players_f = players_f[['PERSON_ID', 'DISPLAY_FIRST_LAST', 'TEAM_NAME']]
    players_f.to_csv("nba_current_players_list.csv")
    return players_f

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
        all_players_c['WEIGHT'] = round(all_players_c['WEIGHT'].astype('float64') / 2.20462, 2)
        all_players_c['HEIGHT'] = (all_players_c['HEIGHT'].str.replace('-', '.')).astype('float64')
        all_players_c['HEIGHT'] = round(all_players_c['HEIGHT'] * 30.48, 2)
        all_players_c = all_players_c.astype({'PLAYER_NAME': 'string', 'TEAM_NAME': 'string', 'POSITION': 'string',
                                              'COUNTRY':'string', 'BIRTHDATE':'datetime64', 'SEASON_EXP':'int64',
                                              'HEIGHT': 'int64'  })
        return all_players_c

    all_players = personal_info_cleanse(all_players)
    all_players.to_csv("nba_players_personal_info.csv")
    return all_players


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
        print(players_career_stats_d.info())
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
                time.sleep(0.5)
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
    salaries_f = salaries_f.dropna()
    salaries_f['2021-22'] = salaries_f['2021-22'].astype('int64')
    salaries_f['Player'] = salaries_f['Player'].astype('string')

    for i, row in salaries_f.iterrows():
        salaries_f.loc[i, 'PLAYER_ID'] = int(players_personal_info.index[players_personal_info['PLAYER_NAME'] == row['Player']][0])

    salaries_f.to_csv('nba_players_salary.csv')
    return salaries_f


players = get_and_save_players_list()
players_list = list(players['PERSON_ID'])
players_personal_info = get_players_personal_information()
players_career_stats = get_players_career_stats()
players_next_game = get_players_next_game()
salaries = get_nba_players_salaries('contracts.csv')
