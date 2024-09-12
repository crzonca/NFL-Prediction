import json
import time

import pandas as pd
from sportsipy.nfl.schedule import Schedule
from sportsipy.nfl.teams import Teams


def get_division(team_name):
    name_to_division = {'49ers': 'NFC West',
                        'Bears': 'NFC North',
                        'Bengals': 'AFC North',
                        'Bills': 'AFC East',
                        'Broncos': 'AFC West',
                        'Browns': 'AFC North',
                        'Buccaneers': 'NFC South',
                        'Cardinals': 'NFC West',
                        'Chargers': 'AFC West',
                        'Chiefs': 'AFC West',
                        'Colts': 'AFC South',
                        'Commanders': 'NFC East',
                        'Cowboys': 'NFC East',
                        'Dolphins': 'AFC East',
                        'Eagles': 'NFC East',
                        'Falcons': 'NFC South',
                        'Giants': 'NFC East',
                        'Jaguars': 'AFC South',
                        'Jets': 'AFC East',
                        'Lions': 'NFC North',
                        'Packers': 'NFC North',
                        'Panthers': 'NFC South',
                        'Patriots': 'AFC East',
                        'Raiders': 'AFC West',
                        'Rams': 'NFC West',
                        'Ravens': 'AFC North',
                        'Saints': 'NFC South',
                        'Seahawks': 'NFC West',
                        'Steelers': 'AFC North',
                        'Texans': 'AFC South',
                        'Titans': 'AFC South',
                        'Vikings': 'NFC North'}

    return name_to_division.get(team_name)


def get_name_from_abbrev(abbrev):
    abbrev_to_name = {'SFO': '49ers',
                      'CHI': 'Bears',
                      'CIN': 'Bengals',
                      'BUF': 'Bills',
                      'DEN': 'Broncos',
                      'CLE': 'Browns',
                      'TAM': 'Buccaneers',
                      'CRD': 'Cardinals',
                      'SDG': 'Chargers',
                      'KAN': 'Chiefs',
                      'CLT': 'Colts',
                      'WAS': 'Commanders',
                      'DAL': 'Cowboys',
                      'MIA': 'Dolphins',
                      'PHI': 'Eagles',
                      'ATL': 'Falcons',
                      'NYG': 'Giants',
                      'JAX': 'Jaguars',
                      'NYJ': 'Jets',
                      'DET': 'Lions',
                      'GNB': 'Packers',
                      'CAR': 'Panthers',
                      'NWE': 'Patriots',
                      'RAI': 'Raiders',
                      'RAM': 'Rams',
                      'RAV': 'Ravens',
                      'NOR': 'Saints',
                      'SEA': 'Seahawks',
                      'PIT': 'Steelers',
                      'HTX': 'Texans',
                      'OTI': 'Titans',
                      'MIN': 'Vikings'}

    return abbrev_to_name.get(abbrev, abbrev)


def create_sch_json():
    teams = Teams()
    team_schedules = list()
    for abbrev in teams.dataframes['abbreviation']:
        sch = Schedule(abbrev).dataframe[['location', 'opponent_abbr', 'week']]
        sch['team'] = get_name_from_abbrev(abbrev)
        sch['opponent'] = sch.apply(lambda r: get_name_from_abbrev(r['opponent_abbr']), axis=1)
        sch = sch.drop(columns=['opponent_abbr'])
        sch = sch.reset_index(drop=True)
        if not sch.empty:
            team_schedules.append(sch)
    full_schedule = pd.concat(team_schedules)
    full_schedule = full_schedule.sort_values(by='week')
    full_schedule = full_schedule.loc[full_schedule['location'] == 'Away']
    sch_dict = dict()
    weeks = list()
    for week in sorted(full_schedule['week'].unique()):
        week_list = list()
        week_df = full_schedule.loc[full_schedule['week'] == week]
        for index, matchup in week_df.iterrows():
            away = matchup['team']
            home = matchup['opponent']
            game = {'away': away,
                    'home': home}
            week_list.append(game)
        weeks.append(week_list)
    sch_dict['weeks'] = weeks
    path = 'Projects/nfl/NFL_Prediction/Pred/resources/2024Schedule.json'
    with open(path, 'w') as out:
        json.dump(sch_dict, out, indent=4)


def load_schedule():
    schedule_path = 'Projects/nfl/NFL_Prediction/Pred/resources/2024Schedule.json'
    with open(schedule_path, 'r') as f:
        schedule = json.load(f)
        return schedule


def get_games_before_week(week, use_persisted=True):
    if use_persisted:
        week_results = pd.read_csv('Projects/nfl/NFL_Prediction/Pred/resources/2024games.csv')
        week_results = week_results.dropna()
    else:
        teams = Teams()
        games_in_week = list()
        for abbrev in teams.dataframes['abbreviation']:
            sch = Schedule(abbrev).dataframe
            sch['team'] = abbrev
            game = sch.loc[sch['week'] <= week]
            game = game.reset_index(drop=True)
            if not game.empty and game.loc[0]['points_scored'] is not None:
                games_in_week.append(game)
            time.sleep(5)
        if games_in_week:
            week_results = pd.concat(games_in_week)
        else:
            week_results = pd.DataFrame()
    return week_results
