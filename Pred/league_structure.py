import json
import time

import scipy
import math
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


def simulate_season():
    intecept = .7102
    off_coefs = {'49ers': .2828,
                 'Bears': -.0276,
                 'Bengals': -.011,
                 'Bills': .1978,
                 'Broncos': -.0359,
                 'Browns': .0678,
                 'Buccaneers': -.0615,
                 'Cardinals': -.1146,
                 'Chargers': -.0672,
                 'Chiefs': .0025,
                 'Colts': .0678,
                 'Commanders': -.1176,
                 'Cowboys': .3188,
                 'Dolphins': .2929,
                 'Eagles': .1571,
                 'Falcons': -.1422,
                 'Giants': -.3302,
                 'Jaguars': .0186,
                 'Jets': -.3227,
                 'Lions': .2197,
                 'Packers': .0344,
                 'Panthers': -.4498,
                 'Patriots': -.4498,
                 'Raiders': -.1085,
                 'Rams': .0878,
                 'Ravens': .2664,
                 'Saints': .0828,
                 'Seahawks': -.0165,
                 'Steelers': -.1966,
                 'Texans': .0186,
                 'Titans': -.1933,
                 'Vikings': -.073}
    def_coefs = {'49ers': -.2166,
                 'Bears': .0239,
                 'Bengals': .037,
                 'Bills': -.1739,
                 'Broncos': .1098,
                 'Browns': -.022,
                 'Buccaneers': -.1298,
                 'Cardinals': .2066,
                 'Chargers': .0728,
                 'Chiefs': -.2301,
                 'Colts': .1146,
                 'Commanders': .3363,
                 'Cowboys': -.1611,
                 'Dolphins': .0551,
                 'Eagles': .1455,
                 'Falcons': .0079,
                 'Giants': .0952,
                 'Jaguars': .0025,
                 'Jets': -.0415,
                 'Lions': .0652,
                 'Packers': -.0557,
                 'Panthers': .117,
                 'Patriots': -.011,
                 'Raiders': -.1115,
                 'Rams': .0186,
                 'Ravens': -.2789,
                 'Saints': -.1237,
                 'Seahawks': .0828,
                 'Steelers': -.1329,
                 'Texans': -.0529,
                 'Titans': -.0083,
                 'Vikings': -.022}

    schedule_df = pd.read_csv(
        'D:\\Colin\\Documents\\Programming\\Python\\PythonProjects\\Projects\\nfl\\NFL_Prediction\\Pred\\resources\\2024games.csv')
    drives_norm = scipy.stats.norm(10.65625, 1.574989557)
    drives = [round(x) for x in drives_norm.rvs(224) for _ in (0, 1)]
    schedule_df['drives'] = pd.Series(list(schedule_df['drives'].head(96)) + drives)

    for index, row in schedule_df.iterrows():
        if not pd.isna(row['points_scored']):
            continue
        team = row['team']
        drives = row['drives']
        game_id = row['boxscore_index']
        relevant_obs = schedule_df.loc[schedule_df['boxscore_index'] == game_id]
        opp = set(relevant_obs['team']) - set([team])
        opp = opp.pop()

        ppd_mu = math.exp(intecept + off_coefs.get(team) + def_coefs.get(opp))
        lamb = proj_ppd * drives
        pois = scipy.stats.nbinom(lamb)
        proj_score = pois.rvs(1)[0]

        opp_obs = schedule_df.loc[(schedule_df['boxscore_index'] == game_id) & (schedule_df['team'] == opp)].squeeze()
        if not pd.isna(opp_obs['points_scored']) and opp_obs['points_scored'] == proj_score:
            proj_score = proj_score + 1

        print(team.ljust(15), 'vs', opp.ljust(15), proj_score, 'points')

        schedule_df.at[index, 'points_scored'] = proj_score

    ties = schedule_df.loc[schedule_df.duplicated(subset=['boxscore_index', 'points_scored'])]
    if not ties.empty:
        print(ties)
    schedule_df.to_csv(
        'D:\\Colin\\Documents\\Programming\\Python\\PythonProjects\\Projects\\nfl\\NFL_Prediction\\Pred\\resources\\2024games_simulated.csv',
        index=False)

    print('Average PPG:', schedule_df['points_scored'].mean())
