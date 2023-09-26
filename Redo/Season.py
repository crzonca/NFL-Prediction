import itertools
import json
import math
import pickle
import random
import statistics
import time
import warnings

import PIL
import choix
import matplotlib.patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.linear_model import PoissonRegressor
from sportsipy.nfl.schedule import Schedule
from sportsipy.nfl.teams import Teams

import Projects.nfl.NFL_Prediction.Core.BayesianResume as br
import Projects.nfl.NFL_Prediction.OddsHelper as Odds

graph = nx.MultiDiGraph()
team_df = pd.DataFrame(columns=['Team', 'Division', 'Games Played',
                                'Wins', 'Losses', 'Ties',
                                'BT', 'BT Var', 'Win Pct', 'Bayes Win Pct',
                                'Avg Points', 'Avg Points Allowed',
                                'Points Intercept 2', 'Points Coef 2', 'Points Allowed Coef 2', 'Adjusted Points 2',
                                'Adjusted Points Allowed 2',
                                'Points Intercept', 'Points Coef', 'Points Allowed Coef', 'Adjusted Points',
                                'Adjusted Points Allowed', 'Yards Intercept', 'Yards Coef', 'Yards Allowed Coef',
                                'Adjusted Yards', 'Adjusted Yards Allowed', 'YPG', 'YPG Allowed'])

game_df = pd.DataFrame(columns=['Team', 'Win', 'Points', 'Points Allowed', 'Yards', 'Yards Allowed'])
individual_df = pd.DataFrame(columns=['Team', 'Opponent', 'Points', 'Yards'])


def load_model():
    model_path = 'D:\\Colin\\Documents\\Programming\\Python\\' \
                 'PythonProjects\\Projects\\nfl\\NFL_Prediction\\Redo\\model.pkl'
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

        return clf


def load_pt():
    pt_path = 'D:\\Colin\\Documents\\Programming\\Python\\' \
              'PythonProjects\\Projects\\nfl\\NFL_Prediction\\Redo\\pt.pkl'
    with open(pt_path, 'rb') as f:
        pt = pickle.load(f)

        return pt


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
    path = 'D:\\Colin\\Documents\\Programming\\Python\\PythonProjects\\Projects\\nfl\\' \
           'NFL_Prediction\\Redo\\2023Schedule.json'
    with open(path, 'w') as out:
        json.dump(sch_dict, out, indent=4)


def load_schedule():
    schedule_path = 'D:\\Colin\\Documents\\Programming\\Python\\' \
                    'PythonProjects\\Projects\\nfl\\NFL_Prediction\\Redo\\2023Schedule.json'
    with open(schedule_path, 'r') as f:
        schedule = json.load(f)
        return schedule


def get_bayes_avg(prior_avg, prior_var, sample_avg, sample_var, n):
    k_0 = sample_var / prior_var
    posterior_avg = ((k_0 / (k_0 + n)) * prior_avg) + ((n / (k_0 + n)) * sample_avg)
    return posterior_avg


def get_bayes_avg_wins(team_name):
    matching_games = game_df.loc[game_df['Team'] == team_name]

    # prior_avg = 0.505220599260847
    prior_avg = 0.5
    prior_var = 0.0393327761147111

    wins = list(matching_games['Win'])
    if len(wins) < 2:
        return prior_avg

    win_pct = statistics.mean(wins)
    win_var = statistics.variance(wins)

    return get_bayes_avg(prior_avg, prior_var, win_pct, win_var, len(wins))


def get_bradley_terry_from_graph(graph_name):
    nodes = graph_name.nodes
    df = pd.DataFrame(nx.to_numpy_array(graph_name), columns=nodes)
    df.index = nodes

    teams = list(df.index)
    df = df.fillna(0)

    teams_to_index = {team: i for i, team in enumerate(teams)}
    index_to_teams = {i: team for team, i in teams_to_index.items()}

    graph_name = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    edges = [list(itertools.repeat((teams_to_index.get(team2),
                                    teams_to_index.get(team1)),
                                   int(weight_dict.get('weight'))))
             for team1, team2, weight_dict in graph_name.edges.data()]
    edges = list(itertools.chain.from_iterable(edges))

    coeffs, cov = choix.ep_pairwise(n_items=len(teams), data=edges, alpha=1)
    coeffs = pd.Series(coeffs)
    cov = pd.Series(cov.diagonal())
    coef_df = pd.DataFrame([coeffs, cov]).T
    coef_df.columns = ['BT', 'Var']
    coef_df.index = [index_to_teams.get(index) for index in coef_df.index]
    coef_df = coef_df.sort_values(by='BT', ascending=False)
    return coef_df


def parity_clock():
    # Iterate over all cycles and find the one with the longest length
    longest_cycle_length = 0
    cycle = None
    for simple_cycle in nx.simple_cycles(graph):
        if len(simple_cycle) > longest_cycle_length:
            longest_cycle_length = len(simple_cycle)
            cycle = simple_cycle
            if longest_cycle_length == 32:
                break

    # If there are any cycles
    if cycle:
        print('Parity Clock')

        # Reverse the cycle direction
        cycle = list(reversed(cycle))

        # Add the starting team to the end to complete the loop
        cycle.append(cycle[0])

        # Format new lines if the length of the cycle is too long to print in one line
        if len(cycle) > 8:
            cycle[8] = '\n' + cycle[8]
        if len(cycle) > 16:
            cycle[16] = '\n' + cycle[16]
        if len(cycle) > 24:
            cycle[24] = '\n' + cycle[24]

        # Print the cycle
        print(' -> '.join(cycle))
        print()

        print('Still missing:')
        missing = set(team_df.index) - {team.strip() for team in cycle}
        print(' | '.join(missing))
        print()


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


def get_games_before_week(week, use_persisted=False):
    if use_persisted:
        week_results = pd.read_csv('Projects/nfl/NFL_Prediction/Redo/2023games.csv')
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


def get_game_results(week, week_results):
    week_results = week_results.loc[week_results['week'] == week]

    games_dict = dict()
    for index, row in week_results.iterrows():
        team = get_name_from_abbrev(row['team'])
        net_pass_yards = row['pass_yards']
        rush_yards = row['rush_yards']
        if net_pass_yards is None or rush_yards is None:
            continue
        total_yards = net_pass_yards + rush_yards
        points = row['points_scored']
        location = row['location']
        game_id = row['boxscore_index']

        games_dict[game_id + ' ' + location] = (team, total_yards, points, location, game_id)

    game_ids = {game[-1] for key, game in games_dict.items()}
    for game_id in game_ids:
        matching_games = [game for key, game in games_dict.items() if game[-1] == game_id]
        home_version = [game for game in matching_games if game[3] == 'Home'][0]
        away_version = [game for game in matching_games if game[3] == 'Away'][0]

        set_game_outcome(away_version[0], home_version[0],
                         away_version[2], away_version[1], home_version[2], home_version[1])


def set_game_outcome(home_name, away_name, home_points, home_yards, away_points, away_yards):
    global graph
    global team_df
    global game_df
    global individual_df

    home_victory = home_points > away_points
    away_victory = away_points > home_points
    tie = not home_victory and not away_victory

    game_df.loc[len(game_df.index)] = [home_name, 1 if home_victory else 0,
                                       home_points, away_points, home_yards, away_yards]
    game_df.loc[len(game_df.index)] = [away_name, 1 if away_victory else 0,
                                       away_points, home_points, away_yards, home_yards]

    individual_df.loc[len(individual_df.index)] = [home_name, away_name, home_points, home_yards]
    individual_df.loc[len(individual_df.index)] = [away_name, home_name, away_points, away_yards]

    # points_regression = Ridge(alpha=5, fit_intercept=True)
    points_regression = PoissonRegressor(alpha=.1, fit_intercept=True)
    points_df = individual_df[['Team', 'Opponent', 'Points']]
    points_dummy_vars = pd.get_dummies(points_df[['Team', 'Opponent']])

    points_regression.fit(X=points_dummy_vars, y=points_df['Points'])

    points_reg_results = pd.DataFrame({
        'coef_name': points_dummy_vars.columns.values,
        'points_reg_coef': points_regression.coef_
    })
    points_reg_results['points_reg_value'] = (points_reg_results['points_reg_coef'] + points_regression.intercept_)
    points_reg_results = points_reg_results.set_index('coef_name')

    for team_name in team_df.index:
        team_df.at[team_name, 'Points Intercept'] = points_regression.intercept_

        if 'Team_' + team_name in points_reg_results.index:
            team_df.at[team_name, 'Points Coef'] = points_reg_results.at['Team_' + team_name, 'points_reg_coef']
            team_df.at[team_name, 'Adjusted Points'] = math.exp(points_reg_results.at['Team_' + team_name, 'points_reg_value'])
        else:
            team_df.at[team_name, 'Points Coef'] = 0
            team_df.at[team_name, 'Adjusted Points'] = math.exp(points_regression.intercept_)

        if 'Opponent_' + team_name in points_reg_results.index:
            team_df.at[team_name, 'Points Allowed Coef'] = points_reg_results.at['Opponent_' + team_name,
                                                                                 'points_reg_coef']
            team_df.at[team_name, 'Adjusted Points Allowed'] = math.exp(points_reg_results.at['Opponent_' + team_name,
                                                                                     'points_reg_value'])
        else:
            team_df.at[team_name, 'Points Allowed Coef'] = 0
            team_df.at[team_name, 'Adjusted Points Allowed'] = math.exp(points_regression.intercept_)

    # yards_regression = Ridge(alpha=10, fit_intercept=True)
    yards_regression = PoissonRegressor(alpha=3.0, fit_intercept=True)
    yards_df = individual_df[['Team', 'Opponent', 'Yards']]
    yards_dummy_vars = pd.get_dummies(yards_df[['Team', 'Opponent']])

    yards_regression.fit(X=yards_dummy_vars, y=yards_df['Yards'])

    yards_reg_results = pd.DataFrame({
        'coef_name': yards_dummy_vars.columns.values,
        'yards_reg_coef': yards_regression.coef_
    })
    yards_reg_results['yards_reg_value'] = (yards_reg_results['yards_reg_coef'] + yards_regression.intercept_)
    yards_reg_results = yards_reg_results.set_index('coef_name')

    for team_name in team_df.index:
        team_df.at[team_name, 'Yards Intercept'] = yards_regression.intercept_

        if 'Team_' + team_name in yards_reg_results.index:
            team_df.at[team_name, 'Yards Coef'] = yards_reg_results.at['Team_' + team_name, 'yards_reg_coef']
            team_df.at[team_name, 'Adjusted Yards'] = math.exp(yards_reg_results.at['Team_' + team_name, 'yards_reg_value'])
        else:
            team_df.at[team_name, 'Yards Coef'] = 0
            team_df.at[team_name, 'Adjusted Yards'] = math.exp(yards_regression.intercept_)

        if 'Opponent_' + team_name in yards_reg_results.index:
            team_df.at[team_name, 'Yards Allowed Coef'] = yards_reg_results.at['Opponent_' + team_name,
                                                                               'yards_reg_coef']
            team_df.at[team_name, 'Adjusted Yards Allowed'] = math.exp(yards_reg_results.at['Opponent_' + team_name,
                                                                                   'yards_reg_value'])
        else:
            team_df.at[team_name, 'Yards Allowed Coef'] = 0
            team_df.at[team_name, 'Adjusted Yards Allowed'] = math.exp(yards_regression.intercept_)

    if not tie:
        winner = home_name if home_victory else away_name
        loser = away_name if home_victory else home_name
        graph.add_edge(loser, winner)
    else:
        graph.add_edge(away_name, home_name)
        graph.add_edge(home_name, away_name)

    bt_df = get_bradley_terry_from_graph(graph)

    bts = {index: row['BT'] for index, row in bt_df.iterrows()}
    bt_vars = {index: row['Var'] for index, row in bt_df.iterrows()}

    home_games_played = team_df.at[home_name, 'Games Played']
    away_games_played = team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Games Played'] = home_games_played + 1
    team_df.at[away_name, 'Games Played'] = away_games_played + 1

    team_df.at[home_name, 'Wins'] = team_df.at[home_name, 'Wins'] + 1 if home_victory else team_df.at[home_name, 'Wins']
    team_df.at[away_name, 'Wins'] = team_df.at[away_name, 'Wins'] + 1 if away_victory else team_df.at[away_name, 'Wins']

    team_df.at[home_name, 'Losses'] = team_df.at[home_name, 'Losses'] + 1 \
        if away_victory else team_df.at[home_name, 'Losses']
    team_df.at[away_name, 'Losses'] = team_df.at[away_name, 'Losses'] + 1 \
        if home_victory else team_df.at[away_name, 'Losses']

    team_df.at[home_name, 'Ties'] = team_df.at[home_name, 'Ties'] + 1 if tie else team_df.at[home_name, 'Ties']
    team_df.at[away_name, 'Ties'] = team_df.at[away_name, 'Ties'] + 1 if tie else team_df.at[away_name, 'Ties']

    for team_name in team_df.index:
        team_df.at[team_name, 'BT'] = bts.get(team_name)
        team_df.at[team_name, 'BT Var'] = bt_vars.get(team_name)

    team_df.at[home_name, 'Win Pct'] = team_df.at[home_name, 'Wins'] / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'Win Pct'] = team_df.at[away_name, 'Wins'] / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Bayes Win Pct'] = get_bayes_avg_wins(home_name)
    team_df.at[away_name, 'Bayes Win Pct'] = get_bayes_avg_wins(away_name)

    team_df.at[home_name, 'Avg Points'] = (team_df.at[home_name, 'Avg Points'] * home_games_played
                                           + home_points) / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'Avg Points'] = (team_df.at[away_name, 'Avg Points'] * away_games_played
                                           + away_points) / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Avg Points Allowed'] = (team_df.at[home_name, 'Avg Points Allowed'] * home_games_played
                                                   + away_points) / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'Avg Points Allowed'] = (team_df.at[away_name, 'Avg Points Allowed'] * away_games_played
                                                   + home_points) / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'YPG'] = (team_df.at[home_name, 'YPG'] * home_games_played
                                    + home_yards) / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'YPG'] = (team_df.at[away_name, 'YPG'] * away_games_played
                                    + away_yards) / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'YPG Allowed'] = (team_df.at[home_name, 'YPG Allowed'] * home_games_played
                                            + away_yards) / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'YPG Allowed'] = (team_df.at[away_name, 'YPG Allowed'] * away_games_played
                                            + home_yards) / team_df.at[away_name, 'Games Played']

    team_df = team_df.fillna(0)


def predict_game(model, pt, away_name, home_name, vegas_line):
    prediction_dict = dict()
    prediction_dict['Home Name'] = home_name
    prediction_dict['Away Name'] = away_name

    home_bt = team_df.at[home_name, 'BT']
    away_bt = team_df.at[away_name, 'BT']

    bt_chance = math.exp(home_bt) / (math.exp(home_bt) + math.exp(away_bt))
    prediction_dict['BT Chance'] = bt_chance

    home_bt_var = team_df.at[home_name, 'BT Var']
    away_bt_var = team_df.at[away_name, 'BT Var']
    home_bt_upper = home_bt + 1.96 * math.sqrt(home_bt_var)
    home_bt_lower = home_bt - 1.96 * math.sqrt(home_bt_var)
    away_bt_upper = away_bt + 1.96 * math.sqrt(away_bt_var)
    away_bt_lower = away_bt - 1.96 * math.sqrt(away_bt_var)
    bt_chance_lower = math.exp(home_bt_lower) / (math.exp(home_bt_lower) + math.exp(away_bt_upper))
    bt_chance_upper = math.exp(home_bt_upper) / (math.exp(home_bt_upper) + math.exp(away_bt_lower))
    prediction_dict['BT Chance Lower'] = bt_chance_lower
    prediction_dict['BT Chance Upper'] = bt_chance_upper

    home_bayes_avg_wins = team_df.at[home_name, 'Bayes Win Pct']
    away_bayes_avg_wins = team_df.at[away_name, 'Bayes Win Pct']
    prediction_dict['Home Bayes Wins'] = home_bayes_avg_wins
    prediction_dict['Away Bayes Wins'] = away_bayes_avg_wins

    home_points_coef = team_df.at[home_name, 'Points Coef']
    away_points_coef = team_df.at[away_name, 'Points Coef']
    home_points_allowed_coef = team_df.at[home_name, 'Points Allowed Coef']
    away_points_allowed_coef = team_df.at[away_name, 'Points Allowed Coef']

    home_yards_coef = team_df.at[home_name, 'Yards Coef']
    away_yards_coef = team_df.at[away_name, 'Yards Coef']
    home_yards_allowed_coef = team_df.at[home_name, 'Yards Allowed Coef']
    away_yards_allowed_coef = team_df.at[away_name, 'Yards Allowed Coef']

    home_expected_points = team_df.at[home_name, 'Points Intercept'] + home_points_coef + away_points_allowed_coef
    away_expected_points = team_df.at[away_name, 'Points Intercept'] + away_points_coef + home_points_allowed_coef
    prediction_dict['Home Expected Points'] = math.exp(home_expected_points)
    prediction_dict['Away Expected Points'] = math.exp(away_expected_points)

    home_expected_yards = team_df.at[home_name, 'Yards Intercept'] + home_yards_coef + away_yards_allowed_coef
    away_expected_yards = team_df.at[away_name, 'Yards Intercept'] + away_yards_coef + home_yards_allowed_coef
    prediction_dict['Home Expected Yards'] = math.exp(home_expected_yards)
    prediction_dict['Away Expected Yards'] = math.exp(away_expected_yards)

    expected_points_diff = home_expected_points - away_expected_points
    expected_yards_diff = home_expected_yards - away_expected_yards

    prediction_features = [vegas_line,
                           expected_points_diff,
                           expected_yards_diff,
                           bt_chance,
                           home_bayes_avg_wins,
                           away_bayes_avg_wins,
                           home_points_coef,
                           away_points_coef]

    prediction_features = np.asarray(prediction_features)
    prediction_features = prediction_features.reshape(1, -1)
    transformed = pt.transform(prediction_features)
    prediction = model.predict_proba(transformed)
    home_win_prob = prediction[0, 1]
    away_win_prob = prediction[0, 0]
    prediction_dict['Home Win Prob'] = home_win_prob
    prediction_dict['Away Win Prob'] = away_win_prob

    return prediction_dict


def order_predictions(model, pt, games_to_predict, verbose=False):
    justify_width = 12
    predictions = list()
    for away, home, line in games_to_predict:
        prediction_dict = predict_game(model, pt, away, home, line)
        home_win_prob = prediction_dict.get('Home Win Prob')
        prediction_dict['Line'] = line
        prediction_dict['Confidence'] = abs(home_win_prob - .5)
        prediction_dict['Favored'] = (home_win_prob > .5 and line <= 0) or (home_win_prob <= .5 and line > 0)
        predictions.append(prediction_dict)
    predictions = sorted(predictions, key=lambda d: d['Confidence'], reverse=True)

    for pred in predictions:
        home_name = pred.get('Home Name')
        away_name = pred.get('Away Name')
        line = pred.get('Line')
        home_win_prob = pred.get('Home Win Prob')
        away_win_prob = pred.get('Away Win Prob')

        winner = home_name if home_win_prob >= .5 else away_name
        loser = home_name if home_win_prob < .5 else away_name

        prob = home_win_prob if home_win_prob >= .5 else away_win_prob

        print('The', winner.ljust(justify_width), 'have a',
              f'{prob * 100:.3f}' + '% chance to beat the', loser.ljust(justify_width))

        if verbose:
            favored_team = home_name if line <= 0 else away_name
            location = 'at home' if line <= 0 else 'on the road'
            print('The', favored_team.ljust(justify_width), 'are favored by', round(abs(line), 1), 'points', location)

            home_expected_points = pred.get('Home Expected Points')
            away_expected_points = pred.get('Away Expected Points')
            winner = home_name if home_expected_points >= away_expected_points else away_name
            winner_points = home_expected_points if winner == home_name else away_expected_points
            loser_points = away_expected_points if winner == home_name else home_expected_points
            pts = 'points' if round(winner_points) - round(loser_points) != 1 else 'point'
            expected_score = '(' + str(round(winner_points)) + ' - ' + str(round(loser_points)) + ')'

            print('The', winner.ljust(justify_width), 'are expected to win by',
                  round(winner_points) - round(loser_points), pts.ljust(6), expected_score)

            home_bt = pred.get('BT Chance')
            home_bt_lower = pred.get('BT Chance Lower')
            home_bt_upper = pred.get('BT Chance Upper')
            favored_bt = home_name if home_bt >= .5 else away_name
            underdog_bt = away_name if home_bt >= .5 else home_name
            bt = home_bt if home_bt >= .5 else 1 - home_bt
            bt_lower = home_bt_lower if home_bt >= .5 else 1 - home_bt_lower
            bt_upper = home_bt_upper if home_bt >= .5 else 1 - home_bt_upper
            if bt_lower > bt_upper:
                temp = bt_lower
                bt_lower = bt_upper
                bt_upper = temp
            print('The', favored_bt.ljust(justify_width), 'have a',
                  f'{bt * 100:.3f}' + '% chance to beat the', underdog_bt.ljust(justify_width + 3),
                  'according to the BT Model')
            # '- 95% CI (' + f'{bt_lower * 100:.3f}' + '%, ' + f'{bt_upper * 100:.3f}' + '%)')

            home_win_pct = pred.get('Home Bayes Wins')
            away_win_pct = pred.get('Away Bayes Wins')
            higher_name = home_name if home_win_pct >= away_win_pct else away_name
            lower_name = away_name if home_win_pct >= away_win_pct else home_name
            higher_wp = home_win_pct if home_win_pct >= away_win_pct else away_win_pct
            lower_wp = away_win_pct if home_win_pct >= away_win_pct else home_win_pct
            print('The', higher_name.ljust(justify_width), 'are on pace to be a',
                  str(round(higher_wp * 17)).ljust(2), 'win team, the', lower_name.ljust(justify_width),
                  'are on pace to be a', str(round(lower_wp * 17)), 'win team')

            home_expected_yards = pred.get('Home Expected Yards')
            away_expected_yards = pred.get('Away Expected Yards')
            winner = home_name if home_expected_yards >= away_expected_yards else away_name
            loser = away_name if home_expected_yards >= away_expected_yards else home_name
            winner_yards = home_expected_yards if winner == home_name else away_expected_yards
            loser_yards = away_expected_yards if winner == home_name else home_expected_yards
            yds = 'yards' if round(winner_yards - loser_yards) != 1 else 'yard'
            expected_yards = '(' + str(round(winner_yards)) + ' - ' + str(round(loser_yards)) + ')'

            print('The', winner.ljust(justify_width), 'are expected to out gain the', loser.ljust(justify_width + 8),
                  'by',
                  round(winner_yards - loser_yards), yds.ljust(5), expected_yards)
            print()

    print()


def order_predictions_bt(games_to_predict):
    predictions = list()
    for away, home, line in games_to_predict:
        home_bt = team_df.at[home, 'BT']
        away_bt = team_df.at[away, 'BT']
        home_win_prob = math.exp(home_bt) / (math.exp(home_bt) + math.exp(away_bt))
        away_win_prob = math.exp(away_bt) / (math.exp(away_bt) + math.exp(home_bt))
        confidence = abs(home_win_prob - .5)
        predictions.append((home, away, home_win_prob, away_win_prob, confidence))
    predictions = sorted(predictions, key=lambda t: t[-1], reverse=True)

    for prediction in predictions:
        winner = prediction[0] if prediction[2] >= .5 else prediction[1]
        loser = prediction[0] if prediction[2] < .5 else prediction[1]

        prob = prediction[2] if prediction[2] >= .5 else prediction[3]
        print('The', winner.ljust(10), 'have a',
              str(round(100 * prob, 3)).ljust(6, '0') + '% chance to beat the', loser)
    print()


def get_remaining_win_probs(team_name):
    schedule = load_schedule().get('weeks')
    opponents = list()
    for week in schedule:
        for game in week:
            if game.get('away') == team_name:
                opponent = game.get('home')
                opponents.append(opponent)
            if game.get('home') == team_name:
                opponent = game.get('away')
                opponents.append(opponent)

    bt = team_df.at[team_name, 'BT']
    wins = team_df.at[team_name, 'Wins']
    losses = team_df.at[team_name, 'Losses']
    ties = team_df.at[team_name, 'Ties']
    games_played = wins + losses + ties
    if games_played == 17:
        return []

    remaining_opponents = opponents[games_played:17]
    opponent_bts = [team_df.at[opponent, 'BT'] for opponent in remaining_opponents]
    win_probs = [math.exp(bt) / (math.exp(bt) + math.exp(opp_bt)) for opp_bt in opponent_bts]
    return win_probs


def get_proj_record(team_name):
    win_probs = get_remaining_win_probs(team_name)
    wins = team_df.at[team_name, 'Wins']
    ties = team_df.at[team_name, 'Ties']

    expected_wins = sum(win_probs) + wins
    expected_losses = 17 - expected_wins - ties

    return round(expected_wins), round(expected_losses), ties


def simulate_season(team_name, num_sims=10000):
    win_probs = get_remaining_win_probs(team_name)
    wins = team_df.at[team_name, 'Wins']
    ties = team_df.at[team_name, 'Ties']

    all_simulations = list()
    for simulation in range(num_sims):
        outcomes = [1 if prob > random.random() else 0 for prob in win_probs]
        simulated_wins = sum(outcomes) + wins
        all_simulations.append(simulated_wins)

    sns.histplot(all_simulations, stat='percent', kde=True, kde_kws={'bw_adjust': 5}, binwidth=1)
    plt.show()


def get_color(value, variance=np.nan, alpha=.05, enabled=True, invert=False):
    if not enabled:
        return value

    green = '\033[32m'
    red = '\033[31m'
    stop = '\033[0m'

    if pd.isna(variance):
        value = round(value, 3)
        return red + str(value) + stop if value < 0 else green + str(value) + stop if value > 0 else value
    elif variance == 0:
        return ''
    else:
        normal = norm(value, math.sqrt(variance))
        if (not invert and normal.ppf(alpha) > 0) or (invert and normal.ppf(1 - alpha) < 0):
            return green
        if (not invert and normal.ppf(1 - alpha) < 0) or (invert and normal.ppf(alpha) > 0):
            return red
        return ''


def rescale_bt(original_bt, avg_value=0.0):
    if avg_value == 0:
        return original_bt
    new_bt = math.exp(original_bt) * avg_value
    return new_bt


def print_table(week, sort_key='BT', sort_by_division=False):
    global team_df

    if sort_key in ['Avg Points Allowed', 'Points Allowed Coef', 'Adjusted Points Allowed',
                    'YPG Allowed', 'Yards Allowed Coef', 'Adjusted Yards Allowed']:
        ascending_order = True
    else:
        ascending_order = False
    team_df = team_df.sort_values(by=sort_key, kind='mergesort', ascending=ascending_order)
    if sort_by_division:
        team_df = team_df.sort_values(by='Division', kind='mergesort', ascending=False)

    if week > 18:
        table = PrettyTable(['Rank', 'Name', 'Record', 'Bayes Win Pct', 'BT',
                             'Adj. PPG', 'Adj. YPG', 'Adj. PPG Allowed', 'Adj. YPG Allowed'])
    else:
        table = PrettyTable(['Rank', 'Name', 'Record', 'Bayes Win Pct', 'BT',
                             'Proj. Record', 'Adj. PPG', 'Adj. YPG', 'Adj. PPG Allowed', 'Adj. YPG Allowed'])
    table.float_format = '0.3'

    points_coefs = team_df['Points Coef']
    points_allowed_coefs = team_df['Points Allowed Coef']
    yards_coefs = team_df['Yards Coef']
    yards_allowed_coefs = team_df['Yards Allowed Coef']

    points_var = statistics.variance(points_coefs)
    points_allowed_var = statistics.variance(points_allowed_coefs)
    yards_var = statistics.variance(yards_coefs)
    yards_allowed_var = statistics.variance(yards_allowed_coefs)

    stop = '\033[0m'

    for index, row in team_df.iterrows():
        table_row = list()

        wins = row['Wins']
        losses = row['Losses']
        ties = row['Ties']
        record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]]) + ' - ' + str(int(ties))

        rank = team_df.index.get_loc(index) + 1

        points_pct = .1
        yards_pct = .1
        points_color = get_color(row['Points Coef'], points_var, alpha=points_pct)
        points_allowed_color = get_color(row['Points Allowed Coef'], points_allowed_var, alpha=points_pct, invert=True)

        yards_color = get_color(row['Yards Coef'], yards_var, alpha=yards_pct)
        yards_allowed_color = get_color(row['Yards Allowed Coef'], yards_allowed_var, alpha=yards_pct, invert=True)

        table_row.append(rank)
        table_row.append(index)
        table_row.append(record)
        table_row.append(row['Bayes Win Pct'])

        bt_color = get_color(row['BT'], row['BT Var'])
        table_row.append(bt_color + str(round(rescale_bt(row['BT']), 3)) + stop)

        if week <= 18:
            proj_record = get_proj_record(index)
            ties = proj_record[-1]
            proj_record = ' - '.join([str(val).rjust(2) for val in proj_record[:-1]]) + ' - ' + str(int(ties))
            table_row.append(proj_record)

        table_row.append(points_color + str(round(row['Adjusted Points'], 1)) + stop)
        table_row.append(yards_color + str(round(row['Adjusted Yards'], 1)) + stop)
        table_row.append(points_allowed_color + str(round(row['Adjusted Points Allowed'], 1)) + stop)
        table_row.append(yards_allowed_color + str(round(row['Adjusted Yards Allowed'], 1)) + stop)

        table.add_row(table_row)

    # Print the table
    print('Rankings')
    print(table)
    print()


def get_vegas_line(home_name, away_name, odds):
    matching_odds = [odd for odd in odds if (away_name in odd[0][0] or home_name in odd[0][0]) and
                     (away_name in odd[0][1] or home_name in odd[0][1])]
    if len(matching_odds) != 1:
        print('Odds not found for', away_name, '@', home_name)
        return 0
    else:
        matching_odds = matching_odds[0]
        return matching_odds[-1]


def surprises():
    last_year_wins = {'Chiefs': 14,
                      'Bills': 13,
                      'Bengals': 12,
                      'Chargers': 10,
                      'Ravens': 10,
                      'Jaguars': 9,
                      'Dolphins': 9,
                      'Steelers': 9,
                      'Patriots': 8,
                      'Jets': 7,
                      'Titans': 7,
                      'Browns': 7,
                      'Raiders': 6,
                      'Broncos': 5,
                      'Colts': 4,
                      'Texans': 3,

                      'Eagles': 14,
                      '49ers': 13,
                      'Vikings': 13,
                      'Cowboys': 12,
                      'Giants': 9,
                      'Seahawks': 9,
                      'Lions': 9,
                      'Commanders': 8,
                      'Buccaneers': 8,
                      'Packers': 8,
                      'Panthers': 7,
                      'Saints': 7,
                      'Falcons': 7,
                      'Rams': 5,
                      'Cardinals': 4,
                      'Bears': 3}

    surprise_dict = dict()
    for team in team_df.index:
        team_wins = last_year_wins.get(team)

        if team == 'Steelers' or team == 'Lions':
            team_wp = team_wins / 16
        else:
            team_wp = team_wins / 17

        proj_wins, proj_losses, proj_ties = get_proj_record(team)
        proj_wp = proj_wins / (proj_wins + proj_losses)
        surprise_dict[team] = team_wp - proj_wp

    surprise_dict = {k: v for k, v in sorted(surprise_dict.items(), key=lambda item: abs(item[1]), reverse=True)}

    green = '\033[32m'
    red = '\033[31m'
    stop = '\033[0m'

    disappointment_dict = {k: v for k, v in surprise_dict.items() if v > .2}
    surprise_dict = {k: v for k, v in surprise_dict.items() if v < -.2}

    print('Biggest Surprises')
    for team, difference in surprise_dict.items():
        last_wins = last_year_wins.get(team)

        if team == 'Steelers' or team == 'Lions':
            team_wp = last_wins / 16
        else:
            team_wp = last_wins / 17

        proj_wp = team_wp - difference
        proj_wins = proj_wp * 17

        print(green, 'The', team.ljust(12), 'are on pace to win', str(round(proj_wins)).ljust(2),
              'games, last year they won', last_wins, stop)
    print()

    print('Biggest Disappointments')
    for team, difference in disappointment_dict.items():
        last_wins = last_year_wins.get(team)

        if team == 'Steelers' or team == 'Lions':
            team_wp = last_wins / 16
        else:
            team_wp = last_wins / 17

        proj_wp = team_wp - difference
        proj_wins = proj_wp * 17

        print(red, 'The', team.ljust(12), 'are on pace to win', str(round(proj_wins)),
              'games, last year they won', last_wins, stop)
    print()


def bt_chance_matrix(home_offset=True):
    home_field_win_pct = .58
    home_bt_offset = -1 * math.log(1 / home_field_win_pct - 1)
    home_bt_offset = home_bt_offset if home_offset else 0

    chance_df = pd.DataFrame(columns=team_df.index, index=team_df.index)
    for team1, team2 in itertools.permutations(team_df.index, 2):
        team1_bt = team_df.at[team1, 'BT']
        team2_bt = team_df.at[team2, 'BT']

        if (team1 == 'Chiefs' and team2 == 'Bills') or (team1 == 'Bills' and team2 == 'Chiefs'):
            team1_win_prob = math.exp(team1_bt) / (math.exp(team1_bt) + math.exp(team2_bt))
        else:
            team1_win_prob = math.exp(team1_bt + home_bt_offset) / (math.exp(team1_bt + home_bt_offset) +
                                                                    math.exp(team2_bt))
        chance_df.at[team2, team1] = team1_win_prob

    return chance_df


def get_schedule_difficulties():
    global team_df

    schedule = load_schedule()
    weeks = schedule.get('weeks')

    team_schedule = dict()
    for team in team_df.index:
        opponents = list()
        for week in weeks:
            for game in week:
                if game.get('away') == team:
                    opponents.append(game.get('home'))

                if game.get('home') == team:
                    opponents.append(game.get('away'))

        opponent_bts = [math.exp(team_df.at[opponent, 'BT']) for opponent in opponents]
        team_schedule[team] = opponent_bts

    team_opponent_bts = {team: math.log(statistics.mean(bts)) for team, bts in team_schedule.items()}
    team_opponent_bts = {k: v for k, v in sorted(team_opponent_bts.items(), key=lambda item: item[1], reverse=True)}

    team_win_chances = {team: [1 / (1 + bt) for bt in bts] for team, bts in team_schedule.items()}
    team_average_wins = {team: round(sum(chances)) for team, chances in team_win_chances.items()}

    team_df = team_df.sort_values(by='BT', kind='mergesort', ascending=False)
    table = PrettyTable(['Rank', 'Name', 'Record', 'BT', 'Avg. Opponent BT', 'Average Team Record'])
    table.float_format = '0.3'

    for index, team_info in enumerate(team_opponent_bts.items()):
        team, avg_opp_bt = team_info
        table_row = list()

        wins = team_df.at[team, 'Wins']
        losses = team_df.at[team, 'Losses']
        ties = team_df.at[team, 'Ties']

        record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]]) + ' - ' + str(int(ties))

        average_wins = team_average_wins.get(team)
        avg_record = ' - '.join([str(val).rjust(2) for val in [average_wins, 17 - average_wins, 0]])

        rank = index + 1

        table_row.append(rank)
        table_row.append(team)
        table_row.append(record)
        table_row.append(team_df.at[team, 'BT'])
        table_row.append(avg_opp_bt)
        table_row.append(avg_record)

        table.add_row(table_row)

    # Print the table
    print('Team Schedule Difficulties')
    print(table)
    print()


def season(week_num,
           use_bt_predictions=False,
           manual_odds=False,
           include_parity=False,
           verbose=True):
    global graph
    global team_df
    global game_df

    model = load_model()
    pt = load_pt()
    schedule = load_schedule()
    if use_bt_predictions or manual_odds:
        odds = []
    else:
        odds = Odds.get_odds()

    teams = pd.Series(['49ers', 'Bears', 'Bengals', 'Bills', 'Broncos', 'Browns', 'Buccaneers', 'Cardinals',
                       'Chargers', 'Chiefs', 'Colts', 'Commanders', 'Cowboys', 'Dolphins', 'Eagles', 'Falcons',
                       'Giants', 'Jaguars', 'Jets', 'Lions', 'Packers', 'Panthers', 'Patriots', 'Raiders',
                       'Rams', 'Ravens', 'Saints', 'Seahawks', 'Steelers', 'Texans', 'Titans', 'Vikings'])

    primary_colors = pd.Series(['#AA0000', '#0B162A', '#FB4F14', '#00338D', '#FB4F14', '#311D00', '#D50A0A', '#97233F',
                                '#0080C6', '#E31837', '#002C5F', '#5A1414', '#003594', '#008E97', '#004C54', '#A71930',
                                '#0B2265', '#006778', '#125740', '#0076B6', '#203731', '#0085CA', '#002244', '#000000',
                                '#003594', '#241773', '#D3BC8D', '#69BE28', '#FFB612', '#03202F', '#4B92DB', '#4F2683'])

    second_colors = pd.Series(['#B3995D', '#C83803', '#000000', '#C60C30', '#002244', '#FF3C00', '#FF7900', '#000000',
                               '#FFC20E', '#FFB81C', '#A2AAAD', '#FFB612', '#869397', '#FC4C02', '#A5ACAF', '#000000',
                               '#A71930', '#D7A22A', '#000000', '#B0B7BC', '#FFB612', '#101820', '#C60C30', '#A5ACAF',
                               '#FFA300', '#000000', '#101820', '#002244', '#101820', '#A71930', '#0C2340', '#FFC62F'])

    team_df['Team'] = teams
    team_df['Division'] = team_df.apply(lambda r: get_division(r['Team']), axis=1)
    team_df['Primary Colors'] = primary_colors
    team_df['Secondary Colors'] = second_colors
    team_df = team_df.set_index('Team')

    team_df = team_df.fillna(0)

    if week_num <= 1:
        preseason_bts = get_preseason_bts()
        for team, bt in preseason_bts.items():
            team_df.at[team, 'BT'] = bt
            team_df.at[team, 'BT Var'] = .15
            team_df.at[team, 'Bayes Win Pct'] = .5
        print('Preseason')
        print_table(week_num)

    all_week_results = get_games_before_week(week_num, use_persisted=True)

    for week in range(week_num):
        get_game_results(week + 1, all_week_results)

        if week == week_num - 1:
            print('Week', week + 1)
            games = list()
            for game in schedule.get('weeks')[week]:
                home_name = game.get('home')
                away_name = game.get('away')
                if manual_odds:
                    odds = game.get('line')
                    games.append((away_name, home_name, odds))
                else:
                    games.append((away_name, home_name, get_vegas_line(home_name, away_name, odds)))

            if use_bt_predictions:
                order_predictions_bt(games)
            else:
                order_predictions(model, pt, games, verbose=verbose)

            if include_parity:
                parity_clock()
            bt_chance_matrix()
            print_table(week, sort_by_division=False)

    show_graph(divisional_edges_only=week_num > 8)

    if week_num <= 18:
        surprises()
        get_schedule_difficulties()


def bt_with_home_field():
    home_field_win_pct = .58
    home_bt_offset = -1 * math.log(1 / home_field_win_pct - 1)

    afc_teams = {'Chiefs': 1,
                 'Bills': 2,
                 'Bengals': 3,
                 'Jaguars': 4,
                 'Chargers': 5,
                 'Ravens': 6,
                 'Dolphins': 7}

    nfc_teams = {'Eagles': 1,
                 '49ers': 2,
                 'Vikings': 3,
                 'Buccaneers': 4,
                 'Cowboys': 5,
                 'Giants': 6,
                 'Seahawks': 7}

    justify_width = 12

    for team1, team2 in itertools.combinations(afc_teams.keys(), 2):
        home_team = team1 if afc_teams.get(team1) < afc_teams.get(team2) else team2
        away_team = team2 if afc_teams.get(team1) < afc_teams.get(team2) else team1

        print(away_team, 'at', home_team)

        home_bt = team_df.at[home_team, 'BT']
        away_bt = team_df.at[away_team, 'BT']

        home_chance = math.exp(home_bt) / (math.exp(home_bt) + math.exp(away_bt))
        home_chance_adj = math.exp(home_bt + home_bt_offset) / (math.exp(home_bt + home_bt_offset) + math.exp(away_bt))

        winner = home_team if home_chance >= .5 else away_team
        loser = away_team if home_chance >= .5 else home_team
        winner_chance = home_chance if home_chance >= .5 else 1 - home_chance

        winner_adj = home_team if home_chance_adj >= .5 else away_team
        loser_adj = away_team if home_chance_adj >= .5 else home_team
        winner_chance_adj = home_chance_adj if home_chance_adj >= .5 else 1 - home_chance_adj

        print('The', winner.ljust(justify_width), 'have a',
              f'{winner_chance * 100:.3f}' + '% chance to beat the', loser.ljust(justify_width))

        print('The', winner_adj.ljust(justify_width), 'have a',
              f'{winner_chance_adj * 100:.3f}' + '% chance to beat the', loser_adj.ljust(justify_width),
              'accounting for home field advantage')

        print()

    print()

    for team1, team2 in itertools.combinations(nfc_teams.keys(), 2):
        home_team = team1 if nfc_teams.get(team1) < nfc_teams.get(team2) else team2
        away_team = team2 if nfc_teams.get(team1) < nfc_teams.get(team2) else team1

        print(away_team, 'at', home_team)

        home_bt = team_df.at[home_team, 'BT']
        away_bt = team_df.at[away_team, 'BT']

        home_chance = math.exp(home_bt) / (math.exp(home_bt) + math.exp(away_bt))
        home_chance_adj = math.exp(home_bt + home_bt_offset) / (math.exp(home_bt + home_bt_offset) + math.exp(away_bt))

        winner = home_team if home_chance >= .5 else away_team
        loser = away_team if home_chance >= .5 else home_team
        winner_chance = home_chance if home_chance >= .5 else 1 - home_chance

        winner_adj = home_team if home_chance_adj >= .5 else away_team
        loser_adj = away_team if home_chance_adj >= .5 else home_team
        winner_chance_adj = home_chance_adj if home_chance_adj >= .5 else 1 - home_chance_adj

        print('The', winner.ljust(justify_width), 'have a',
              f'{winner_chance * 100:.3f}' + '% chance to beat the', loser.ljust(justify_width))

        print('The', winner_adj.ljust(justify_width), 'have a',
              f'{winner_chance_adj * 100:.3f}' + '% chance to beat the', loser_adj.ljust(justify_width),
              'accounting for home field advantage')

        print()


def get_preseason_bts(use_mse=True):
    win_totals = {'Chiefs': 11.5,
                  'Bengals': 11.5,
                  'Eagles': 11.5,
                  '49ers': 10.5,
                  'Bills': 10.5,
                  'Ravens': 10.5,
                  'Cowboys': 9.5,
                  'Jaguars': 9.5,
                  'Chargers': 9.5,
                  'Lions': 9.5,
                  'Jets': 9.5,
                  'Dolphins': 9.5,
                  'Saints': 9.5,
                  'Browns': 9.5,
                  'Steelers': 8.5,
                  'Seahawks': 8.5,
                  'Falcons': 8.5,
                  'Vikings': 8.5,
                  'Broncos': 8.5,
                  'Bears': 7.5,
                  'Packers': 7.5,
                  'Titans': 7.5,
                  'Giants': 7.5,
                  'Panthers': 7.5,
                  'Patriots': 7.5,
                  'Colts': 6.5,
                  'Commanders': 6.5,
                  'Rams': 6.5,
                  'Raiders': 6.5,
                  'Buccaneers': 6.5,
                  'Texans': 6.5,
                  'Cardinals': 4.5}

    schedule = load_schedule()

    schedules = {team: [] for team in win_totals.keys()}
    for week in schedule.get('weeks'):
        for game in week:
            home = game.get('home')
            away = game.get('away')

            schedules.get(home).append(away)
            schedules.get(away).append(home)

    teams = list(win_totals.keys())
    bts = np.zeros(len(teams))
    win_proj = np.array(list(win_totals.values()))

    def objective(params):
        val = np.float64(0)

        for team, opponents in schedules.items():
            team_proj = np.float64(0)
            team_index = teams.index(team)
            for opponent in opponents:
                opponent_index = teams.index(opponent)

                team_proj += 1 / np.exp(np.logaddexp(0, -(params[team_index] - params[opponent_index])))

            if use_mse:
                val += (win_proj[team_index] - team_proj) ** 2
            else:
                val += np.abs(win_proj[team_index] - team_proj)

        return val

    res = minimize(objective, bts, method='Powell', jac=False)

    def get_bt_prob(bt1, bt2):
        return math.exp(bt1) / (math.exp(bt1) + math.exp(bt2))

    team_bts = {team: bt for team, bt in zip(teams, res.x)}

    rows = list()
    for team, opponents in schedules.items():
        proj_wins = sum([get_bt_prob(team_bts.get(team), team_bts.get(opponent)) for opponent in opponents])
        diff = proj_wins - win_totals.get(team)

        row = {'Team': team,
               'BT': team_bts.get(team),
               'BT Projection': proj_wins,
               'Odds Projection': win_totals.get(team),
               'Diff': diff,
               'Abs Diff': abs(diff)}
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by='BT', ascending=False)
    df = df.reset_index(drop=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    df['Rank'] = range(1, 33)
    df = df.set_index('Rank', drop=True)
    df = df[['Team', 'BT', 'Odds Projection', 'BT Projection']]
    df['BT Projection'] = df['BT Projection'].round(1)

    return team_bts


def get_total_wins_chances(team):
    wins = team_df.at[team, 'Wins']
    wins_dict = {win_total: 0.0 for win_total in range(18)}

    win_probs = get_remaining_win_probs(team)
    loss_probs = [1 - win_prob for win_prob in win_probs]

    win_mask = list(itertools.product([0, 1], repeat=len(win_probs)))
    for win_combo in win_mask:
        loss_combo = [0 if game == 1 else 1 for game in win_combo]

        win_combo_probs = list(itertools.compress(win_probs, win_combo))
        loss_combo_probs = list(itertools.compress(loss_probs, loss_combo))
        win_combo_wins = len(win_combo_probs) + wins

        total_wins_prob = np.prod(win_combo_probs)
        total_losses_prob = np.prod(loss_combo_probs)

        combo_prob = total_wins_prob * total_losses_prob

        wins_dict[win_combo_wins] = wins_dict.get(win_combo_wins) + combo_prob

    return wins_dict


def show_graph(divisional_edges_only=True):
    warnings.filterwarnings("ignore")

    sns.set(style="ticks")

    nfl = graph.copy()

    # Format and title the graph
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_aspect('auto')
    ax.set_title('League Graph')
    ax.set_facecolor('#FAFAFA')

    # Get the Pagerank of each node
    bts = {team: row['BT'] for team, row in team_df.iterrows()}
    primary = {team: row['Primary Colors'] for team, row in team_df.iterrows()}
    secondary = {team: row['Secondary Colors'] for team, row in team_df.iterrows()}
    max_bt = team_df['BT'].max()
    min_bt = team_df['BT'].min()
    bt_dev = statistics.stdev(team_df['BT'])
    subset = {team: np.digitize(row['BT'], np.arange(min_bt, max_bt, bt_dev / 2)) for team, row in team_df.iterrows()}

    nx.set_node_attributes(nfl, bts, 'BT')
    nx.set_node_attributes(nfl, primary, 'Primary')
    nx.set_node_attributes(nfl, secondary, 'Secondary')
    nx.set_node_attributes(nfl, subset, 'subset')

    images = {team: PIL.Image.open('Projects/nfl/NFL_Prediction/Redo/logos/' + team + '.png')
              for team, row in team_df.iterrows()}
    nx.set_node_attributes(nfl, images, 'image')

    pos = nx.multipartite_layout(nfl, align='horizontal')

    if divisional_edges_only:
        edge_list = [(t1, t2) for t1, t2 in nfl.edges() if get_division(t1) == get_division(t2)]
    else:
        edge_list = nfl.edges()
    vertical_edge_list = [(t1, t2) for t1, t2 in edge_list if subset.get(t1) != subset.get(t2)]
    horizontal_edge_list = [(t1, t2) for t1, t2 in edge_list if subset.get(t1) == subset.get(t2)]

    # Draw the edges in the graph
    for source, target in edge_list:
        conn_stlye = "Arc3, rad=0.2" if subset.get(source) == subset.get(target) else "Arc3, rad=0.05"
        target_bt = bts.get(target)
        target_margin = math.exp(target_bt) * 18

        nx.draw_networkx_edges(nfl,
                               pos,
                               edgelist=[(source, target)],
                               width=1,
                               alpha=0.1,
                               edge_color='black',
                               connectionstyle=conn_stlye,
                               arrowsize=10,
                               min_target_margin=target_margin)


    # Select the size of the image (relative to the X axis)
    icon_size = {team: (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025 * math.exp(bt) for team, bt in bts.items()}
    icon_center = {team: size / 2.0 for team, size in icon_size.items()}

    for n in nfl.nodes:
        xa, ya = fig.transFigure.inverted().transform(ax.transData.transform(pos[n]))
        a = plt.axes([xa - icon_center.get(n), ya - icon_center.get(n), icon_size.get(n), icon_size.get(n)])
        a.set_aspect('auto')
        a.imshow(nfl.nodes[n]['image'])
        a.axis("off")

    plt.savefig('D:\\Colin\\Documents\\Programming\\Python\\PythonProjects\\Projects\\nfl\\'
                'NFL_Prediction\\Redo\\LeagueGraph.png', dpi=300)

    # Show the graph
    # plt.show()
