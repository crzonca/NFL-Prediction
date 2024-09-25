import math
import statistics

import networkx as nx
import numpy as np
import pandas as pd
import warnings
from statsmodels.discrete.discrete_model import GeneralizedPoisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
from prettytable import PrettyTable
from scipy.stats import chi2, norm, skellam

import Projects.nfl.NFL_Prediction.OddsHelper as Odds
from Projects.nfl.NFL_Prediction.Pred import league_structure
from Projects.nfl.NFL_Prediction.Pred.betting import Bettor
from Projects.nfl.NFL_Prediction.Pred.evaluation import LeagueEvaluator
from Projects.nfl.NFL_Prediction.Pred.helper import Helper
from Projects.nfl.NFL_Prediction.Pred import playoff_chances as pc
from Projects.nfl.NFL_Prediction.Pred.playoff_chances import PlayoffPredictor

graph = nx.MultiDiGraph()
gen_poisson_model = None
team_df = pd.DataFrame(columns=['Team', 'Division', 'Games Played', 'Wins', 'Losses', 'Ties',
                                'BT', 'BT Var', 'BT Pct', 'Bayes BT', 'Win Pct', 'Bayes Win Pct',
                                'Avg Points', 'Avg Points Allowed', 'Bayes Avg Points', 'Bayes Avg Points Allowed',
                                'Adjusted Points', 'Adjusted Points Allowed', 'Adjusted Point Diff',
                                'Bayes Adjusted Points', 'Bayes Adjusted Points Allowed', 'Bayes Adjusted Point Diff'])

game_df = pd.DataFrame(columns=['Team', 'Win', 'Points', 'Points Allowed'])
individual_df = pd.DataFrame(columns=['Game ID', 'Team', 'Opponent', 'Points',
                                      'Drives', 'PPAvg', 'Game Num', 'Is_Home'])


def get_game_results(week, week_results, current_week=False):
    week_results = week_results.loc[week_results['week'] == week]

    games_dict = dict()
    for index, row in week_results.iterrows():
        team = league_structure.get_name_from_abbrev(row['team'])
        points = row['points_scored']
        drives = row['drives']
        location = row['location']
        game_id = row['boxscore_index']

        games_dict[game_id + ' ' + location] = (team, points, drives, location, game_id)

    game_ids = {game[-1] for key, game in games_dict.items()}
    for game_id in game_ids:
        matching_games = [game for key, game in games_dict.items() if game[-1] == game_id]
        home_version = [game for game in matching_games if game[3] == 'Home'][0]
        away_version = [game for game in matching_games if game[3] == 'Away'][0]

        set_game_outcome(game_id, home_version[0], away_version[0],
                         home_version[1], away_version[1],
                         home_version[2], away_version[2])

    fit_gen_poisson(verbose=current_week)
    fit_bt()
    fit_drives()


def set_game_outcome(game_id, home_name, away_name, home_points, away_points, home_drives, away_drives):
    global graph
    global team_df
    global game_df
    global individual_df

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)

    home_victory = home_points > away_points
    away_victory = away_points > home_points
    tie = not home_victory and not away_victory

    # Update Game DF
    game_df.loc[len(game_df.index)] = [home_name, 1 if home_victory else 0, home_points, away_points]
    game_df.loc[len(game_df.index)] = [away_name, 1 if away_victory else 0, away_points, home_points]

    # Update Individual DF
    home_game_num = len(individual_df.loc[individual_df['Team'] == home_name]) + 1
    away_game_num = len(individual_df.loc[individual_df['Team'] == away_name]) + 1

    individual_df.loc[len(individual_df.index)] = [game_id, home_name, away_name, home_points, home_drives, 0,
                                                   home_game_num, 1]
    individual_df.loc[len(individual_df.index)] = [game_id, away_name, home_name, away_points, away_drives, 0,
                                                   away_game_num, 0]

    if not tie:
        winner = home_name if home_victory else away_name
        loser = away_name if home_victory else home_name
        graph.add_edge(loser, winner)
    else:
        graph.add_edge(away_name, home_name)
        graph.add_edge(home_name, away_name)

    # Update Team DF
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

    team_df.at[home_name, 'Win Pct'] = team_df.at[home_name, 'Wins'] / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'Win Pct'] = team_df.at[away_name, 'Wins'] / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Bayes Win Pct'] = helper.get_bayes_avg_wins(game_df, home_name)
    team_df.at[away_name, 'Bayes Win Pct'] = helper.get_bayes_avg_wins(game_df, away_name)

    team_df.at[home_name, 'Avg Points'] = (team_df.at[home_name, 'Avg Points'] * home_games_played
                                           + home_points) / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'Avg Points'] = (team_df.at[away_name, 'Avg Points'] * away_games_played
                                           + away_points) / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Bayes Avg Points'] = helper.get_bayes_avg_points(game_df, home_name)
    team_df.at[away_name, 'Bayes Avg Points'] = helper.get_bayes_avg_points(game_df, away_name)

    team_df.at[home_name, 'Avg Points Allowed'] = (team_df.at[home_name, 'Avg Points Allowed'] * home_games_played
                                                   + away_points) / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'Avg Points Allowed'] = (team_df.at[away_name, 'Avg Points Allowed'] * away_games_played
                                                   + home_points) / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Bayes Avg Points Allowed'] = helper.get_bayes_avg_points(game_df, home_name, allowed=True)
    team_df.at[away_name, 'Bayes Avg Points Allowed'] = helper.get_bayes_avg_points(game_df, away_name, allowed=True)

    team_df = team_df.fillna(0)


def fit_gen_poisson(verbose=False):
    global team_df
    global gen_poisson_model

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    average_drives = helper.predict_drives_team_averages('', '')
    individual_df['PPAvg'] = individual_df.apply(lambda r: (r['Points'] / r['Drives']) * average_drives, axis=1)

    if individual_df.empty or len(individual_df) < 96:
        team_df['Adjusted Points'] = 21
        team_df['Adjusted Points Allowed'] = 21
        team_df['Adjusted Point Diff'] = team_df.apply(lambda r: r['Adjusted Points'] - r['Adjusted Points Allowed'],
                                                       axis=1)

        team_df['Bayes Adjusted Points'] = 21
        team_df['Bayes Adjusted Points Allowed'] = 21
        team_df['Bayes Adjusted Point Diff'] = team_df.apply(
            lambda r: r['Bayes Adjusted Points'] - r['Bayes Adjusted Points Allowed'], axis=1)

        team_df = team_df.fillna(0)
        return

    response = individual_df['Points']
    explanatory = pd.get_dummies(individual_df[['Team', 'Opponent']])
    explanatory = sm.add_constant(explanatory)

    gen_poisson = GeneralizedPoisson(endog=response,
                                     exog=explanatory,
                                     p=1)

    gen_poisson_model = gen_poisson.fit_regularized(method='l1', maxiter=int(1e6), alpha=.2)

    if verbose:
        print(gen_poisson_model.summary())

    series_index = explanatory.columns

    for team in team_df.index:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf_prediction_series = pd.Series(index=series_index)
            pf_prediction_series.at['const'] = 1.0
            pf_prediction_series.at['Team_' + team] = 1.0
            pf_prediction_series = pf_prediction_series.fillna(0.0)

            adj_points_for = gen_poisson_model.predict(pf_prediction_series).squeeze()
            team_df.at[team, 'Adjusted Points'] = adj_points_for

            pa_prediction_series = pd.Series(index=series_index)
            pa_prediction_series.at['const'] = 1.0
            pa_prediction_series.at['Opponent_' + team] = 1.0
            pa_prediction_series = pa_prediction_series.fillna(0.0)

            adj_points_allowed = gen_poisson_model.predict(pa_prediction_series).squeeze()
            team_df.at[team, 'Adjusted Points Allowed'] = adj_points_allowed

    team_df['Adjusted Point Diff'] = team_df.apply(lambda r: r['Adjusted Points'] - r['Adjusted Points Allowed'],
                                                   axis=1)

    team_df['Bayes Adjusted Points'] = team_df['Adjusted Points']
    team_df['Bayes Adjusted Points Allowed'] = team_df['Adjusted Points Allowed']
    team_df['Bayes Adjusted Point Diff'] = team_df['Adjusted Point Diff']
    i = 0


def fit_bt():
    global graph
    global team_df

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    bt_df = helper.get_bradley_terry_from_graph()

    bts = {index: row['BT'] for index, row in bt_df.iterrows()}
    bt_vars = {index: row['Var'] for index, row in bt_df.iterrows()}

    for team_name in team_df.index:
        team_df.at[team_name, 'BT'] = bts.get(team_name)
        team_df.at[team_name, 'BT Var'] = bt_vars.get(team_name)

        set_bayes_bt(team_name)

    bayes_bts = {index: row['Bayes BT'] for index, row in team_df.iterrows()}
    bt_var = statistics.variance(bayes_bts.values())
    bt_sd = math.sqrt(bt_var)
    bt_norm = norm(0, bt_sd)
    for team_name in team_df.index:
        team_df.at[team_name, 'BT Pct'] = bt_norm.cdf(bayes_bts.get(team_name, 0))

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    tiers = helper.get_tiers()
    for team, tier in tiers.items():
        team_df.at[team, 'Tier'] = tier
    team_df = team_df.fillna(0)


def fit_drives(verbose=False):
    Y = individual_df['Drives']
    X = pd.get_dummies(individual_df[['Team', 'Opponent']])
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()

    if verbose:
        if results.f_pvalue > .05:
            print('OLS model is not a good fit')
        print(results.summary())

    off_coefs = {name.replace('Team_', '').replace('[T.', '').replace(']', ''): coef
                 for name, coef in results.params.items() if name.startswith('Team')}
    def_coefs = {name.replace('Opponent_', '').replace('[T.', '').replace(']', ''): coef
                 for name, coef in results.params.items() if name.startswith('Opponent')}
    intercept = results.params['const']

    for team, row in team_df.iterrows():
        team_df.at[team, 'Drives Model Good'] = results.f_pvalue < .05
        team_df.at[team, 'Drives Intercept'] = intercept
        team_df.at[team, 'Off Drives Coef'] = off_coefs.get(team, 0)
        team_df.at[team, 'Def Drives Coef'] = def_coefs.get(team, 0)


def predict_game(model, pt, away_name, home_name, vegas_line):
    bets = Bettor(team_df, individual_df, graph, gen_poisson_model)

    prediction_dict = dict()
    prediction_dict['Home Name'] = home_name
    prediction_dict['Away Name'] = away_name

    home_bt = team_df.at[home_name, 'Bayes BT']
    away_bt = team_df.at[away_name, 'Bayes BT']

    bt_chance = math.exp(home_bt) / (math.exp(home_bt) + math.exp(away_bt))
    prediction_dict['BT Chance'] = bt_chance

    home_avg_points = team_df.at[home_name, 'Bayes Avg Points']
    away_avg_points = team_df.at[away_name, 'Bayes Avg Points']
    avg_points_margin = home_avg_points - away_avg_points

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    home_expected_points, away_expected_points = helper.predict_score_from_gen_poisson_model(home_name, away_name)

    prediction_dict['Home Expected Points'] = home_expected_points
    prediction_dict['Away Expected Points'] = away_expected_points

    poisson_win_chance, poisson_tie_chance, poisson_loss_chance = bets.get_spread_chance(home_name, away_name, 0.0)
    poisson_win_chance = poisson_win_chance / (1 - poisson_tie_chance)
    poisson_win_chance = .5 if pd.isna(poisson_win_chance) else poisson_win_chance

    expected_points_diff = home_expected_points - away_expected_points

    home_bayes_avg_wins = team_df.at[home_name, 'Bayes Win Pct']
    away_bayes_avg_wins = team_df.at[away_name, 'Bayes Win Pct']
    prediction_dict['Home Bayes Wins'] = home_bayes_avg_wins
    prediction_dict['Away Bayes Wins'] = away_bayes_avg_wins
    win_pct_margin = home_bayes_avg_wins - away_bayes_avg_wins

    prediction_features = [vegas_line,
                           bt_chance,
                           poisson_win_chance,
                           avg_points_margin,
                           win_pct_margin,
                           expected_points_diff]

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

    le = LeagueEvaluator(team_df, individual_df, graph, gen_poisson_model)
    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    bets = Bettor(team_df, individual_df, graph, gen_poisson_model)
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

        favored_team = home_name if line <= 0 else away_name
        underdog = home_name if line > 0 else away_name
        location = 'at home' if line <= 0 else 'on the road'

        rounded_line = round(line * 2.0) / 2.0

        pp = PlayoffPredictor(team_df, graph)

        if verbose:
            print('The', favored_team.ljust(justify_width), 'are favored by',
                  round(abs(rounded_line), 1), 'points', location)

            home_expected_points = pred.get('Home Expected Points')
            away_expected_points = pred.get('Away Expected Points')
            pois_winner = home_name if home_expected_points >= away_expected_points else away_name
            pois_loser = home_name if home_expected_points < away_expected_points else away_name

            poisson_win, poisson_tie, poisson_loss = bets.get_spread_chance(pois_winner, pois_loser, 0.0)
            poisson_win = poisson_win / (1 - poisson_tie)

            print('The', pois_winner.ljust(justify_width), 'have a',
                  f'{poisson_win * 100:.3f}' + '% chance to beat the',
                  pois_loser.ljust(justify_width + 3), 'according to the Poisson Regression')

            winner_points = home_expected_points if pois_winner == home_name else away_expected_points
            loser_points = away_expected_points if pois_winner == home_name else home_expected_points
            pts = 'points' if round(winner_points - loser_points, 1) != 1.0 else 'point'
            common_winner_points = helper.get_common_score(winner_points)
            common_loser_points = helper.get_common_score(loser_points)
            is_ot = common_winner_points == common_loser_points
            if common_winner_points == common_loser_points:
                common_winner_points = common_winner_points + 3
            end = ' OT)' if is_ot else ')'
            expected_score = '(Projected Score: ' + str(common_winner_points) + ' - ' + str(common_loser_points) + end

            print('The', pois_winner.ljust(justify_width), 'are expected to win by an average of',
                  str(round(winner_points - loser_points, 1)).ljust(4), pts.ljust(7), expected_score)

            home_bt = pred.get('BT Chance')
            favored_bt = home_name if home_bt >= .5 else away_name
            underdog_bt = away_name if home_bt >= .5 else home_name
            bt = home_bt if home_bt >= .5 else 1 - home_bt
            print('The', favored_bt.ljust(justify_width), 'have a',
                  f'{bt * 100:.3f}' + '% chance to beat the', underdog_bt.ljust(justify_width + 3),
                  'according to the BT Model')

            home_proj_wins = pp.get_proj_record(home_name)[0]
            away_proj_wins = pp.get_proj_record(away_name)[0]
            higher_name = home_name if home_proj_wins >= away_proj_wins else away_name
            lower_name = away_name if home_proj_wins >= away_proj_wins else home_name
            higher_proj = home_proj_wins if home_proj_wins >= away_proj_wins else away_proj_wins
            lower_proj = away_proj_wins if home_proj_wins >= away_proj_wins else home_proj_wins

            print('The', higher_name.ljust(justify_width), 'are on pace to be a',
                  str(higher_proj).ljust(2), 'win team, the', lower_name.ljust(justify_width),
                  'are on pace to be a', str(lower_proj), 'win team')

        le.plot_matchup(favored_team, underdog, prob, rounded_line if location == 'at home' else -rounded_line)
        print()

    print()


def print_table(week, sort_key='Bayes BT', sort_by_division=False):
    global team_df

    if sort_key in ['Avg Points Allowed', 'Points Allowed Coef', 'Adjusted Points Allowed']:
        ascending_order = True
    else:
        ascending_order = False
    team_df = team_df.sort_values(by=sort_key, kind='mergesort', ascending=ascending_order)
    if sort_by_division:
        team_df = team_df.sort_values(by='Division', kind='mergesort', ascending=False)

    index_col = 'Division' if sort_by_division else 'Rank'

    if week >= 18:
        columns = [index_col, 'Name', 'Record', 'Bayes Win %', 'Score',
                   'Adj. PPG', 'Adj. PPG Allowed', 'Adj. Point Diff']
    else:
        columns = [index_col, 'Name', 'Record', 'Bayes Win %', 'Score',
                   'Proj. Record', 'Adj. PPG', 'Adj. PPG Allowed', 'Adj. Point Diff']
        if week >= 10:
            columns.append('Win Division')
            columns.append('Make Wild Card')
            columns.append('Make Playoffs')
            columns.append('First Round Bye')
            columns.append('Win Superbowl')

    table = PrettyTable(columns)
    table.float_format = '0.3'

    points_coefs = team_df['Bayes Adjusted Points']
    points_allowed_coefs = team_df['Bayes Adjusted Points Allowed']

    points_avg = statistics.mean(points_coefs)
    points_allowed_avg = statistics.mean(points_allowed_coefs)
    points_diff_avg = statistics.mean(team_df['Bayes Adjusted Point Diff'])

    points_var = statistics.variance(points_coefs)
    points_allowed_var = statistics.variance(points_allowed_coefs)
    points_diff_var = statistics.variance(team_df['Bayes Adjusted Point Diff'])

    stop = '\033[0m'
    pp = PlayoffPredictor(team_df, graph)
    helper = Helper(team_df, individual_df, graph, gen_poisson_model)

    for index, row in team_df.iterrows():
        table_row = list()

        wins = row['Wins']
        losses = row['Losses']
        ties = row['Ties']
        record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]]) + ' - ' + str(int(ties))

        rank = team_df.index.get_loc(index) + 1

        points_pct = .1
        points_color = helper.get_color(row['Bayes Adjusted Points'], points_avg, points_var, alpha=points_pct)
        points_allowed_color = helper.get_color(row['Bayes Adjusted Points Allowed'], points_allowed_avg, points_allowed_var, alpha=points_pct,
                                                invert=True)
        points_diff_color = helper.get_color(row['Bayes Adjusted Point Diff'], points_diff_avg, points_diff_var, alpha=points_pct,
                                             invert=False)

        if sort_by_division:
            table_row.append(row['Division'])
        else:
            table_row.append(rank)
        table_row.append(index)
        table_row.append(record)
        table_row.append((f"{row['Bayes Win Pct'] * 100:.1f}" + '%').rjust(5))

        bt_color = helper.get_color(row['BT'], 0, row['BT Var'])
        table_row.append(bt_color + f"{row['BT Pct'] * 100:.1f}".rjust(5) + stop)

        if week < 18:
            proj_record = pp.get_proj_record(index)
            ties = proj_record[-1]
            proj_record = ' - '.join([str(val).rjust(2) for val in proj_record[:-1]]) + ' - ' + str(int(ties))
            table_row.append(proj_record)

        table_row.append(points_color + str(round(row['Bayes Adjusted Points'], 1)) + stop)
        table_row.append(points_allowed_color + str(round(row['Bayes Adjusted Points Allowed'], 1)) + stop)
        table_row.append(points_diff_color + str(round(row['Bayes Adjusted Point Diff'], 1)).rjust(5) + stop)

        if 10 <= week < 18:
            table_row.append((f'{row["Win Division"] * 100:.1f}' + '%').rjust(6))
            table_row.append((f'{row["Make Wild Card"] * 100:.1f}' + '%').rjust(6))
            table_row.append((f'{row["Make Playoffs"] * 100:.1f}' + '%').rjust(6))
            table_row.append((f'{row["First Round Bye"] * 100:.1f}' + '%').rjust(6))
            table_row.append((f'{row["Win Superbowl"] * 100:.1f}' + '%').rjust(5))

        table.add_row(table_row)

    # Print the table
    print('Rankings')
    print(table)
    print()


def set_bayes_bt(team):
    global team_df

    games_played = team_df.at[team, 'Wins'] + team_df.at[team, 'Losses'] + team_df.at[team, 'Ties']

    bt_var = team_df.at[team, 'BT Var']
    bt_var = .15 if pd.isna(bt_var) else bt_var

    sample_avg = team_df.at[team, 'BT']
    sample_avg = team_df.at[team, 'Preseason BT'] if pd.isna(sample_avg) else sample_avg

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    bayes_bt = helper.get_bayes_avg(team_df.at[team, 'Preseason BT'],
                                    .275,
                                    sample_avg,
                                    bt_var,
                                    games_played)

    team_df.at[team, 'Bayes BT'] = bayes_bt


def season(week_num,
           manual_odds=False,
           include_parity=True,
           verbose=True):
    global graph
    global team_df
    global game_df

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    le = LeagueEvaluator(team_df, individual_df, graph, gen_poisson_model)

    model = helper.load_model()
    pt = helper.load_pt()
    schedule = league_structure.load_schedule()

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
    team_df['Division'] = team_df.apply(lambda r: league_structure.get_division(r['Team']), axis=1)
    team_df['Primary Colors'] = primary_colors
    team_df['Secondary Colors'] = second_colors
    team_df = team_df.set_index('Team')

    team_df['BT Var'] = team_df['BT Var'].fillna(.15)
    team_df = team_df.fillna(0)

    preseason_bts = le.get_preseason_bts()
    for team, pre_bt in preseason_bts.items():
        team_df.at[team, 'Preseason BT'] = pre_bt
        if week_num <= 1:
            team_df.at[team, 'Bayes Win Pct'] = .5
        set_bayes_bt(team)

    bayes_bts = {index: row['Bayes BT'] for index, row in team_df.iterrows()}
    bt_var = statistics.variance(bayes_bts.values())
    bt_sd = math.sqrt(bt_var)
    bt_norm = norm(0, bt_sd)
    for team_name in team_df.index:
        team_df.at[team_name, 'BT Pct'] = bt_norm.cdf(bayes_bts.get(team_name, 0))

    if week_num <= 1:
        print('Preseason')
        print_table(week_num)

    if manual_odds:
        odds = []
    else:
        odds = Odds.get_odds()

    all_week_results = league_structure.get_games_before_week(week_num, use_persisted=True)

    for week in range(week_num):
        get_game_results(week + 1, all_week_results,
                         current_week=week == week_num - 1,)

        if week == week_num - 1:
            print('Week', week + 1)
            games = list()
            for game in schedule.get('weeks')[week]:
                home_name = game.get('home')
                away_name = game.get('away')
                if manual_odds:
                    odds = game.get('line', 0)
                    games.append((away_name, home_name, odds))
                else:
                    bets = Bettor(team_df, individual_df, graph, gen_poisson_model)
                    games.append((away_name, home_name, bets.get_vegas_line(home_name, away_name, odds)))

            order_predictions(model, pt, games, verbose=verbose)

            afc_playoff_teams = list()
            nfc_playoff_teams = list()
            playoff_chances = pd.DataFrame()
            if week >= 10:
                pp = PlayoffPredictor(team_df, graph)
                for team in team_df.index:
                    team_df.at[team, 'Win Division'] = pp.get_division_winner_chance(team)
                    team_df.at[team, 'Make Wild Card'] = pp.get_wildcard_chance(team)
                    team_df.at[team, 'Make Playoffs'] = team_df.at[team, 'Win Division'] + \
                                                        team_df.at[team, 'Make Wild Card']
                    team_df.at[team, 'First Round Bye'] = pp.get_first_round_bye_chance(team)

                afc_playoff_teams = pp.get_conf_playoff_seeding(is_afc=True)
                nfc_playoff_teams = pp.get_conf_playoff_seeding(is_afc=False)
                playoff_chances = pp.get_win_super_bowl_chances(afc_playoff_teams, nfc_playoff_teams)
                for team in team_df.index:
                    if team in playoff_chances.index:
                        team_df.at[team, 'Win Superbowl'] = playoff_chances.at[team, 'Win Superbowl Chance']
                    else:
                        team_df.at[team, 'Win Superbowl'] = 0.0

            print_table(week, sort_by_division=False)

            if week >= 10:
                pc.print_full_playoff_chances(afc_playoff_teams, nfc_playoff_teams, playoff_chances)

    le = LeagueEvaluator(team_df, individual_df, graph, gen_poisson_model)
    # le.show_off_def()
    le.show_off_def_interactive()
    le.show_graph(divisional_edges_only=week_num > 5)

    if 5 < week_num <= 18:
        le.surprises()
        le.get_schedule_difficulties()

    if not manual_odds and week_num > 5:
        bets = Bettor(team_df, individual_df, graph, gen_poisson_model)
        bets.all_bets(100)

    if include_parity:
        le.parity_clock()
