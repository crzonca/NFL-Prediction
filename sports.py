import itertools
import json
import math
import statistics
import warnings
from datetime import datetime

import PIL
import choix
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from prettytable import PrettyTable
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, poisson, chi2, skellam
from sklearn.linear_model import PoissonRegressor

import Projects.nfl.NFL_Prediction.OddsHelper as Odds

graph = nx.MultiDiGraph()
team_df = pd.DataFrame()
game_df = pd.DataFrame()
individual_df = pd.DataFrame()


def get_path(use_nba=True):
    league = 'NBA' if use_nba else 'NHL'
    schedule_path = 'Projects/play/sports/' + league + '23Schedule.json'
    return schedule_path


def load_schedule(schedule_path):
    with open(schedule_path, 'r') as f:
        schedule = json.load(f)
        return schedule


def get_bayes_avg(prior_avg, prior_var, sample_avg, sample_var, n):
    k_0 = sample_var / prior_var
    posterior_avg = ((k_0 / (k_0 + n)) * prior_avg) + ((n / (k_0 + n)) * sample_avg)
    return posterior_avg


def get_bayes_avg_wins(team_name):
    matching_games = game_df.loc[game_df['Team'] == team_name]

    prior_avg = 0.5
    prior_var = 0.01

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


def get_game_results(results_df, alpha):
    for index, row in results_df.iterrows():
        home = row['Home']
        away = row['Away']
        home_points = row['Home Points']
        away_points = row['Away Points']
        home_otl = row['Home OTL']
        away_otl = row['Away OTL']

        set_game_outcome(home, away, home_points, away_points, home_otl, away_otl)

    fit_poisson(alpha)
    fit_bt()


def set_game_outcome(home_name, away_name, home_points, away_points, home_otl, away_otl):
    global graph
    global team_df
    global game_df
    global individual_df

    home_victory = home_points > away_points
    away_victory = away_points > home_points

    game_df.loc[len(game_df.index)] = [home_name, 1 if home_victory else 0, home_points, away_points]
    game_df.loc[len(game_df.index)] = [away_name, 1 if away_victory else 0, away_points, home_points]

    home_points = home_points - 1 if away_otl else home_points
    away_points = away_points - 1 if home_otl else away_points

    individual_df.loc[len(individual_df.index)] = [home_name, away_name, home_points, 1]
    individual_df.loc[len(individual_df.index)] = [away_name, home_name, away_points, 0]

    winner = home_name if home_victory else away_name
    loser = away_name if home_victory else home_name
    graph.add_edge(loser, winner)

    try:
        home_games_played = team_df.at[home_name, 'Games Played']
        away_games_played = team_df.at[away_name, 'Games Played']
    except KeyError as e:
        print(home_name)
        print(away_name)
        i = 0
        return

    team_df.at[home_name, 'Games Played'] = home_games_played + 1
    team_df.at[away_name, 'Games Played'] = away_games_played + 1

    team_df.at[home_name, 'Wins'] = team_df.at[home_name, 'Wins'] + 1 if home_victory else team_df.at[home_name, 'Wins']
    team_df.at[away_name, 'Wins'] = team_df.at[away_name, 'Wins'] + 1 if away_victory else team_df.at[away_name, 'Wins']

    team_df.at[home_name, 'Losses'] = team_df.at[home_name, 'Losses'] + 1 \
        if away_victory else team_df.at[home_name, 'Losses']
    team_df.at[away_name, 'Losses'] = team_df.at[away_name, 'Losses'] + 1 \
        if home_victory else team_df.at[away_name, 'Losses']

    team_df.at[home_name, 'OTL'] = team_df.at[home_name, 'OTL'] + 1 if home_otl else team_df.at[home_name, 'OTL']
    team_df.at[away_name, 'OTL'] = team_df.at[away_name, 'OTL'] + 1 if away_otl else team_df.at[away_name, 'OTL']

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

    team_df = team_df.fillna(0)


def fit_neg_bin():
    global individual_df
    points_df = individual_df[['Team', 'Opponent', 'Points', 'Is_Home']]
    points_df = points_df.rename(columns={'Team': 'Offense',
                                          'Opponent': 'Defense'})

    poisson_model = smf.glm(formula="Points ~ Offense + Defense + Is_Home",
                            data=points_df,
                            family=sm.families.Poisson()).fit(disp=1)
    print('Poisson Model:')
    print(poisson_model.summary())
    print()

    pearson_chi2 = chi2(df=poisson_model.df_resid)
    alpha = .05
    print('Critical Value for alpha=.05:', pearson_chi2.ppf(1 - alpha))
    print('Test Statistic:              ', poisson_model.pearson_chi2)
    p_value = pearson_chi2.sf(poisson_model.pearson_chi2)
    print('P-Value:                     ', p_value)
    if p_value < alpha:
        print('The Poisson Model is not a good fit for the data')
    else:
        print('The Poisson Model is a good fit for the data')
    print()

    points_df['event_rate'] = poisson_model.mu
    points_df['auxiliary_reg'] = points_df.apply(lambda x: ((x['Points'] - x['event_rate']) ** 2 - x['event_rate']) / x['event_rate'], axis=1)

    aux_olsr_results = smf.ols("auxiliary_reg ~ event_rate - 1", data=points_df).fit()
    print(aux_olsr_results.summary())
    print()

    neg_bin_model = smf.glm(formula="Points ~ Offense + Defense + Is_Home",
                            data=points_df,
                            family=sm.genmod.families.family.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()

    print('Negative Binomial Model:')
    print(neg_bin_model.summary())
    print()

    pearson_chi2 = chi2(df=neg_bin_model.df_resid)
    print('Critical Value for alpha=.05:', pearson_chi2.ppf(1 - alpha))
    print('Test Statistic:              ', neg_bin_model.pearson_chi2)
    p_value = pearson_chi2.sf(neg_bin_model.pearson_chi2)
    print('P-Value:                     ', p_value)
    if p_value < alpha:
        print('The Negative Binomial Model is not a good fit for the data')
    else:
        print('The Negative Binomial Model is a good fit for the data')
    print()


def fit_poisson(alpha=.1):
    global team_df
    global individual_df

    points_regression = PoissonRegressor(alpha=alpha, fit_intercept=True)
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
            team_df.at[team_name, 'Adjusted Points'] = math.exp(
                points_reg_results.at['Team_' + team_name, 'points_reg_value'])
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

        team_df['Adjusted Point Diff'] = team_df['Adjusted Points'] - team_df['Adjusted Points Allowed']

    team_df = team_df.fillna(0)


def fit_bt():
    global graph
    global team_df

    bt_df = get_bradley_terry_from_graph(graph)

    bts = {index: row['BT'] for index, row in bt_df.iterrows()}
    bt_vars = {index: row['Var'] for index, row in bt_df.iterrows()}

    for team_name in team_df.index:
        team_df.at[team_name, 'BT'] = bts.get(team_name, team_df.at[team_name, 'BT'])
        team_df.at[team_name, 'BT Var'] = bt_vars.get(team_name, team_df.at[team_name, 'BT Var'])

    team_df = team_df.fillna(0)


def get_remaining_win_probs(team_name, schedule_path, total_games=82):
    opponents = load_schedule(schedule_path=schedule_path).get(team_name)

    bt = team_df.at[team_name, 'BT']
    wins = team_df.at[team_name, 'Wins']
    losses = team_df.at[team_name, 'Losses']
    games_played = wins + losses
    if games_played == total_games:
        return []

    remaining_opponents = opponents[games_played:total_games]
    opponent_bts = [team_df.at[opponent, 'BT'] for opponent in remaining_opponents]
    win_probs = [math.exp(bt) / (math.exp(bt) + math.exp(opp_bt)) for opp_bt in opponent_bts]
    return win_probs


def get_proj_record(team_name, schedule_path, total_games=82):
    if team_name == 'Lakers' or team_name == 'Pacers':
        total_games = total_games + 1
    win_probs = get_remaining_win_probs(team_name, schedule_path, total_games=total_games)
    wins = team_df.at[team_name, 'Wins']

    expected_wins = sum(win_probs) + wins
    expected_losses = total_games - expected_wins

    if team_name == 'Lakers':
        expected_wins = expected_wins - 1
    if team_name == 'Pacers':
        expected_losses = expected_losses - 1

    return round(expected_wins), round(expected_losses)


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


def print_table(schedule_path, use_nba, total_games=82, sort_key='BT'):
    global team_df

    if sort_key in ['Avg Points Allowed', 'Points Allowed Coef']:
        ascending_order = True
    else:
        ascending_order = False
    team_df = team_df.sort_values(by=sort_key, kind='mergesort', ascending=ascending_order)

    nba_columns = ['Rank', 'Name', 'Record', 'Bayes Win Pct', 'Score',
                   'Proj. Record', 'Adj. PPG', 'Adj. PPG Allowed', 'Adj. Point Diff']

    nhl_columns = ['Rank', 'Name', 'Record', 'Points', 'Bayes Win Pct', 'Score',
                   'Proj. Record', 'Proj. Points', 'Adj. PPG', 'Adj. PPG Allowed', 'Adj. Point Diff']

    columns = nba_columns if use_nba else nhl_columns

    table = PrettyTable(columns)
    table.float_format = '0.3'

    points_intercept = team_df['Points Intercept'][0]
    points_coefs = (points_intercept + team_df['Points Coef']).apply(np.exp)
    points_allowed_coefs = (points_intercept + team_df['Points Allowed Coef']).apply(np.exp)
    points_diff_coefs = team_df['Adjusted Point Diff']

    points_var = statistics.variance(team_df['Points Coef'])
    points_allowed_var = statistics.variance(team_df['Points Allowed Coef'])
    points_diff_var = statistics.variance(points_coefs - points_allowed_coefs)

    bt_var = statistics.variance(team_df['BT'])
    bt_sd = math.sqrt(bt_var)
    bt_norm = norm(0, bt_sd)

    stop = '\033[0m'

    for index, row in team_df.iterrows():
        table_row = list()

        wins = row['Wins']
        losses = row['Losses']
        otl = row['OTL']

        if index == 'Lakers':
            wins = wins - 1
        if index == 'Pacers':
            losses = losses - 1

        nba_record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]])
        nhl_record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses - otl, otl]])
        record = nba_record if use_nba else nhl_record

        rank = team_df.index.get_loc(index) + 1

        points_pct = .1
        points_color = get_color(row['Points Coef'], points_var, alpha=points_pct)
        points_allowed_color = get_color(row['Points Allowed Coef'], points_allowed_var, alpha=points_pct, invert=True)
        points_diff_color = get_color(row['Adjusted Point Diff'], points_diff_var, alpha=points_pct)

        table_row.append(rank)
        table_row.append(index)
        table_row.append(record)
        if not use_nba:
            table_row.append(int(wins * 2 + otl))

        table_row.append((f"{row['Bayes Win Pct'] * 100:.1f}" + '%').rjust(5))

        bt_color = get_color(row['BT'], row['BT Var'])
        table_row.append(bt_color + f"{bt_norm.cdf(row['BT']) * 100:.1f}".rjust(5) + stop)

        proj_record = get_proj_record(index, schedule_path, total_games=total_games)

        if not use_nba:
            proj_wins, proj_total_losses = proj_record
            proj_otl = otl + round(proj_total_losses * .1)
            proj_losses = proj_total_losses - proj_otl
            proj_record = [proj_wins, int(proj_losses), int(proj_otl)]
            proj_points = int(proj_wins * 2 + proj_otl)
        proj_record = ' - '.join([str(val).rjust(2) for val in proj_record])
        table_row.append(proj_record)

        if not use_nba:
            table_row.append(proj_points)

        table_row.append(points_color + str(round(row['Adjusted Points'], 1)) + stop)
        table_row.append(points_allowed_color + str(round(row['Adjusted Points Allowed'], 1)) + stop)
        jw = 5 if use_nba else 4
        table_row.append(points_diff_color + str(round(row['Adjusted Point Diff'], 1)).rjust(jw) + stop)

        table.add_row(table_row)

    # Print the table
    print('Rankings')
    print(table)
    print()


def get_schedule_difficulties(schedule_path, total_games=82):
    global team_df

    schedule = load_schedule(schedule_path)

    team_schedule = dict()
    for team in team_df.index:
        opponents = schedule.get(team)

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

        record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]])

        average_wins = team_average_wins.get(team)
        avg_record = ' - '.join([str(val).rjust(2) for val in [average_wins, total_games - average_wins]])

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


def show_off_def(use_nba):
    warnings.filterwarnings("ignore")

    sns.set(style="ticks")

    # Format and title the graph
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_title('')
    ax.set_xlabel('Adjusted Points For')
    ax.set_ylabel('Adjusted Points Against')
    ax.set_facecolor('#FAFAFA')

    base_dir = 'nba_logos' if use_nba else 'nhl_logos'

    images = {team: PIL.Image.open('Projects/play/sports/' + base_dir + '/' + team + '.png')
              for team, row in team_df.iterrows()}

    intercept = team_df['Points Intercept'].mean()

    margin = 3 if use_nba else .1
    min_x = math.exp(intercept + team_df['Points Coef'].min()) - margin
    max_x = math.exp(intercept + team_df['Points Coef'].max()) + margin

    min_y = math.exp(intercept + team_df['Points Allowed Coef'].min()) - margin
    max_y = math.exp(intercept + team_df['Points Allowed Coef'].max()) + margin

    ax = plt.gca()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)
    ax.set_aspect(aspect=0.3, adjustable='datalim')

    for team in team_df.index:
        xa = math.exp(intercept + team_df.at[team, 'Points Coef'])
        ya = math.exp(intercept + team_df.at[team, 'Points Allowed Coef'])

        offset = 1.25 if use_nba else .065
        offset = 1.25 if use_nba else .065
        ax.imshow(images.get(team), extent=(xa - offset, xa + offset, ya + offset, ya - offset), alpha=.8)

    plt.axvline(x=math.exp(intercept), color='r', linestyle='--', alpha=.5)
    plt.axhline(y=math.exp(intercept), color='r', linestyle='--', alpha=.5)

    average = math.exp(intercept)
    step = 5 if use_nba else .5
    offset_dist = step * math.sqrt(2)
    offsets = set(np.arange(0, 150, offset_dist))
    offsets = offsets.union({-offset for offset in offsets})

    for offset in [average + offset for offset in offsets]:
        plt.axline(xy1=(average, offset), slope=1, alpha=.1)

    # Show the graph
    plt.show()


def show_graph(use_nba):
    warnings.filterwarnings("ignore")

    sns.set(style="ticks")

    nfl = graph.copy()

    # Format and title the graph
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_aspect('auto')
    ax.set_title('')
    ax.set_facecolor('#FAFAFA')

    # Get the Pagerank of each node
    bts = {team: row['BT'] for team, row in team_df.iterrows()}
    max_bt = team_df['BT'].max()
    min_bt = team_df['BT'].min()
    bt_dev = statistics.stdev(team_df['BT'])
    subset = {team: np.digitize(row['BT'], np.arange(min_bt, max_bt, bt_dev / 2)) for team, row in team_df.iterrows()}

    nx.set_node_attributes(nfl, bts, 'BT')
    nx.set_node_attributes(nfl, subset, 'subset')

    base_dir = 'nba_logos' if use_nba else 'nhl_logos'
    images = {team: PIL.Image.open('Projects/play/sports/' + base_dir + '/' + team + '.png')
              for team, row in team_df.iterrows()}
    nx.set_node_attributes(nfl, images, 'image')

    pos = nx.multipartite_layout(nfl, align='horizontal')

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
                               alpha=0.01,
                               edge_color='black',
                               connectionstyle=conn_stlye,
                               arrowsize=10,
                               min_target_margin=target_margin)

    # Select the size of the image (relative to the X axis)
    icon_size = {team: (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025 * math.exp(bt) for team, bt in bts.items()}
    # icon_size['Bears'] = icon_size.get('Bears') * .8
    icon_center = {team: size / 2.0 for team, size in icon_size.items()}

    for n in nfl.nodes:
        xa, ya = fig.transFigure.inverted().transform(ax.transData.transform(pos[n]))
        a = plt.axes([xa - icon_center.get(n), ya - icon_center.get(n), icon_size.get(n), icon_size.get(n)])
        a.set_aspect('auto')
        a.imshow(nfl.nodes[n]['image'])
        a.axis("off")

    # Show the graph
    plt.show()


def get_results_df(use_nba):
    domain = 'https://api.natstat.com/v2/'
    api_key = '87ab-ba0b1b'

    sport = 'NBA' if use_nba else 'NHL'

    end_date = datetime.now().astimezone()
    end_date = end_date.strftime('%Y-%m-%d')

    def get_games(page=1):
        params = {'key': api_key,
                  'format': 'json',
                  'start': '2023-09-09',
                  'end': end_date,
                  'page': page,
                  'max': 1000}
        endpoint = 'games/' + sport + '/'
        return requests.get(domain + endpoint, params=params).json()

    all_games = dict()
    x = get_games(1)
    all_games.update(x.get('games'))
    total_pages = int(x.get('meta').get('Total_Pages'))

    for page in range(2, total_pages + 1):
        x = get_games(page)
        all_games.update(x.get('games'))

    all_games_df = pd.DataFrame.from_dict(all_games, orient='index')
    all_games_df = all_games_df.loc[all_games_df['GameStatus'] == 'Final']

    nba_name_map = {'Cleveland Cavaliers': 'Cavaliers',
                    'Cleveland': 'Cavaliers',
                    'Detroit Pistons': 'Pistons',
                    'Detroit': 'Pistons',
                    'Atlanta Hawks': 'Hawks',
                    'Atlanta': 'Hawks',
                    'New Orleans Pelicans': 'Pelicans',
                    'New Orleans': 'Pelicans',
                    'Charlotte Hornets': 'Hornets',
                    'Charlotte': 'Hornets',
                    'Washington Wizards': 'Wizards',
                    'Washington': 'Wizards',
                    'Denver Nuggets': 'Nuggets',
                    'Denver': 'Nuggets',
                    'Brooklyn Nets': 'Nets',
                    'Brooklyn': 'Nets',
                    'Minnesota Timberwolves': 'Timberwolves',
                    'Minnesota': 'Timberwolves',
                    'Oklahoma City Thunder': 'Thunder',
                    'Oklahoma City': 'Thunder',
                    'San Antonio Spurs': 'Spurs',
                    'San Antonio': 'Spurs',
                    'Houston Rockets': 'Rockets',
                    'Houston': 'Rockets',
                    'Dallas Mavericks': 'Mavericks',
                    'Dallas': 'Mavericks',
                    'Portland Trail Blazers': 'Trail Blazers',
                    'Portland': 'Trail Blazers',
                    'Chicago Bulls': 'Bulls',
                    'Chicago': 'Bulls',
                    'Utah Jazz': 'Jazz',
                    'Utah': 'Jazz',
                    'Orlando Magic': 'Magic',
                    'Orlando': 'Magic',
                    'New York Knicks': 'Knicks',
                    'New York': 'Knicks',
                    'Los Angeles Lakers': 'Lakers',
                    'L.A. Lakers': 'Lakers',
                    'Milwaukee Bucks': 'Bucks',
                    'Milwaukee': 'Bucks',
                    'Sacramento Kings': 'Kings',
                    'Sacramento': 'Kings',
                    'Philadelphia 76ers': '76ers',
                    'Philadelphia': '76ers',
                    'Toronto Raptors': 'Raptors',
                    'Toronto': 'Raptors',
                    'Memphis Grizzlies': 'Grizzlies',
                    'Memphis': 'Grizzlies',
                    'Miami Heat': 'Heat',
                    'Miami': 'Heat',
                    'Los Angeles Clippers': 'Clippers',
                    'L.A. Clippers': 'Clippers',
                    'Phoenix Suns': 'Suns',
                    'Phoenix': 'Suns',
                    'Golden State Warriors': 'Warriors',
                    'Golden State': 'Warriors',
                    'Indiana Pacers': 'Pacers',
                    'Indiana': 'Pacers',
                    'Boston Celtics': 'Celtics',
                    'Boston': 'Celtics'}

    nhl_name_map = {'Pittsburgh Penguins': 'Penguins',
                    'Arizona Coyotes': 'Coyotes',
                    'Ottawa Senators': 'Senators',
                    'San Jose Sharks': 'Sharks',
                    'Seattle Kraken': 'Kraken',
                    'Vegas Golden Knights': 'Golden Knights',
                    'Nashville Predators': 'Predators',
                    'Detroit Red Wings': 'Red Wings',
                    'Anaheim Ducks': 'Ducks',
                    'Florida Panthers': 'Panthers',
                    'Edmonton Oilers': 'Oilers',
                    'Vancouver Canucks': 'Canucks',
                    'Columbus Blue Jackets': 'Blue Jackets',
                    'Philadelphia Flyers': 'Flyers',
                    'Montreal Canadiens': 'Canadiens',
                    'Winnipeg Jets': 'Jets',
                    'Colorado Avalanche': 'Avalanche',
                    'Toronto Maple Leafs': 'Maple Leafs',
                    'New York Islanders': 'Islanders',
                    'Minnesota Wild': 'Wild',
                    'Chicago Blackhawks': 'Blackhawks',
                    'Los Angeles Kings': 'Kings',
                    'Washington Capitals': 'Capitals',
                    'Calgary Flames': 'Flames',
                    'Tampa Bay Lightning': 'Lightning',
                    'New Jersey Devils': 'Devils',
                    'Buffalo Sabres': 'Sabres',
                    'Boston Bruins': 'Bruins',
                    'Dallas Stars': 'Stars',
                    'Carolina Hurricanes': 'Hurricanes',
                    'New York Rangers': 'Rangers',
                    'St. Louis Blues': 'Blues'}

    name_map = nba_name_map if use_nba else nhl_name_map

    all_games_df['Home'] = all_games_df['Home'].map(name_map)
    all_games_df['Away'] = all_games_df['Visitor'].map(name_map)

    def get_num_ot(row):
        if 'OT' not in row['Overtime']:
            return 0
        if 'OT' == row['Overtime']:
            return 1
        ot_str = row['Overtime'].replace('OT', '')
        return int(ot_str)

    def get_otl(row, home=True):
        if 'OT' in row['Overtime'] or 'SO' in row['Overtime']:
            if home:
                if row['Home Points'] < row['Away Points']:
                    return True
            else:
                if row['Home Points'] > row['Away Points']:
                    return True
        return False

    if use_nba:
        all_games_df['NumOT'] = all_games_df.apply(lambda r: get_num_ot(r), axis=1)
        all_games_df['Home Points'] = all_games_df.apply(
            lambda r: round(int(r['ScoreHome']) * 48 / (48 + r['NumOT'] * 5)), axis=1)
        all_games_df['Away Points'] = all_games_df.apply(
            lambda r: round(int(r['ScoreVis']) * 48 / (48 + r['NumOT'] * 5)), axis=1)
        all_games_df['Home OTL'] = False
        all_games_df['Away OTL'] = False
    else:
        all_games_df['Home Points'] = all_games_df['ScoreHome'].astype(int)
        all_games_df['Away Points'] = all_games_df['ScoreVis'].astype(int)
        all_games_df['Home OTL'] = all_games_df.apply(lambda r: get_otl(r, True), axis=1)
        all_games_df['Away OTL'] = all_games_df.apply(lambda r: get_otl(r, False), axis=1)

    for index, row in all_games_df.iterrows():
        if row['Home Points'] == row['Away Points']:
            if row['ScoreHome'] > row['ScoreVis']:
                all_games_df.at[index, 'Home Points'] = all_games_df.at[index, 'Home Points'] + 1
            else:
                all_games_df.at[index, 'Away Points'] = all_games_df.at[index, 'Away Points'] + 1

    all_games_df = all_games_df[['Home', 'Away', 'Home Points', 'Away Points', 'Home OTL', 'Away OTL']]

    return all_games_df


def season(use_nba,
           pot=20.0,
           total_games=82,
           preseason_path=None,
           preseason=False,
           include_schedule_difficulty=False,
           alpha=0.1):
    global graph
    global team_df
    global game_df
    global individual_df

    graph.clear()

    schedule_path = get_path(use_nba=use_nba)

    team_df = pd.DataFrame(columns=['Team', 'Games Played',
                                    'Wins', 'Losses', 'OTL',
                                    'BT', 'BT Var',
                                    'Win Pct', 'Bayes Win Pct',
                                    'Avg Points', 'Avg Points Allowed',
                                    'Points Intercept', 'Points Coef', 'Points Allowed Coef',
                                    'Adjusted Points', 'Adjusted Points Allowed', 'Adjusted Point Diff'])

    game_df = pd.DataFrame(columns=['Team', 'Win', 'Points', 'Points Allowed'])
    individual_df = pd.DataFrame(columns=['Team', 'Opponent', 'Points', 'Is_Home'])

    schedule = load_schedule(schedule_path)

    teams = pd.Series(schedule.keys())
    team_df['Team'] = teams
    team_df = team_df.set_index('Team')

    team_df = team_df.fillna(0)

    if preseason and preseason_path is not None:
        preseason_bts = get_preseason_bts(schedule_path, preseason_path)
        for team, bt in preseason_bts.items():
            team_df.at[team, 'BT'] = bt
            team_df.at[team, 'BT Var'] = .15
            team_df.at[team, 'Bayes Win Pct'] = .5
        print('Preseason')
        print_table(schedule_path, use_nba, total_games=total_games)

    results_df = get_results_df(use_nba)
    get_game_results(results_df, alpha=alpha)
    print_table(schedule_path, use_nba, total_games=total_games)

    show_off_def(use_nba)
    show_graph(use_nba)
    # fit_neg_bin()

    if use_nba:
        all_bets(pot)

        # allocation = pot / 2
        # ats_bets(allocation)
        # straight_up_bets(allocation)

    if include_schedule_difficulty:
        get_schedule_difficulties(schedule_path, total_games=total_games)


def get_preseason_bts(schedule_path, preseason_path, use_mse=True):
    preseason_df = pd.read_csv(preseason_path)
    win_totals = {row['Team']: row['Projected Wins'] for index, row in preseason_df.iterrows()}

    schedules = load_schedule(schedule_path)

    teams = list(win_totals.keys())
    bts = np.zeros(len(teams))
    win_proj = np.array(list(win_totals.values()))

    def objective(params):
        val = np.float64(0)

        for team, opps in schedules.items():
            team_proj = np.float64(0)
            team_index = teams.index(team)
            for opponent in opps:
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

    df['Rank'] = range(1, len(win_totals.keys()) + 1)
    df = df.set_index('Rank', drop=True)
    df = df[['Team', 'BT', 'Odds Projection', 'BT Projection']]
    df['BT Projection'] = df['BT Projection'].round(1)

    return team_bts


def predict_score(team1, team2):
    intercept = team_df.at[team1, 'Points Intercept']

    team1_off_coef = team_df.at[team1, 'Points Coef']
    team2_off_coef = team_df.at[team2, 'Points Coef']
    team1_def_coef = team_df.at[team1, 'Points Allowed Coef']
    team2_def_coef = team_df.at[team2, 'Points Allowed Coef']

    team1_score = math.exp(intercept + team1_off_coef + team2_def_coef)
    team2_score = math.exp(intercept + team2_off_coef + team1_def_coef)

    winner = team1 if team1_score >= team2_score else team2
    winner_score = team1_score if team1_score >= team2_score else team2_score
    loser = team1 if team1_score < team2_score else team2
    loser_score = team1_score if team1_score < team2_score else team2_score

    return winner, loser, winner_score, loser_score


def predict_scores(matchups):
    teams = set([matchup[0] for matchup in matchups]).union(set([matchup[1] for matchup in matchups]))
    justify_width = 0

    predictions = list()
    for team1, team2 in matchups:
        team1 = team1.split()[-1]
        team2 = team2.split()[-1]

        if team1 == 'Blazers':
            team1 = 'Trail Blazers'

        if team2 == 'Blazers':
            team2 = 'Trail Blazers'

        max_len = max([len(team1), len(team2)])
        if max_len > justify_width:
            justify_width = max_len

        winner, loser, winner_score, loser_score = predict_score(team1, team2)
        margin = winner_score - loser_score
        predictions.append((winner, loser, winner_score, loser_score, margin))

    for prediction in sorted(predictions, key=lambda t: t[-1], reverse=True):
        winner, loser, winner_score, loser_score, margin = prediction
        print('The', winner.ljust(justify_width), 'are projected to beat the', loser.ljust(justify_width),
              round(winner_score), '-', round(loser_score))


def predict_outcomes(matchups):
    teams = set([matchup[0] for matchup in matchups]).union(set([matchup[1] for matchup in matchups]))
    justify_width = 0

    predictions = list()
    for team1, team2 in matchups:
        team1 = team1.split()[-1]
        team2 = team2.split()[-1]

        if team1 == 'Blazers':
            team1 = 'Trail Blazers'

        if team2 == 'Blazers':
            team2 = 'Trail Blazers'

        max_len = max([len(team1), len(team2)])
        if max_len > justify_width:
            justify_width = max_len

        team1_bt = team_df.at[team1, 'BT']
        team2_bt = team_df.at[team2, 'BT']

        favorite = team1 if team1_bt >= team2_bt else team2
        underdog = team2 if team1_bt >= team2_bt else team1
        favorite_bt = team1_bt if team1_bt >= team2_bt else team2_bt
        underdog_bt = team2_bt if team1_bt >= team2_bt else team1_bt

        favorite_chance = math.exp(favorite_bt) / (math.exp(favorite_bt) + math.exp(underdog_bt))
        predictions.append((favorite, underdog, favorite_chance))

    for prediction in sorted(predictions, key=lambda t: t[-1], reverse=True):
        winner, loser, chance = prediction
        print('The', winner.ljust(justify_width), 'have a', f'{chance * 100:.3f}' + '%', 'chance to beat the', loser)


def get_spread_chance(favorite, underdog, spread):
    if spread > 0:
        return

    intercept = team_df.at[favorite, 'Points Intercept']
    favorite_off_coef = team_df.at[favorite, 'Points Coef']
    underdog_off_coef = team_df.at[underdog, 'Points Coef']
    favorite_def_coef = team_df.at[favorite, 'Points Allowed Coef']
    underdog_def_coef = team_df.at[underdog, 'Points Allowed Coef']

    favorite_lambda = math.exp(intercept + favorite_off_coef + underdog_def_coef)
    underdog_lambda = math.exp(intercept + underdog_off_coef + favorite_def_coef)

    skel = skellam(favorite_lambda, underdog_lambda)
    cover_chance = skel.sf(-spread)
    push_chance = skel.pmf(-spread)
    fail_chance = 1 - cover_chance - push_chance

    return cover_chance, push_chance, fail_chance


def utility(amount, pos_chance, pos_payout, push_chance, neg_chance, pot):
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    def crra_utility(x, alpha=1.5):
        # Risk Averse:
        #   alpha > 1
        # Risk Neutral:
        #   alpha = 1
        # Risk Seeking:
        #   alpha < 1
        if alpha == 1:
            return math.log(x)
        return (math.pow(x, 1 - alpha) - 1) / (1 - alpha)

    u = pos_chance * crra_utility(pot + (amount * pos_payout) - amount) + \
        push_chance * crra_utility(pot) + \
        neg_chance * crra_utility(pot - amount)
    return -u


def all_bets(pot):
    ats_odds = Odds.get_fanduel_odds(sport='nba', future_days=1)
    h2h_odds = Odds.get_fanduel_odds(sport='nba', future_days=1, bet_type='h2h')
    omit_teams = {'Clippers', 'Grizzlies'}

    predict_scores([(t[0], t[1]) for t in ats_odds])
    print()
    predict_outcomes([(t[0], t[1]) for t in h2h_odds])
    print()

    bets = list()
    for game in ats_odds:
        home_team, away_team, home_spread, away_spread, home_american, away_american = game

        home_spread = float(home_spread)
        away_spread = float(away_spread)

        home_team = home_team.split()[-1]
        away_team = away_team.split()[-1]

        if home_team == 'Blazers':
            home_team = 'Trail Blazers'

        if away_team == 'Blazers':
            away_team = 'Trail Blazers'

        if home_team in omit_teams or away_team in omit_teams:
            continue

        favorite = home_team if home_spread < 0.0 else away_team
        underdog = away_team if home_spread < 0.0 else home_team

        favorite_spread = home_spread if home_spread < 0 else away_spread
        underdog_spread = away_spread if home_spread < 0 else home_spread

        favorite_american = home_american if home_spread < 0 else away_american
        underdog_american = away_american if home_spread < 0 else home_american

        cover_chance, push_chance, fail_chance = get_spread_chance(favorite, underdog, favorite_spread)

        favorite_chance = Odds.convert_american_to_probability(favorite_american)
        underdog_chance = Odds.convert_american_to_probability(underdog_american)

        favorite_payout = 1 / favorite_chance
        underdog_payout = 1 / underdog_chance

        expected_favorite_payout = favorite_payout * cover_chance + push_chance
        expected_underdog_payout = underdog_payout * fail_chance + push_chance

        amount = 1.5
        favorite_bet_pct = minimize_scalar(utility,
                                           amount,
                                           bounds=(0.0, 1),
                                           args=(cover_chance, favorite_payout, push_chance, fail_chance, 1))
        underdog_bet_pct = minimize_scalar(utility,
                                           amount,
                                           bounds=(0.0, 1),
                                           args=(fail_chance, underdog_payout, push_chance, cover_chance, 1))

        favorite_bet_amount = favorite_bet_pct.x * pot
        underdog_bet_amount = underdog_bet_pct.x * pot

        favorite_row = {'Team': favorite,
                        'Spread': favorite_spread,
                        'Opponent': underdog,
                        'American Odds': favorite_american,
                        'Probability': favorite_chance,
                        'Payout': favorite_payout,
                        'Model Chance': cover_chance,
                        'Push Chance': push_chance,
                        'Expected Value': expected_favorite_payout,
                        'Bet Percent': favorite_bet_pct.x,
                        'Bet Amount': favorite_bet_amount,
                        'Bet Type': 'ATS'}

        underdog_row = {'Team': underdog,
                        'Spread': underdog_spread,
                        'Opponent': favorite,
                        'American Odds': underdog_american,
                        'Probability': underdog_chance,
                        'Payout': underdog_payout,
                        'Model Chance': fail_chance,
                        'Push Chance': push_chance,
                        'Expected Value': expected_underdog_payout,
                        'Bet Percent': underdog_bet_pct.x,
                        'Bet Amount': underdog_bet_amount,
                        'Bet Type': 'ATS'}

        bets.append(favorite_row)
        bets.append(underdog_row)

    for game in h2h_odds:
        home_team, away_team, _, _, home_american, away_american = game

        home_team = home_team.split()[-1]
        away_team = away_team.split()[-1]

        if home_team == 'Blazers':
            home_team = 'Trail Blazers'

        if away_team == 'Blazers':
            away_team = 'Trail Blazers'

        if home_team in omit_teams or away_team in omit_teams:
            continue

        home_bt = team_df.at[home_team, 'BT']
        away_bt = team_df.at[away_team, 'BT']

        home_chance = Odds.convert_american_to_probability(home_american)
        away_chance = Odds.convert_american_to_probability(away_american)

        home_payout = 1 / home_chance
        away_payout = 1 / away_chance

        home_bt_chance = math.exp(home_bt) / (math.exp(home_bt) + math.exp(away_bt))
        away_bt_chance = math.exp(away_bt) / (math.exp(home_bt) + math.exp(away_bt))

        expected_home_payout = home_payout * home_bt_chance
        expected_away_payout = away_payout * away_bt_chance

        amount = 1.5
        home_bet_pct = minimize_scalar(utility,
                                       amount,
                                       bounds=(0.0, 1),
                                       args=(home_bt_chance, home_payout, 0, 1 - home_bt_chance, 1))
        away_bet_pct = minimize_scalar(utility,
                                       amount,
                                       bounds=(0.0, 1),
                                       args=(away_bt_chance, away_payout, 0, 1 - away_bt_chance, 1))

        home_bet_amount = home_bet_pct.x * pot
        away_bet_amount = away_bet_pct.x * pot

        home_row = {'Team': home_team,
                    'Spread': 0,
                    'Opponent': away_team,
                    'American Odds': home_american,
                    'Probability': home_chance,
                    'Payout': home_payout,
                    'Model Chance': home_bt_chance,
                    'Push Chance': 0,
                    'Expected Value': expected_home_payout,
                    'Bet Percent': home_bet_pct.x,
                    'Bet Amount': home_bet_amount,
                    'Bet Type': 'H2H'}

        away_row = {'Team': away_team,
                    'Spread': 0,
                    'Opponent': home_team,
                    'American Odds': away_american,
                    'Probability': away_chance,
                    'Payout': away_payout,
                    'Model Chance': away_bt_chance,
                    'Push Chance': 0,
                    'Expected Value': expected_away_payout,
                    'Bet Percent': away_bet_pct.x,
                    'Bet Amount': away_bet_amount,
                    'Bet Type': 'H2H'}

        bets.append(home_row)
        bets.append(away_row)

    bet_df = pd.DataFrame(bets)
    bet_df = bet_df.sort_values(by='Bet Percent', ascending=False)
    remaining_pot = pot
    for index, row in bet_df.iterrows():
        bet_amount = bet_df.at[index, 'Bet Percent'] * remaining_pot
        bet_df.at[index, 'Bet Amount'] = bet_amount
        remaining_pot = remaining_pot - bet_amount

    bet_df['Expected Return'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Expected Value'], axis=1)
    bet_df['Expected Profit'] = bet_df.apply(lambda r: r['Expected Return'] - r['Bet Amount'], axis=1)
    bet_df['To Win'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Payout'] - r['Bet Amount'], axis=1)

    bet_df['Swing'] = bet_df.apply(lambda r: r['Bet Amount'] + r['To Win'], axis=1)
    bet_df = bet_df.sort_values(by='Swing', ascending=False)

    good_bet_df = bet_df.loc[bet_df['Expected Value'] > 1].reset_index(drop=True)
    bad_bet_df = bet_df.loc[bet_df['Expected Value'] <= 1].reset_index(drop=True)

    green = '\033[32m'
    red = '\033[31m'
    print_bet_table(good_bet_df, is_ats=True, color=green)
    print_bet_table(bad_bet_df, is_ats=True, color=red)


# def ats_bets(pot):
#     odds = Odds.get_fanduel_odds(sport='nba', future_days=1)
#     omit_teams = {}
#
#     predict_scores([(t[0], t[1]) for t in odds])
#
#     bets = list()
#     for game in odds:
#         home_team, away_team, home_spread, away_spread, home_american, away_american = game
#
#         home_spread = float(home_spread)
#         away_spread = float(away_spread)
#
#         home_team = home_team.split()[-1]
#         away_team = away_team.split()[-1]
#
#         if home_team == 'Blazers':
#             home_team = 'Trail Blazers'
#
#         if away_team == 'Blazers':
#             away_team = 'Trail Blazers'
#
#         if home_team in omit_teams or away_team in omit_teams:
#             continue
#
#         favorite = home_team if home_spread < 0.0 else away_team
#         underdog = away_team if home_spread < 0.0 else home_team
#
#         favorite_spread = home_spread if home_spread < 0 else away_spread
#         underdog_spread = away_spread if home_spread < 0 else home_spread
#
#         favorite_american = home_american if home_spread < 0 else away_american
#         underdog_american = away_american if home_spread < 0 else home_american
#
#         cover_chance, push_chance, fail_chance = get_spread_chance(favorite, underdog, favorite_spread)
#
#         favorite_chance = Odds.convert_american_to_probability(favorite_american)
#         underdog_chance = Odds.convert_american_to_probability(underdog_american)
#
#         favorite_payout = 1 / favorite_chance
#         underdog_payout = 1 / underdog_chance
#
#         expected_favorite_payout = favorite_payout * cover_chance + push_chance
#         expected_underdog_payout = underdog_payout * fail_chance + push_chance
#
#         amount = 1.5
#         favorite_bet_pct = minimize_scalar(utility,
#                                            amount,
#                                            bounds=(0.0, 1),
#                                            args=(cover_chance, favorite_payout, push_chance, fail_chance, 1))
#         underdog_bet_pct = minimize_scalar(utility,
#                                            amount,
#                                            bounds=(0.0, 1),
#                                            args=(fail_chance, underdog_payout, push_chance, cover_chance, 1))
#
#         favorite_bet_amount = favorite_bet_pct.x * pot
#         underdog_bet_amount = underdog_bet_pct.x * pot
#
#         favorite_row = {'Team': favorite,
#                         'Spread': favorite_spread,
#                         'Opponent': underdog,
#                         'American Odds': favorite_american,
#                         'Probability': favorite_chance,
#                         'Payout': favorite_payout,
#                         'Poisson Chance': cover_chance,
#                         'Push Chance': push_chance,
#                         'Expected Value': expected_favorite_payout,
#                         'Bet Percent': favorite_bet_pct.x,
#                         'Bet Amount': favorite_bet_amount}
#
#         underdog_row = {'Team': underdog,
#                         'Spread': underdog_spread,
#                         'Opponent': favorite,
#                         'American Odds': underdog_american,
#                         'Probability': underdog_chance,
#                         'Payout': underdog_payout,
#                         'Poisson Chance': fail_chance,
#                         'Push Chance': push_chance,
#                         'Expected Value': expected_underdog_payout,
#                         'Bet Percent': underdog_bet_pct.x,
#                         'Bet Amount': underdog_bet_amount}
#
#         bets.append(favorite_row)
#         bets.append(underdog_row)
#
#     bet_df = pd.DataFrame(bets)
#     bet_df = bet_df.sort_values(by='Bet Percent', ascending=False)
#     remaining_pot = pot
#     for index, row in bet_df.iterrows():
#         bet_amount = bet_df.at[index, 'Bet Percent'] * remaining_pot
#         bet_df.at[index, 'Bet Amount'] = bet_amount
#         remaining_pot = remaining_pot - bet_amount
#
#     bet_df['Expected Return'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Expected Value'], axis=1)
#     bet_df['Expected Profit'] = bet_df.apply(lambda r: r['Expected Return'] - r['Bet Amount'], axis=1)
#     bet_df['To Win'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Payout'] - r['Bet Amount'], axis=1)
#
#     bet_df['Swing'] = bet_df.apply(lambda r: r['Bet Amount'] + r['To Win'], axis=1)
#     bet_df = bet_df.sort_values(by='Swing', ascending=False)
#
#     good_bet_df = bet_df.loc[bet_df['Expected Value'] > 1].reset_index(drop=True)
#     bad_bet_df = bet_df.loc[bet_df['Expected Value'] <= 1].reset_index(drop=True)
#
#     green = '\033[32m'
#     red = '\033[31m'
#     print_bet_table(good_bet_df, is_ats=True, color=green)
#     print_bet_table(bad_bet_df, is_ats=True, color=red)
#
#
# def straight_up_bets(pot):
#     odds = Odds.get_fanduel_odds(sport='nba', future_days=1, bet_type='h2h')
#     omit_teams = {}
#
#     predict_outcomes([(t[0], t[1]) for t in odds])
#
#     bets = list()
#     for game in odds:
#         home_team, away_team, _, _, home_american, away_american = game
#
#         home_team = home_team.split()[-1]
#         away_team = away_team.split()[-1]
#
#         if home_team == 'Blazers':
#             home_team = 'Trail Blazers'
#
#         if away_team == 'Blazers':
#             away_team = 'Trail Blazers'
#
#         if home_team in omit_teams or away_team in omit_teams:
#             continue
#
#         home_bt = team_df.at[home_team, 'BT']
#         away_bt = team_df.at[away_team, 'BT']
#
#         home_chance = Odds.convert_american_to_probability(home_american)
#         away_chance = Odds.convert_american_to_probability(away_american)
#
#         home_payout = 1 / home_chance
#         away_payout = 1 / away_chance
#
#         home_bt_chance = math.exp(home_bt) / (math.exp(home_bt) + math.exp(away_bt))
#         away_bt_chance = math.exp(away_bt) / (math.exp(home_bt) + math.exp(away_bt))
#
#         expected_home_payout = home_payout * home_bt_chance
#         expected_away_payout = away_payout * away_bt_chance
#
#         amount = 1.5
#         home_bet_pct = minimize_scalar(utility,
#                                        amount,
#                                        bounds=(0.0, 1),
#                                        args=(home_bt_chance, home_payout, 0, 1 - home_bt_chance, 1))
#         away_bet_pct = minimize_scalar(utility,
#                                        amount,
#                                        bounds=(0.0, 1),
#                                        args=(away_bt_chance, away_payout, 0, 1 - away_bt_chance, 1))
#
#         home_bet_amount = home_bet_pct.x * pot
#         away_bet_amount = away_bet_pct.x * pot
#
#         home_row = {'Team': home_team,
#                     'Opponent': away_team,
#                     'American Odds': home_american,
#                     'Probability': home_chance,
#                     'Payout': home_payout,
#                     'BT Chance': home_bt_chance,
#                     'Expected Value': expected_home_payout,
#                     'Bet Percent': home_bet_pct.x,
#                     'Bet Amount': home_bet_amount}
#
#         away_row = {'Team': away_team,
#                     'Opponent': home_team,
#                     'American Odds': away_american,
#                     'Probability': away_chance,
#                     'Payout': away_payout,
#                     'BT Chance': away_bt_chance,
#                     'Expected Value': expected_away_payout,
#                     'Bet Percent': away_bet_pct.x,
#                     'Bet Amount': away_bet_amount}
#
#         bets.append(home_row)
#         bets.append(away_row)
#
#     bet_df = pd.DataFrame(bets)
#     bet_df = bet_df.sort_values(by='Bet Percent', ascending=False)
#     remaining_pot = pot
#     for index, row in bet_df.iterrows():
#         bet_amount = bet_df.at[index, 'Bet Percent'] * remaining_pot
#         bet_df.at[index, 'Bet Amount'] = bet_amount
#         remaining_pot = remaining_pot - bet_amount
#
#     bet_df['Expected Return'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Expected Value'], axis=1)
#     bet_df['Expected Profit'] = bet_df.apply(lambda r: r['Expected Return'] - r['Bet Amount'], axis=1)
#     bet_df['To Win'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Payout'] - r['Bet Amount'], axis=1)
#
#     bet_df['Swing'] = bet_df.apply(lambda r: r['Bet Amount'] + r['To Win'], axis=1)
#     bet_df = bet_df.sort_values(by='Swing', ascending=False)
#
#     good_bet_df = bet_df.loc[bet_df['Expected Value'] > 1].reset_index(drop=True)
#     bad_bet_df = bet_df.loc[bet_df['Expected Value'] <= 1].reset_index(drop=True)
#
#     green = '\033[32m'
#     red = '\033[31m'
#     print_bet_table(good_bet_df, color=green)
#     print_bet_table(bad_bet_df, color=red)


def print_bet_table(df, is_ats=False, color=''):
    columns = ['Num', 'Team', 'Spread', 'Opponent', 'American Odds', 'Probability', 'Payout', 'Model Chance',
               'Push Chance', 'Bet Amount', 'Expected Value', 'Expected Return', 'Expected Profit', 'To Win']

    table = PrettyTable(columns)
    table.float_format = '0.3'

    stop = '\033[0m'

    row_num = 1
    for index, row in df.iterrows():
        table_row = list()
        table_row.append(color + str(row_num) + stop)
        row_num = row_num + 1
        for col in columns[1:]:
            if col == 'American Odds':
                val = '+' + str(row[col]) if row[col] > 0 else str(row[col])
            elif col == 'Spread':
                val = '--' if row['Bet Type'] == 'H2H' else str(row[col])
            elif col == 'Payout' or col == 'Expected Value':
                val = str(round(row[col], 2)) + 'x'
            elif col == 'Probability' or col == 'Model Chance' or col == 'Push Chance':
                val = f'{row[col] * 100:.3f}' + '%'
            elif col == 'Bet Amount' or col == 'Expected Return' or col == 'Expected Profit' or col == 'To Win':
                val = 0 if col == 'Expected Profit' and -.01 < row[col] < 0 else row[col]
                val = '${:,.2f}'.format(val)
            elif col == 'Expected Value':
                val = str(round(row[col], 2))
            else:
                val = str(row[col])
            table_row.append(color + val + stop)

        table.add_row(table_row)

    # Print the table
    print(table)
    print()