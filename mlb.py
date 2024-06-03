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
from bokeh.models import ColumnDataSource, FixedTicker, NumeralTickFormatter, Whisker
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap, jitter
from bokeh.layouts import gridplot
from prettytable import PrettyTable
from scipy.optimize import minimize
from scipy.stats import chi2, norm

domain = 'https://api.natstat.com/v2/'
api_key = '87ab-ba0b1b'

graph = nx.MultiDiGraph()
team_df = pd.DataFrame(columns=['Team', 'League', 'Division', 'Games Played', 'Wins', 'Losses',
                                'Preseason BT', 'Bayes BT', 'BT', 'BT Var', 'BT Pct', 'Win Pct',
                                'Preseason Projected Wins', 'Preseason Projected Win SD',
                                'Preseason Projected Wins Lower', 'Preseason Projected Wins Upper',
                                'Projected Wins', 'Projected Win SD', 'Projected Wins Lower', 'Projected Wins Upper',
                                'Points Intercept', 'Points Coef', 'Points Allowed Coef',
                                'Adjusted Points', 'Adjusted Points Allowed', 'Adjusted Point Diff',
                                'Primary', 'Secondary'])


def get(endpoint, params):
    req = requests.get(domain + endpoint, params=params)
    resp = req.json()
    return resp


def get_games(page=1):
    params = {'key': api_key,
              'format': 'json',
              'start': '2024-03-20',
              'end': '2024-09-30',
              'page': page,
              'max': 1000}
    endpoint = 'games/MLB/'
    return get(endpoint, params=params)


def get_pitches(game, page=1):
    params = {'key': api_key,
              'format': 'json',
              'game': game,
              'page': page,
              'max': 1000}
    endpoint = 'pitch/MLB/'
    return get(endpoint, params=params)


def get_starting_pitchers(game_id):
    pitches = get_pitches(game_id)
    total_pages = int(pitches.get('meta').get('Total_Pages'))
    all_pitches = dict()
    for page in range(total_pages):
        pitches = get_pitches(game_id, page=page + 1)
        all_pitches.update(pitches.get('pitches'))
    pitches_df = pd.DataFrame(all_pitches).T

    pitches_df = pitches_df.loc[pitches_df['Inning'] == '1']
    sp_df = pd.DataFrame(list([r.get('pitcher') for r in pitches_df['players']]))

    sp_df['Team'] = fix_team_names(sp_df['Team'])
    sp_df = sp_df.drop_duplicates(subset=['Team'])

    sp_dict = {row['Team']: row['Name'] for index, row in sp_df.iterrows()}

    return sp_dict


def get_postseason(row, df):
    df = df.sort_values(by=['Projected Wins', 'BT'], ascending=False)
    team_order = list(df['Team'])
    team = row['Team']

    nl_east = {'Braves', 'Marlins', 'Phillies', 'Mets', 'Nationals'}
    nl_central = {'Brewers', 'Cubs', 'Cardinals', 'Pirates', 'Reds'}
    nl_west = {'Diamondbacks', 'Giants', 'Padres', 'Dodgers', 'Rockies'}
    nl_teams = nl_east.union(nl_central.union(nl_west))

    al_east = {'Orioles', 'Rays', 'Blue Jays', 'Red Sox', 'Yankees'}
    al_central = {'Twins', 'Guardians', 'White Sox', 'Tigers', 'Royals'}
    al_west = {'Angels', 'Mariners', 'Astros', 'Rangers', 'Athletics'}
    al_teams = al_east.union(al_central.union(al_west))

    other_teams = nl_teams.union(al_teams) - {team}

    division_map = {t: 'NL East' for t in nl_east}
    division_map.update({t: 'NL Central' for t in nl_central})
    division_map.update({t: 'NL West' for t in nl_west})
    division_map.update({t: 'AL East' for t in al_east})
    division_map.update({t: 'AL Central' for t in al_central})
    division_map.update({t: 'AL West' for t in al_west})

    conference_map = {t: 'NL' for t in nl_teams}
    conference_map.update({t: 'AL' for t in al_teams})

    teams_ahead = {other_team for other_team in other_teams if team_order.index(other_team) < team_order.index(team)}
    divisons_ahead = [division_map.get(other_team) for other_team in teams_ahead
                      if conference_map.get(other_team) == conference_map.get(team)]
    divisons_ahead_count = {div: divisons_ahead.count(div) for div in divisons_ahead}

    divisional_teams_ahead = {other_team for other_team in teams_ahead
                              if division_map.get(other_team) == division_map.get(team)}

    if len(divisional_teams_ahead) == 0:
        return division_map.get(team)

    wild_card_divs_ahead = {div: count for div, count in divisons_ahead_count.items() if count > 1}
    wild_card_teams_ahead = sum([count - 1 for count in wild_card_divs_ahead.values()])

    if wild_card_teams_ahead == 0:
        return conference_map.get(team) + ' WC 1'
    elif wild_card_teams_ahead == 1:
        return conference_map.get(team) + ' WC 2'
    elif wild_card_teams_ahead == 2:
        return conference_map.get(team) + ' WC 3'

    return ''


# -----------------------------------------------------------------


def fix_team_names(names):
    cities = ['Atlanta',
              'Baltimore',
              'Tampa Bay',
              'Texas',
              'Los Angeles',
              'Toronto',
              'Houston',
              'Philadelphia',
              'New York',
              'Seattle',
              'Milwaukee',
              'San Francisco',
              'Boston',
              'Chicago',
              'Cincinnati',
              'Miami',
              'Minnesota',
              'Arizona',
              'San Diego',
              'Pittsburgh',
              'Cleveland',
              'Washington',
              'St. Louis',
              'Detroit',
              'Colorado',
              'Kansas City',
              'Oakland']

    new_names = list()
    for name in names:
        for city in cities:
            name = name.replace(city, '').strip()
        new_names.append(name)

    return new_names


def get_team_ranking(games_graph):
    nodes = games_graph.nodes
    df = pd.DataFrame(nx.to_numpy_array(games_graph), columns=nodes)
    df.index = nodes

    teams = list(df.index)
    df = df.fillna(0)

    teams_to_index = {team: i for i, team in enumerate(teams)}
    index_to_teams = {i: team for team, i in teams_to_index.items()}

    graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    edges = [list(itertools.repeat((teams_to_index.get(team2),
                                    teams_to_index.get(team1)),
                                   int(weight_dict.get('weight'))))
             for team1, team2, weight_dict in graph.edges.data()]
    edges = list(itertools.chain.from_iterable(edges))

    try:
        coeffs, cov = choix.ep_pairwise(n_items=len(teams), data=edges, alpha=1.0)
        coeffs = pd.Series(coeffs)
        cov = pd.Series(cov.diagonal())
        coef_df = pd.DataFrame([coeffs, cov]).T
        coef_df.columns = ['BT', 'Var']
        coef_df.index = [index_to_teams.get(index) for index in coef_df.index]
    except np.linalg.LinAlgError:
        print('EP Failed')

        coeffs = pd.Series(choix.opt_pairwise(n_items=len(teams), data=edges))
        coeffs = coeffs.sort_values(ascending=False)
        coeffs = {index_to_teams.get(index): coeff for index, coeff in coeffs.items()}
        coef_df = pd.DataFrame(columns=['BT', 'Var'], index=coeffs.keys())
        for team, bt in coeffs.items():
            coef_df.at[team, 'BT'] = coeffs.get(team)
            coef_df.at[team, 'Var'] = 1.0
    coef_df = coef_df.sort_values(by='BT', ascending=False)
    return coef_df


def get_team_colors(team):
    color_map = {'Dodgers': ('#005A9C', '#EF3E42'),
                 'Braves': ('#CE1141', '#13274F'),
                 'Astros': ('#002D62', '#EB6E1F'),
                 'Yankees': ('#003087', '#E4002C'),
                 'Phillies': ('#E81828', '#002D72'),
                 'Orioles': ('#DF4601', '#000000'),
                 'Rangers': ('#003278', '#C0111F'),
                 'Mariners': ('#0C2C56', '#005C5C'),
                 'Twins': ('#002B5C', '#D31145'),
                 'Blue Jays': ('#134A8E', '#1D2D5C'),
                 'Rays': ('#092C5C', '#8FBCE6'),
                 'Cardinals': ('#C41E3A', '#0C2340'),
                 'Giants': ('#FD5A1E', '#27251F'),
                 'Diamondbacks': ('#A71930', '#E3D4AD'),
                 'Cubs': ('#0E3386', '#CC3433'),
                 'Padres': ('#2F241D', '#FFC425'),
                 'Reds': ('#C6011F', '#000000'),
                 'Mets': ('#002D72', '#FF5910'),
                 'Tigers': ('#0C2340', '#FA4616'),
                 'Guardians': ('#00385D', '#E50022'),
                 'Marlins': ('#00A3E0', '#EF3340'),
                 'Red Sox': ('#BD3039', '#0C2340'),
                 'Brewers': ('#FFC52F', '#12284B'),
                 'Pirates': ('#27251F', '#FDB827'),
                 'Royals': ('#004687', '#BD9B60'),
                 'Angels': ('#BA0021', '#003263'),
                 'Nationals': ('#AB0003', '#14225A'),
                 'White Sox': ('#27251F', '#C4CED4'),
                 'Rockies': ('#333366', '#C4CED4'),
                 'Athletics': ('#003831', '#EFB21E')}
    return color_map.get(team)


def run(update=True, include_sp=False, pct_evidence=0.9):
    global team_df

    if update:
        all_games = dict()
        for page in range(3):
            games = get_games(page=page + 1)
            all_games.update(games.get('games'))
        games_df = pd.DataFrame(all_games).T
        games_df = games_df.loc[games_df['GameStatus'] == 'Final']
        mistake_ids = ['5063334', '5063586', '5063239', '5063225', '5063411', '5063774', '5063840', '5063432',
                       '5063447', '5063349', '5102633', '5109903']
        games_df = games_df.loc[~games_df['ID'].isin(mistake_ids)]

        games_df['GameDay'] = pd.to_datetime(games_df['GameDay'])
        # today = datetime.strptime('05-01-24', '%m-%d-%y')
        # games_df = games_df.loc[games_df['GameDay'] < today]

        potential = games_df.duplicated(subset=['Home', 'Visitor', 'ScoreHome', 'ScoreVis', 'Overtime'], keep=False)
        potential_df = games_df.loc[potential]
        potential_df = potential_df.sort_values(by=['Home', 'Visitor', 'ScoreHome', 'ScoreVis'], kind='mergesort')
        non_mistake_ids = ['5063888', '5063864', '5063524', '5063518', '5063285', '5063251', '5063517', '5063499',
                           '5063238', '5063224', '5063901', '5063871']
        potential_df = potential_df.loc[~potential_df['ID'].isin(non_mistake_ids)]

        if not potential_df.empty:
            print('Potentially Incorrect Games')

            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', None)

            print(potential_df)

        games_df['Home'] = fix_team_names(games_df['Home'])
        games_df['Visitor'] = fix_team_names(games_df['Visitor'])

        if include_sp:
            for index, row in games_df.iterrows():
                game_id = games_df.at[index, 'ID']
                sp_dict = get_starting_pitchers(game_id)
                games_df.at[index, 'Home SP'] = sp_dict.get(row['Home'], 'Missing')
                games_df.at[index, 'Away SP'] = sp_dict.get(row['Visitor'], 'Missing')

        schedule = load_schedule()
        remaining_schedule = {team: sched.copy() for team, sched in schedule.items()}
        for team in schedule.keys():
            remaining_games = remaining_schedule.get(team)
            relevant_games = games_df.loc[(games_df['Visitor'] == team) | (games_df['Home'] == team)]
            played_road_opponents = [opp for opp in relevant_games['Visitor'] if opp != team]
            played_home_opponents = [opp for opp in relevant_games['Home'] if opp != team]
            played_opponents = played_road_opponents + played_home_opponents
            for played_opponent in played_opponents:
                try:
                    remaining_games.remove(played_opponent)
                except ValueError as e:
                    print(team, 'vs.', played_opponent)
                    print(e)

        rows = list()
        for index, row in games_df.iterrows():
            obs1 = {'Date': row['GameDay'],
                    'Game ID': row['ID'],
                    'Offense': row['Home'],
                    'Defense': row['Visitor'],
                    'Score': row['ScoreHome'],
                    'Innings': 9 if row['Overtime'] == 'N' else row['Overtime']}
            if include_sp:
                obs1['SP'] = row['Away SP']
            rows.append(obs1)

            obs2 = {'Date': row['GameDay'],
                    'Game ID': row['ID'],
                    'Offense': row['Visitor'],
                    'Defense': row['Home'],
                    'Score': row['ScoreVis'],
                    'Innings': 9 if row['Overtime'] == 'N' else row['Overtime']}
            if include_sp:
                obs2['SP'] = row['Home SP']
            rows.append(obs2)

        games_df = pd.DataFrame(rows)
        games_df.to_csv('C:\\Users\\Colin\\Desktop\\mlb2024games.csv', index=False)
        print('Games Updated')
    else:
        games_df = pd.read_csv('C:\\Users\\Colin\\Desktop\\mlb2024games.csv')
        schedule = load_schedule()
        remaining_schedule = {team: sched.copy() for team, sched in schedule.items()}

    preseason_bts = get_preseason_bts(schedule)
    teams = schedule.keys()

    team_df['Team'] = teams
    team_df['Primary'] = team_df.apply(lambda r: get_team_colors(r['Team'])[0], axis=1)
    team_df['Secondary'] = team_df.apply(lambda r: get_team_colors(r['Team'])[1], axis=1)

    divisions = {'AL East': ['Orioles', 'Red Sox', 'Yankees', 'Rays', 'Blue Jays'],
                 'AL Central': ['White Sox', 'Guardians', 'Tigers', 'Royals', 'Twins'],
                 'AL West': ['Astros', 'Angels', 'Athletics', 'Mariners', 'Rangers'],
                 'NL East': ['Braves', 'Marlins', 'Mets', 'Phillies', 'Nationals'],
                 'NL Central': ['Cubs', 'Reds', 'Brewers', 'Pirates', 'Cardinals'],
                 'NL West': ['Diamondbacks', 'Rockies', 'Dodgers', 'Padres', 'Giants']}


    team_divs = dict()
    team_leagues = dict()
    for div, ts in divisions.items():
        for team in ts:
            team_divs[team] = div
            team_leagues[team] = div.split(' ')[0].strip()

    team_df['Division'] = team_df.apply(lambda r: team_divs.get(r['Team']), axis=1)
    team_df['League'] = team_df.apply(lambda r: team_leagues.get(r['Team']), axis=1)

    team_df = team_df.set_index('Team')
    bts = get_bt()

    for team in teams:
        team_df.at[team, 'Preseason BT'] = preseason_bts.at[team, 'BT']
        team_df.at[team, 'BT'] = preseason_bts.at[team, 'BT']

    for team in teams:
        mean, dev = get_total_wins_approx(team, schedule, wins=0, losses=0)
        team_df.at[team, 'Preseason Projected Wins'] = mean
        team_df.at[team, 'Preseason Projected Win SD'] = dev
        wins95_lower = max(0, mean - norm.ppf(.975) * dev)
        wins95_upper = min(162, mean + norm.ppf(.975) * dev)
        team_df.at[team, 'Preseason Projected Wins Lower'] = wins95_lower
        team_df.at[team, 'Preseason Projected Wins Upper'] = wins95_upper

    for team in teams:
        wins = graph.in_degree(team) if team in graph.nodes else 0
        losses = graph.out_degree(team) if team in graph.nodes else 0
        team_df.at[team, 'Wins'] = wins
        team_df.at[team, 'Losses'] = losses
        team_df.at[team, 'Games Played'] = wins + losses

        team_df.at[team, 'BT'] = bts.at[team, 'BT'] if team in bts.index else preseason_bts.at[team, 'BT']
        team_df.at[team, 'BT Var'] = bts.at[team, 'Var'] if team in bts.index else 1

        pct_evidence = 1e-16 if pct_evidence == 0 else pct_evidence
        evidence_factor = (162 - pct_evidence * 162) / pct_evidence
        evidence_factor = 1e-16 if evidence_factor == 0 else evidence_factor

        bayes_bt = get_bayes_avg(team_df.at[team, 'Preseason BT'],
                                 team_df.at[team, 'BT Var'] / evidence_factor,
                                 team_df.at[team, 'BT'],
                                 team_df.at[team, 'BT Var'],
                                 wins + losses)

        team_df.at[team, 'Bayes BT'] = bayes_bt

    for team in teams:
        wins = graph.in_degree(team) if team in graph.nodes else 0
        losses = graph.out_degree(team) if team in graph.nodes else 0
        mean, dev = get_total_wins_approx(team, remaining_schedule, wins=wins, losses=losses)
        mean = mean + wins
        team_df.at[team, 'Projected Wins'] = mean
        team_df.at[team, 'Projected Win SD'] = dev
        wins95_lower = max(wins, mean - norm.ppf(.975) * dev)
        wins95_upper = min(162, mean + norm.ppf(.975) * dev)
        team_df.at[team, 'Projected Wins Lower'] = wins95_lower
        team_df.at[team, 'Projected Wins Upper'] = wins95_upper

        team_df['Win Pct'] = team_df.apply(lambda r: 0 if r['Games Played'] == 0 else
        r['Wins'] / r['Games Played'], axis=1)

    bt_var = statistics.variance(team_df['Bayes BT'])
    bt_sd = math.sqrt(bt_var)
    bt_norm = norm(0, bt_sd)
    for team_name in team_df.index:
        team_df.at[team_name, 'BT Pct'] = bt_norm.cdf(team_df.at[team_name, 'Bayes BT'])

    team_coefs, pitcher_coefs = fit_runs(include_sp)
    # pitcher_coefs.at[pitcher, 'Intercept'] = intercept
    # pitcher_coefs.at[pitcher, 'Pitching Coef'] = sp_coefs.get(pitcher, 0)

    for team in teams:
        team_df.at[team, 'Points Intercept'] = team_coefs['Intercept'].mean()
        team_df.at[team, 'Points Coef'] = team_coefs.at[team, 'Offense Coef'] if team in team_coefs.index else 0
        team_df.at[team, 'Points Allowed Coef'] = team_coefs.at[team, 'Defense Coef'] if team in team_coefs.index else 0

    team_df['Adjusted Points'] = team_df.apply(lambda r: math.exp(r['Points Intercept'] + r['Points Coef']), axis=1)
    team_df['Adjusted Points Allowed'] = team_df.apply(lambda r: math.exp(r['Points Intercept'] + r['Points Allowed Coef']), axis=1)
    team_df['Adjusted Point Diff'] = team_df.apply(lambda r: r['Adjusted Points'] - r['Adjusted Points Allowed'], axis=1)

    is_playoff_team('Cubs')

    show_off_def()
    show_graph()
    plot_projected_wins()
    plot_projected_wins_div()

    print_table()

    if include_sp:
        pitcher_coefs = pitcher_coefs.sort_values(by='Pitching Coef')
        print(pitcher_coefs)


def load_schedule():
    # remaining_df = pd.read_csv("C:\\Users\\Colin\\Desktop\\mlb2024games.csv")
    # teams = set(remaining_df['Home'].unique()).union(set(remaining_df['Visitor'].unique()))
    # schedule_dict = dict()
    # for team in teams:
    #     relevant_games = remaining_df.loc[(remaining_df['Home'] == team) | (remaining_df['Visitor'] == team)]
    #     opponents = [row['Home'] if row['Home'] != team else row['Visitor'] for index, row in relevant_games.iterrows()]
    #     team = fix_team_names([team])[0]
    #     schedule_dict[team] = fix_team_names(opponents)
    # with open('C:\\Users\\Colin\\Desktop\\mlb2024schedule.json', 'w') as f:
    #     json.dump(schedule_dict, f)

    schedule_path = 'C:\\Users\\Colin\\Desktop\\mlb2024schedule.json'
    with open(schedule_path, 'r') as f:
        schedule = json.load(f)
        return schedule


def get_bayes_avg(prior_avg, prior_var, sample_avg, sample_var, n):
    k_0 = sample_var / prior_var
    posterior_avg = ((k_0 / (k_0 + n)) * prior_avg) + ((n / (k_0 + n)) * sample_avg)
    return posterior_avg


def get_preseason_bts(schedules, use_mse=True):
    win_totals = {'Dodgers': 102.5,
                  'Braves': 101.5,
                  'Astros': 93.5,
                  'Yankees': 90.5,
                  'Phillies': 89.5,
                  'Orioles': 89.5,
                  'Rangers': 88.5,
                  'Mariners': 87.5,
                  'Twins': 87.5,
                  'Blue Jays': 86.5,
                  'Rays': 85.5,
                  'Cardinals': 84.5,
                  'Giants': 83.5,
                  'Diamondbacks': 83.5,
                  'Cubs': 83.5,
                  'Padres': 83.5,
                  'Reds': 81.5,
                  'Mets': 81.5,
                  'Tigers': 80.5,
                  'Guardians': 79.5,
                  'Marlins': 77.5,
                  'Red Sox': 77.5,
                  'Brewers': 77.5,
                  'Pirates': 75.5,
                  'Royals': 73.5,
                  'Angels': 72.5,
                  'Nationals': 66.5,
                  'White Sox': 61.5,
                  'Rockies': 59.5,
                  'Athletics': 58.5}

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

    df = df.set_index('Team', drop=True)
    df = df[['BT', 'Odds Projection', 'BT Projection']]
    df['BT Projection'] = df['BT Projection'].round(1)
    # df.to_csv('C:\\Users\\Colin\\Desktop\\mlb2024preseason_bts.csv')

    return df


def get_bt():
    games_df = pd.read_csv("C:\\Users\\Colin\\Desktop\\mlb2024games.csv")
    complete_games_df = games_df.loc[games_df['Score'].notna()]
    for game_id in complete_games_df['Game ID'].unique():
        matching_games = complete_games_df.loc[complete_games_df['Game ID'] == game_id].reset_index()
        assert len(matching_games) == 2
        r1 = matching_games.loc[0]
        r2 = matching_games.loc[1]
        team1 = r1['Offense']
        team2 = r2['Offense']
        team1_score = r1['Score']
        team2_score = r2['Score']

        if team1_score > team2_score:
            graph.add_edge(team2, team1)
        else:
            graph.add_edge(team1, team2)

    bts = get_team_ranking(graph)
    return bts


def get_remaining_win_probs(team_name, remaining_schedule, wins=0, losses=0, metric='Bayes BT'):
    remaining_opponents = remaining_schedule.get(team_name)

    bt = team_df.at[team_name, 'Bayes BT']
    games_played = wins + losses
    if games_played == 162:
        return []

    # remaining_opponents = opponents[games_played:162]
    opponent_bts = [team_df.at[opponent, metric] for opponent in remaining_opponents]
    win_probs = [math.exp(bt) / (math.exp(bt) + math.exp(opp_bt)) for opp_bt in opponent_bts]

    return win_probs


def get_total_wins_chances(team, schedule, wins=0, losses=0, metric='Bayes BT'):
    # This is infeasible when len(win_probs) is large
    wins_dict = {win_total: 0.0 for win_total in range(163)}

    win_probs = get_remaining_win_probs(team, schedule, wins=wins, losses=losses, metric=metric)
    loss_probs = [1 - win_prob for win_prob in win_probs]

    for win_combo in itertools.product([0, 1], repeat=len(win_probs)):
        loss_combo = [0 if game == 1 else 1 for game in win_combo]

        win_combo_probs = list(itertools.compress(win_probs, win_combo))
        loss_combo_probs = list(itertools.compress(loss_probs, loss_combo))
        win_combo_wins = len(win_combo_probs) + wins

        total_wins_prob = np.prod(win_combo_probs)
        total_losses_prob = np.prod(loss_combo_probs)

        combo_prob = total_wins_prob * total_losses_prob

        wins_dict[win_combo_wins] = wins_dict.get(win_combo_wins) + combo_prob

    return wins_dict


def get_total_wins_approx(team, schedule, wins=0, losses=0, metric='Bayes BT'):
    win_probs = get_remaining_win_probs(team, schedule, wins=wins, losses=losses, metric=metric)
    loss_probs = [1 - win_prob for win_prob in win_probs]

    mean = sum(win_probs)
    var = sum([wp * lp for wp, lp in zip(win_probs, loss_probs)])

    return mean, math.sqrt(var)


def fit_runs(include_sp):
    df = pd.read_csv("C:\\Users\\Colin\\Desktop\\mlb2024games.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    df['Runs_Per_Inning'] = df.apply(lambda r: 0 if r['Innings'] == 0 else r['Score'] / r['Innings'], axis=1)
    df['Runs_Per_9'] = df.apply(lambda r: r['Runs_Per_Inning'] * 9, axis=1)
    df['Weight'] = df.apply(lambda r: r['Innings'] / 9, axis=1)

    def get_days_ago(r):
        game_date = r['Date']
        today = datetime.today()
        delta = today - game_date
        days_ago = delta.days
        return days_ago

    df['Days_Ago'] = df.apply(lambda r: get_days_ago(r), axis=1)
    # df['Weight'] = df.apply(lambda r: r['Weight'] * .9999 ** r['Days_Ago'], axis=1)

    df = df.loc[df['Score'].notna()]
    df = df.dropna()

    model, dispersion, is_poisson = fit_neg_bin(df, include_sp)

    off_coefs = {name.replace('Offense', '').replace('[T.', '').replace(']', ''): coef
                 for name, coef in model.params.items() if name.startswith('Offense')}
    def_coefs = {name.replace('Defense', '').replace('[T.', '').replace(']', ''): coef
                 for name, coef in model.params.items() if name.startswith('Defense')}
    sp_coefs = {name.replace('SP', '').replace('[T.', '').replace(']', ''): coef
                for name, coef in model.params.items() if name.startswith('SP')}
    intercept = model.params['Intercept']

    teams = set(df['Offense'].unique()).union(set(df['Defense'].unique()))

    team_coefs = pd.DataFrame(columns=['Intercept', 'Offense Coef', 'Defense Coef'])
    team_coefs['Team'] = list(teams)
    team_coefs = team_coefs.set_index('Team')
    for team in teams:
        team_coefs.at[team, 'Intercept'] = intercept
        team_coefs.at[team, 'Offense Coef'] = off_coefs.get(team, 0)
        team_coefs.at[team, 'Defense Coef'] = def_coefs.get(team, 0)

    pitcher_teams = dict()
    team_pitchers = dict()
    pitcher_coefs = None
    if include_sp:
        pitchers = df['SP'].unique()
        pitcher_coefs = pd.DataFrame(columns=['Intercept', 'Pitching Coef'])
        pitcher_coefs['Pitcher'] = pitchers
        pitcher_coefs = pitcher_coefs.set_index('Pitcher')
        for pitcher in pitchers:
            pitcher_coefs.at[pitcher, 'Intercept'] = intercept
            pitcher_coefs.at[pitcher, 'Pitching Coef'] = sp_coefs.get(pitcher, 0)

            relevant_games = df.loc[df['SP'] == pitcher]
            pitcher_team = relevant_games['Defense'].mode()
            pitcher_teams[pitcher] = pitcher_team.squeeze()
            pitcher_coefs.at[pitcher, 'Team'] = pitcher_team.squeeze()
            team_pitchers[pitcher_team.squeeze()] = team_pitchers.get(pitcher_team.squeeze(), []) + [pitcher]

        for team, row in team_coefs.iterrows():
            pitchers = team_pitchers.get(team)
            sp_coefs = pitcher_coefs.loc[team_pitchers.get(team)]
            sp_coefs['Value'] = sp_coefs.apply(lambda r: math.exp(r['Intercept'] + r['Pitching Coef']), axis=1)

            avg_value = sp_coefs['Value'].mean()
            team_defense_coef = math.log(avg_value) - intercept

            team_coefs.at[team, 'Defense Coef'] = team_defense_coef

    return team_coefs, pitcher_coefs


def fit_neg_bin(df, include_sp, verbose=True):
    formula = "Runs_Per_9 ~ Offense + SP" if include_sp else "Runs_Per_9 ~ Offense + Defense"
    poisson_model = smf.glm(formula=formula,
                            data=df,
                            family=sm.families.Poisson(),
                            var_weights=df['Weight']).fit(method="lbfgs",
                                                          maxiter=int(1e5))

    pearson_chi2 = chi2(df=poisson_model.df_resid)
    alpha = .05
    p_value = pearson_chi2.sf(poisson_model.pearson_chi2)
    poisson_good_fit = p_value >= alpha
    if verbose:
        print(poisson_model.summary())
        print()

        print('Critical Value for alpha=.05:', pearson_chi2.ppf(1 - alpha))
        print('Test Statistic:              ', poisson_model.pearson_chi2)
        print('P-Value:                     ', p_value)
        if not poisson_good_fit:
            print('The Poisson Model is not a good fit for the data')
        else:
            print('The Poisson Model is a good fit for the data')
        print()

    if poisson_good_fit:
        return poisson_model, 0, True

    df['event_rate'] = poisson_model.mu
    df['auxiliary_reg'] = df.apply(lambda x: ((x['Runs_Per_9'] - x['event_rate']) ** 2 -
                                               x['event_rate']) / x['event_rate'], axis=1)
    aux_olsr_results = smf.ols("auxiliary_reg ~ event_rate - 1", data=df).fit()

    relevant_disp_param = aux_olsr_results.f_pvalue < alpha
    if not relevant_disp_param:
        print('The dispersion parameter', '(' + str(aux_olsr_results.params[0]) + ')',
              'is not statistically relevant: ', aux_olsr_results.f_pvalue)
        return poisson_model, 0, True

    if aux_olsr_results.params[0] < 0:
        print('The dispersion parameter is negative:', aux_olsr_results.params[0])
        return poisson_model, 0, True

    if verbose:
        print('Dispersion Parameter:', aux_olsr_results.params[0])
        print(aux_olsr_results.summary())
        print()

    # https://www.statsmodels.org/dev/generated/statsmodels.genmod.families.family.NegativeBinomial.html
    neg_bin_model = smf.glm(formula=formula,
                            data=df,
                            family=sm.genmod.families.family.NegativeBinomial(alpha=aux_olsr_results.params[0]),
                            var_weights=df['Weight']).fit(method="lbfgs")

    pearson_chi2 = chi2(df=neg_bin_model.df_resid)
    p_value = pearson_chi2.sf(neg_bin_model.pearson_chi2)
    neg_bin_good_fit = p_value >= alpha
    if verbose:
        print(neg_bin_model.summary())
        print()

        print('Critical Value for alpha=.05:', pearson_chi2.ppf(1 - alpha))
        print('Test Statistic:              ', neg_bin_model.pearson_chi2)
        print('P-Value:                     ', p_value)
        if not neg_bin_good_fit:
            print('The Negative Binomial Model is not a good fit for the data')
        else:
            print('The Negative Binomial Model is a good fit for the data')
        print()

    return neg_bin_model, aux_olsr_results.params[0], False


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


def print_table(sort_key='Bayes BT'):
    global team_df

    if sort_key in ['Avg Points Allowed', 'Points Allowed Coef', 'Adjusted Points Allowed',
                    'YPG Allowed', 'Yards Allowed Coef', 'Adjusted Yards Allowed']:
        ascending_order = True
    else:
        ascending_order = False
    team_df = team_df.sort_values(by=sort_key, kind='mergesort', ascending=ascending_order)
    columns = ['Rank', 'Team', 'Record', 'Grade', 'Projected Record', 'Projected Wins 95% CI',
               'Adj Runs', 'Adj Runs Allowed', 'Adj Run Diff']

    table = PrettyTable(columns)
    table.float_format = '0.3'

    points_var = statistics.variance(team_df['Points Coef'])
    points_allowed_var = statistics.variance(team_df['Points Allowed Coef'])
    points_diff_var = statistics.variance(team_df['Adjusted Point Diff'])

    stop = '\033[0m'

    for index, row in team_df.iterrows():
        table_row = list()

        wins = row['Wins']
        losses = row['Losses']
        record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]])

        rank = team_df.index.get_loc(index) + 1

        points_pct = .05
        points_color = get_color(row['Points Coef'], points_var, alpha=points_pct, invert=False)
        points_allowed_color = get_color(row['Points Allowed Coef'], points_allowed_var, alpha=points_pct, invert=True)
        points_diff_color = get_color(row['Adjusted Point Diff'], points_diff_var, alpha=points_pct, invert=False)

        table_row.append(rank)
        table_row.append(index)
        table_row.append(record)

        bt_color = get_color(row['BT'], row['BT Var'])
        table_row.append(bt_color + f"{row['BT Pct'] * 100:.1f}".rjust(5) + stop)

        projected_wins = round(row['Projected Wins'])
        projected_losses = 162 - projected_wins
        proj_record = ' - '.join([str(int(val)).rjust(3) for val in [projected_wins, projected_losses]])
        table_row.append(proj_record)

        projected_wins_lower = round(row['Projected Wins Lower'])
        projected_wins_upper = round(row['Projected Wins Upper'])
        proj_95 = '(' + (str(projected_wins_lower) + ',').ljust(5) + str(projected_wins_upper).rjust(1) + ')'
        table_row.append(proj_95)

        table_row.append(points_color + str(round(row['Adjusted Points'], 1)) + stop)
        table_row.append(points_allowed_color + str(round(row['Adjusted Points Allowed'], 1)) + stop)
        table_row.append(points_diff_color + str(round(row['Adjusted Point Diff'], 1)).rjust(5) + stop)

        table.add_row(table_row)

    # Print the table
    print('Rankings')
    print(table)
    print()


def is_playoff_team(team_name):
    df = team_df.copy()

    df['Projected Wins'] = pd.to_numeric(df['Projected Wins'])

    division_leaders = df.groupby(by='Division').idxmax(numeric_only=True)['Projected Wins']
    division_leaders = set(division_leaders)

    if team_name in division_leaders:
        return True

    non_division_leaders = set(df.index) - division_leaders
    df = df.loc[list(non_division_leaders)]

    df = df.sort_values(by='Projected Wins', ascending=False)

    wild_card_teams = df.groupby('League').head(3).index
    wild_card_teams = set(wild_card_teams)

    return team_name in wild_card_teams


def print_projected_wins():
    global team_df

    team_df = team_df.sort_values(by='Bayes BT', kind='mergesort', ascending=False)
    columns = ['Rank', 'Team', 'Record', 'Grade', 'BT Projection', 'Bayes BT Projection',
               'Pythagorean Projection', 'Win % Projection']

    table = PrettyTable(columns)
    table.float_format = '0.3'

    points_var = statistics.variance(team_df['Points Coef'])
    points_allowed_var = statistics.variance(team_df['Points Allowed Coef'])
    points_diff_var = statistics.variance(team_df['Adjusted Point Diff'])

    stop = '\033[0m'

    for index, row in team_df.iterrows():
        table_row = list()

        wins = row['Wins']
        losses = row['Losses']
        record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]])

        rank = team_df.index.get_loc(index) + 1

        points_pct = .05
        points_color = get_color(row['Points Coef'], points_var, alpha=points_pct, invert=False)
        points_allowed_color = get_color(row['Points Allowed Coef'], points_allowed_var, alpha=points_pct, invert=True)
        points_diff_color = get_color(row['Adjusted Point Diff'], points_diff_var, alpha=points_pct, invert=False)

        rank = team_df.index.get_loc(index) + 1
        table_row.append(rank)

        table_row.append(index)

        wins = row['Wins']
        losses = row['Losses']
        record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]])
        table_row.append(record)

        bt_color = get_color(row['BT'], row['BT Var'])
        table_row.append(bt_color + f"{row['BT Pct'] * 100:.1f}".rjust(5) + stop)

        projected_wins = round(row['Projected Wins'])
        projected_losses = 162 - projected_wins
        proj_record = ' - '.join([str(int(val)).rjust(3) for val in [projected_wins, projected_losses]])
        table_row.append(proj_record)

        projected_wins_lower = round(row['Projected Wins Lower'])
        projected_wins_upper = round(row['Projected Wins Upper'])
        proj_95 = '(' + (str(projected_wins_lower) + ',').ljust(5) + str(projected_wins_upper).rjust(1) + ')'
        table_row.append(proj_95)

        table_row.append(points_color + str(round(row['Adjusted Points'], 1)) + stop)
        table_row.append(points_allowed_color + str(round(row['Adjusted Points Allowed'], 1)) + stop)
        table_row.append(points_diff_color + str(round(row['Adjusted Point Diff'], 1)).rjust(5) + stop)

        table.add_row(table_row)

    # Print the table
    print('Rankings')
    print(table)
    print()


def show_off_def():
    warnings.filterwarnings("ignore")

    sns.set(style="ticks")

    # Format and title the graph
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_title('')
    ax.set_xlabel('Adjusted Runs')
    ax.set_ylabel('Adjusted Runs Allowed')
    ax.set_facecolor('#FAFAFA')

    images = {team: PIL.Image.open('Projects/play/sports/mlb_logos/' + team + '.png')
              for team, row in team_df.iterrows()}

    intercept = team_df['Points Intercept'].mean()

    x_margin = .5
    y_margin = 1
    min_x = math.exp(intercept + team_df['Points Coef'].min()) - x_margin
    max_x = math.exp(intercept + team_df['Points Coef'].max()) + x_margin

    min_y = math.exp(intercept + team_df['Points Allowed Coef'].min()) - y_margin
    max_y = math.exp(intercept + team_df['Points Allowed Coef'].max()) + y_margin

    ax = plt.gca()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)
    ax.set_aspect(aspect=0.3, adjustable='datalim')

    for team in team_df.index:
        xa = math.exp(intercept + team_df.at[team, 'Points Coef'])
        ya = math.exp(intercept + team_df.at[team, 'Points Allowed Coef'])

        offset = .15
        ax.imshow(images.get(team), extent=(xa - offset, xa + offset, ya + offset, ya - offset), alpha=.8)

    vert_mean = team_df['Adjusted Points'].mean()
    horiz_mean = team_df['Adjusted Points Allowed'].mean()
    plt.axvline(x=vert_mean, color='r', linestyle='--', alpha=.5)
    plt.axhline(y=horiz_mean, color='r', linestyle='--', alpha=.5)

    step = 2
    offset_dist = step * math.sqrt(2)
    offsets = set(np.arange(0, 30, offset_dist))
    offsets = offsets.union({-offset for offset in offsets})

    for offset in [horiz_mean + offset for offset in offsets]:
        plt.axline(xy1=(vert_mean, offset), slope=1, alpha=.1)

    # Show the graph
    plt.show()


def show_graph():
    warnings.filterwarnings("ignore")

    sns.set(style="ticks")

    nfl = graph.copy()

    # Format and title the graph
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_aspect('auto')
    ax.set_title('')
    ax.set_facecolor('#FAFAFA')

    # Get the Pagerank of each node
    bts = {team: row['Bayes BT'] for team, row in team_df.iterrows()}
    max_bt = team_df['Bayes BT'].max()
    min_bt = team_df['Bayes BT'].min()
    bt_dev = statistics.stdev(team_df['Bayes BT'])
    subset = {team: np.digitize(row['Bayes BT'], np.arange(min_bt, max_bt, bt_dev / 2)) for team, row in team_df.iterrows()}

    nx.set_node_attributes(nfl, bts, 'BT')
    nx.set_node_attributes(nfl, subset, 'subset')

    images = {team: PIL.Image.open('Projects/play/sports/mlb_logos/' + team + '.png')
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


def plot_projected_wins(absolute_range=False):
    df = team_df.copy()
    df = df.sort_values(by='Projected Wins', ascending=True)

    teams = list(df.index)

    min_x = 0 if absolute_range else min(df['Projected Wins Lower']) - 5
    max_x = 162 if absolute_range else max(df['Projected Wins Upper']) + 5
    x_range = max_x - min_x

    p = figure(y_range=teams,
               width=1000,
               height=1700,
               x_range=(min_x, max_x),
               toolbar_location=None)

    source_data = dict(base=teams, upper=list(df['Projected Wins Upper']), lower=list(df['Projected Wins Lower']))
    source = ColumnDataSource(data=source_data)

    error = Whisker(base="base", upper='upper', lower='lower', source=source,
                    dimension="width", line_width=1)
    error.upper_head.size = 20
    error.lower_head.size = 20
    p.add_layout(error)

    images = {team: 'Projects/play/sports/mlb_logos/' + team + '.png' for team in teams}
    image_source = {'Team': list(images.keys()),
                    'Image': list(images.values()),
                    'Proj': [team_df.at[team, 'Projected Wins'] for team in images.keys()],
                    'Alpha': [1.0 if is_playoff_team(team) else 0.5 for team in images.keys()]}
    image_source = ColumnDataSource(data=image_source)
    height = .75
    width = height * 9.5 * (x_range / 162)
    p.image_url(url='Image', x='Proj', y='Team', global_alpha='Alpha',
                w=width, h=height, anchor='center', source=image_source)

    p.xaxis.axis_label = 'Projected Wins'

    show(p)


def plot_projected_wins_div(absolute_range=False):
    df = team_df.copy()
    df = df.sort_values(by='Projected Wins', ascending=True)

    min_x = 0 if absolute_range else min(df['Projected Wins Lower']) - 5
    max_x = 162 if absolute_range else max(df['Projected Wins Upper']) + 5
    x_range = max_x - min_x

    divisions = {'AL East': ['Orioles', 'Red Sox', 'Yankees', 'Rays', 'Blue Jays'],
                 'AL Central': ['White Sox', 'Guardians', 'Tigers', 'Royals', 'Twins'],
                 'AL West': ['Astros', 'Angels', 'Athletics', 'Mariners', 'Rangers'],
                 'NL East': ['Braves', 'Marlins', 'Mets', 'Phillies', 'Nationals'],
                 'NL Central': ['Cubs', 'Reds', 'Brewers', 'Pirates', 'Cardinals'],
                 'NL West': ['Diamondbacks', 'Rockies', 'Dodgers', 'Padres', 'Giants']}

    fig_map = dict()
    for division, teams in divisions.items():
        div_df = df.loc[teams]
        div_df = div_df.sort_values(by='Projected Wins', ascending=True)
        p = figure(y_range=list(div_df.index),
                   width=267,
                   height=200,
                   x_range=(min_x, max_x),
                   toolbar_location=None)

        source_data = dict(base=div_df.index,
                           upper=list(div_df['Projected Wins Upper']),
                           lower=list(div_df['Projected Wins Lower']))
        source = ColumnDataSource(data=source_data)

        error = Whisker(base="base", upper='upper', lower='lower', source=source,
                        dimension="width", line_width=1)
        error.upper_head.size = 20
        error.lower_head.size = 20
        p.add_layout(error)

        images = {team: 'Projects/play/sports/mlb_logos/' + team + '.png' for team in div_df.index}
        image_source = {'Team': list(images.keys()),
                        'Image': list(images.values()),
                        'Proj': [div_df.at[team, 'Projected Wins'] for team in images.keys()],
                        'Alpha': [1.0 if is_playoff_team(team) else 0.5 for team in images.keys()]}
        image_source = ColumnDataSource(data=image_source)
        height = .75
        width = height * 15 * (x_range / 162)
        p.image_url(url='Image', x='Proj', y='Team', global_alpha='Alpha',
                    w=width, h=height, anchor='center', source=image_source)

        p.xaxis.axis_label = division
        fig_map[division] = p

    grid = gridplot([[fig_map.get('AL West'), fig_map.get('AL Central'), fig_map.get('AL East')],
                     [fig_map.get('NL West'), fig_map.get('NL Central'), fig_map.get('NL East')]],
                    width=800, height=400)
    show(grid)


def plot_projected_wins_div2():
    df = team_df.copy()

    min_y = min(df['Projected Wins Lower']) - 5
    max_y = max(df['Projected Wins Upper']) + 5

    fig, ((al_west_ax, al_cent_ax, al_east_ax),
          (nl_west_ax, nl_cent_ax, nl_east_ax)) = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    divisions = {'AL East': ['Orioles', 'Red Sox', 'Yankees', 'Rays', 'Blue Jays'],
                 'AL Central': ['White Sox', 'Guardians', 'Tigers', 'Royals', 'Twins'],
                 'AL West': ['Astros', 'Angels', 'Athletics', 'Mariners', 'Rangers'],
                 'NL East': ['Braves', 'Marlins', 'Mets', 'Phillies', 'Nationals'],
                 'NL Central': ['Cubs', 'Reds', 'Brewers', 'Pirates', 'Cardinals'],
                 'NL West': ['Diamondbacks', 'Rockies', 'Dodgers', 'Padres', 'Giants']}

    div_axes = {'AL East': al_east_ax,
                'AL Central': al_cent_ax,
                'AL West': al_west_ax,
                'NL East': nl_east_ax,
                'NL Central': nl_cent_ax,
                'NL West': nl_west_ax}

    for div, div_ax in div_axes.items():
        div_ax.set_xlabel(div)
        div_ax.xaxis.set_label_position('top')
        div_ax.set_ylim(min_y, max_y)

        div_teams = divisions.get(div)
        relevant_df = df.loc[div_teams]
        relevant_df = relevant_df.sort_values(by='Projected Wins', ascending=False)
        sample_data = {team: norm(loc=df.at[team, 'Projected Wins'],
                                  scale=df.at[team, 'Projected Win SD']).rvs(size=10_000) for team in relevant_df.index}
        div_ax.boxplot(sample_data.values(), sym='')
        div_ax.set_xticklabels(sample_data.keys())

    plt.show()

