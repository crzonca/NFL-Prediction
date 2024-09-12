import itertools
import math
import pickle
import statistics
import warnings

import choix
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import poisson


class Helper:
    def __init__(self, team_df, individual_df, graph):
        self.team_df = team_df
        self.individual_df = individual_df
        self.graph = graph

    def predict_drives(self, team1, team2):
        average_drives = 12
        if team1 in self.individual_df['Team'] or team1 in self.individual_df['Opponent']:
            if team2 in self.individual_df['Team'] or team2 in self.individual_df['Opponent']:
                relevant_df = self.individual_df.loc[(self.individual_df['Team'] == team1) |
                                                     (self.individual_df['Opponent'] == team1) |
                                                     (self.individual_df['Team'] == team2) |
                                                     (self.individual_df['Opponent'] == team2)]
                if not relevant_df.empty:
                    average_drives = relevant_df['Drives'].mean()
        else:
            if not self.individual_df.empty:
                average_drives = self.individual_df['Drives'].mean()
        return average_drives

    def get_bayes_avg(self, prior_avg, prior_var, sample_avg, sample_var, n):
        k_0 = sample_var / prior_var
        if k_0 + n == 0:
            return prior_avg
        posterior_avg = ((k_0 / (k_0 + n)) * prior_avg) + ((n / (k_0 + n)) * sample_avg)
        return posterior_avg

    def get_bayes_avg_wins(self, game_df, team_name):
        matching_games = game_df.loc[game_df['Team'] == team_name]

        # prior_avg = 0.505220599260847
        prior_avg = 0.5
        prior_var = 0.0393327761147111

        wins = list(matching_games['Win'])
        if len(wins) < 2:
            return prior_avg

        win_pct = statistics.mean(wins)
        win_var = statistics.variance(wins)

        return self.get_bayes_avg(prior_avg, prior_var, win_pct, win_var, len(wins))

    def get_bayes_avg_points(self, game_df, team_name, allowed=False):
        matching_games = game_df.loc[game_df['Team'] == team_name]

        prior_avg = 22.139485653060646
        prior_var = 20.7125550580956

        points = list(matching_games['Points Allowed']) if allowed else list(matching_games['Points'])
        if len(points) < 2:
            return prior_avg

        avg_points = statistics.mean(points)
        points_var = statistics.variance(points)

        return self.get_bayes_avg(prior_avg, prior_var, avg_points, points_var, len(points))

    def get_color(self, value, variance=np.nan, alpha=.05, enabled=True, invert=False):
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

    def get_common_score(self, expected):
        expected_poisson = poisson(expected)

        empirical_score_df = pd.read_csv('Projects/nfl/NFL_Prediction/Pred/resources/empirical_scores.csv')

        empirical_score_df['poisson'] = empirical_score_df.apply(lambda r: expected_poisson.pmf(r['Score']), axis=1)
        empirical_score_df['posterior'] = empirical_score_df.apply(lambda r: r['poisson'] * r['Pct'], axis=1)

        common_score = empirical_score_df['posterior'].idxmax()
        return common_score

    def load_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_path = 'Projects/nfl/NFL_Prediction/Pred/resources/model.pkl'
            with open(model_path, 'rb') as f:
                clf = pickle.load(f)

                return clf

    def load_pt(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt_path = 'Projects/nfl/NFL_Prediction/Pred/resources/pt.pkl'
            with open(pt_path, 'rb') as f:
                pt = pickle.load(f)

                return pt

    def get_bradley_terry_from_graph(self):
        nodes = self.graph.nodes
        df = pd.DataFrame(nx.to_numpy_array(self.graph), columns=nodes)
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

        if not edges:
            coef_df = pd.DataFrame(columns=['BT', 'Var'], index=self.team_df.index)
            coef_df['BT'] = coef_df['BT'].fillna(0)
            coef_df['Var'] = coef_df['Var'].fillna(1)
            return coef_df

        coeffs, cov = choix.ep_pairwise(n_items=len(teams), data=edges, alpha=1)
        coeffs = pd.Series(coeffs)
        cov = pd.Series(cov.diagonal())
        coef_df = pd.DataFrame([coeffs, cov]).T
        coef_df.columns = ['BT', 'Var']
        coef_df.index = [index_to_teams.get(index) for index in coef_df.index]
        coef_df = coef_df.sort_values(by='BT', ascending=False)
        return coef_df
