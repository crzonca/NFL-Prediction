import itertools
import math
import statistics
import warnings

import PIL
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import uuid
from prettytable import PrettyTable
from scipy.optimize import minimize
from scipy.stats import rankdata, kendalltau
from bokeh.models import HoverTool
from scipy.stats import skellam, poisson

from Projects.nfl.NFL_Prediction.Pred import league_structure
from Projects.nfl.NFL_Prediction.Pred.betting import Bettor
from Projects.nfl.NFL_Prediction.Pred.helper import Helper
from Projects.nfl.NFL_Prediction.Pred.playoff_chances import PlayoffPredictor


class LeagueEvaluator:
    def __init__(self, team_df, individual_df, graph):
        self.team_df = team_df
        self.individual_df = individual_df
        self.graph = graph

    def get_preseason_bts(self, use_mse=True, use_persisted=True):
        path = 'Projects/nfl/NFL_Prediction/Pred/resources/preseason_bts.csv'
        if use_persisted:
            pre_bt_df = pd.read_csv(path)
            team_bts = {row['Team']: row['BT'] for index, row in pre_bt_df.iterrows()}
            return team_bts
        win_totals = {'49ers': 11.5,
                      'Chiefs': 11.5,
                      'Eagles': 11.5,
                      'Ravens': 10.5,
                      'Lions': 10.5,
                      'Bills': 10.5,
                      'Bengals': 10.5,
                      'Cowboys': 9.5,
                      'Dolphins': 9.5,
                      'Jets': 9.5,
                      'Falcons': 9.5,
                      'Texans': 9.5,
                      'Packers': 9.5,
                      'Bears': 8.5,
                      'Colts': 8.5,
                      'Browns': 8.5,
                      'Rams': 8.5,
                      'Steelers': 8.5,
                      'Jaguars': 8.5,
                      'Chargers': 8.5,
                      'Cardinals': 7.5,
                      'Buccaneers': 7.5,
                      'Seahawks': 7.5,
                      'Saints': 7.5,
                      'Vikings': 7.5,
                      'Giants': 6.5,
                      'Raiders': 6.5,
                      'Commanders': 6.5,
                      'Titans': 6.5,
                      'Panthers': 5.5,
                      'Broncos': 5.5,
                      'Patriots': 4.5}

        schedule = league_structure.load_schedule()

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

        df['Rank'] = range(1, 33)
        df = df.set_index('Rank', drop=True)
        df = df[['Team', 'BT', 'Odds Projection', 'BT Projection']]
        df['BT Projection'] = df['BT Projection'].round(1)
        df.to_csv(path, index=False)

        return team_bts

    def parity_clock(self, verbose=False):
        # def rotate(l, n):
        #     return l[n:] + l[:n]
        #
        # cycles = {str(uuid.uuid4()): simple_cycle for simple_cycle in nx.simple_cycles(self.graph)}
        # if not cycles:
        #     return
        # cycle_lengths = {guid: len(simple_cycle) for guid, simple_cycle in cycles.items()}
        # max_cycle_len = max(cycle_lengths.values())
        # cycle_bts = {guid: [self.team_df.at[team, 'Bayes BT'] for team in simple_cycle]
        #              for guid, simple_cycle in cycles.items() if cycle_lengths.get(guid) == max_cycle_len}
        # cycle_bt_ranks = {guid: rankdata(simple_cycle, method='max') for guid, simple_cycle in cycle_bts.items()}
        # cycle_bt_ranks = {guid: rotate(list(simple_cycle), list(simple_cycle).index(1))
        #                   for guid, simple_cycle in cycle_bt_ranks.items()}
        # cycle_taus = {guid: abs(kendalltau(range(1, max_cycle_len + 1), simple_cycle).statistic)
        #               for guid, simple_cycle in cycle_bt_ranks.items()}
        # cycle_taus = {guid: tau for guid, tau in sorted(cycle_taus.items(), key=lambda t: t[1], reverse=True)}
        # cycle_guid = list(cycle_taus.items())[0][0]
        # cycle = cycles.get(cycle_guid)

        # Iterate over all cycles and find the one with the longest length
        longest_cycle_length = 0
        cycle = None
        for simple_cycle in nx.simple_cycles(self.graph):
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

            parity_graph = nx.DiGraph()
            bts = {team: row['Bayes BT'] for team, row in self.team_df.iterrows() if team in cycle}
            images = {team: PIL.Image.open('Projects/nfl/NFL_Prediction/Pred/resources/logos/' + team + '.png')
                      for team, row in self.team_df.iterrows() if team in cycle}
            for team in cycle:
                parity_graph.add_node(team)

            edge_list = list(itertools.pairwise(cycle))
            parity_graph.add_edges_from(edge_list)

            nx.set_node_attributes(parity_graph, bts, 'Bayes BT')
            nx.set_node_attributes(parity_graph, images, 'Image')

            pos = nx.circular_layout(parity_graph, scale=1)

            # Format and title the graph
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.set_aspect('auto')
            ax.set_title('Parity Clock')
            ax.set_facecolor('#FAFAFA')

            # Draw the edges in the graph
            for source, target in edge_list:
                target_bt = bts.get(target)
                target_margin = math.exp(target_bt) * 18

                nx.draw_networkx_edges(parity_graph,
                                       pos,
                                       edgelist=[(source, target)],
                                       width=1,
                                       alpha=0.1,
                                       edge_color='black',
                                       connectionstyle="Arc3, rad=0.05",
                                       arrowsize=10,
                                       min_target_margin=target_margin)

            # Select the size of the image (relative to the X axis)
            icon_size = {team: (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025 * math.exp(bt) for team, bt in bts.items()}
            icon_size['Bears'] = icon_size.get('Bears') * .8
            icon_center = {team: size / 2.0 for team, size in icon_size.items()}

            for n in parity_graph.nodes:
                xa, ya = fig.transFigure.inverted().transform(ax.transData.transform(pos[n]))
                a = plt.axes([xa - icon_center.get(n), ya - icon_center.get(n), icon_size.get(n), icon_size.get(n)])
                a.set_aspect('auto')
                a.imshow(parity_graph.nodes[n]['Image'], alpha=1.0)
                a.axis("off")

            # Show the graph
            plt.show()

            if verbose:
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
                missing = set(self.team_df.index) - {team.strip() for team in cycle}
                print(' | '.join(missing))
                print()

    def show_off_def(self):
        # playoff_teams = ['49ers', 'Chiefs']
        warnings.filterwarnings("ignore")

        sns.set(style="ticks")

        # Format and title the graph
        fig, ax = plt.subplots(figsize=(20, 10))

        ax.set_title('')
        ax.set_xlabel('Adjusted Points For')
        ax.set_ylabel('Adjusted Points Against')
        ax.set_facecolor('#FAFAFA')

        images = {team: PIL.Image.open('Projects/nfl/NFL_Prediction/Pred/resources/logos/' + team + '.png')
                  for team, row in self.team_df.iterrows()}

        margin = 1
        min_x = self.team_df['Bayes Adjusted Points'].min() - margin
        max_x = self.team_df['Bayes Adjusted Points'].max() + margin

        min_y = self.team_df['Bayes Adjusted Points Allowed'].min() - margin
        max_y = self.team_df['Bayes Adjusted Points Allowed'].max() + margin

        ax = plt.gca()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(max_y, min_y)
        ax.set_aspect(aspect=0.3, adjustable='datalim')

        for team in self.team_df.index:
            xa = self.team_df.at[team, 'Bayes Adjusted Points']
            ya = self.team_df.at[team, 'Bayes Adjusted Points Allowed']

            offset = .4 if team == 'Bears' else .5
            img_alpha = .8  # if team in playoff_teams else .2
            ax.imshow(images.get(team), extent=(xa - offset, xa + offset, ya + offset, ya - offset), alpha=img_alpha)

        vert_mean = self.team_df['Bayes Adjusted Points'].mean()
        horiz_mean = self.team_df['Bayes Adjusted Points Allowed'].mean()
        plt.axvline(x=vert_mean, color='r', linestyle='--', alpha=.5)
        plt.axhline(y=horiz_mean, color='r', linestyle='--', alpha=.5)

        offset_dist = 3 * math.sqrt(2)
        offsets = set(np.arange(0, 75, offset_dist))
        offsets = offsets.union({-offset for offset in offsets})

        for offset in [horiz_mean + offset for offset in offsets]:
            plt.axline(xy1=(vert_mean, offset), slope=1, alpha=.1)

        plt.savefig('D:\\Colin\\Documents\\Programming\\Python\\PythonProjects\\Projects\\nfl\\'
                    'NFL_Prediction\\Pred\\resources\\OffenseDefense.png', dpi=300)

        # Show the graph
        plt.show()

    def show_off_def_interactive(self, use_screen=True):
        images = {team: 'Projects/nfl/NFL_Prediction/Pred/resources/logos/' + team + '.png'
                  for team, row in self.team_df.iterrows()}

        margin = 1
        min_x = self.team_df['Bayes Adjusted Points'].min() - margin
        max_x = self.team_df['Bayes Adjusted Points'].max() + margin
        mid_x = (min_x + max_x) / 2

        min_y = self.team_df['Bayes Adjusted Points Allowed'].min() - margin
        max_y = self.team_df['Bayes Adjusted Points Allowed'].max() + margin

        from bokeh.plotting import figure, show, output_file

        output_file('image.html')

        fig_width = 2400
        fig_height = 900

        y_range = max_y - min_y
        x_range = max_x - min_x
        angle_aspect = (x_range / (fig_width - 33)) / (y_range / (fig_height - 33))

        p = figure(width=fig_width,
                   height=fig_height,
                   x_range=(min_x, max_x),
                   y_range=(max_y, min_y))
        if use_screen:
            height = width = 75
            image_unit = 'screen'
        else:
            height = y_range / 15
            fig_aspect = fig_height / (fig_width - 165)
            width = x_range / 15 * fig_aspect
            image_unit = 'data'
        for team in self.team_df.index:
            xa = self.team_df.at[team, 'Bayes Adjusted Points']
            ya = self.team_df.at[team, 'Bayes Adjusted Points Allowed']
            if team == 'Bears':
                height = height * .8
                width = width * .8
            if team == 'Ravens':
                height = height * 1.4
                width = width * 1.4
            if team == 'Patriots':
                height = height * 1.1
                width = width * 1.1
            p.image_url(url=[images.get(team)],
                        x=xa, y=ya,
                        w=width, h=height,
                        name=team,
                        anchor='center',
                        h_units=image_unit, w_units=image_unit,
                        alpha=.7)

        vert_mean = self.team_df['Bayes Adjusted Points'].mean()
        horiz_mean = self.team_df['Bayes Adjusted Points Allowed'].mean()
        for ray_angle in range(0, 360, 90):
            p.ray(x=[vert_mean],
                  y=[horiz_mean],
                  length=0,
                  angle=ray_angle,
                  angle_units='deg',
                  line_width=1,
                  line_color='red',
                  line_dash='dashed',
                  line_alpha=.5)

        offset_dist = 3 * math.sqrt(2)
        offsets = set(np.arange(0, 75, offset_dist))
        offsets = offsets.union({-offset for offset in offsets})

        for offset in [horiz_mean + offset for offset in offsets]:
            for rotation in range(2):
                angle = -math.atan(angle_aspect) + math.pi * rotation
                p.ray(x=[vert_mean],
                      y=[offset],
                      length=0,
                      angle=angle,
                      angle_units='rad',
                      line_width=1,
                      line_color='black',
                      line_alpha=.1)

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.xaxis.axis_label = 'Adjusted Points For'
        p.yaxis.axis_label = 'Adjusted Points Against'
        show(p)

    def show_graph(self, divisional_edges_only=True):
        warnings.filterwarnings("ignore")

        sns.set(style="ticks")

        nfl = self.graph.copy()

        # Format and title the graph
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_aspect('auto')
        ax.set_title('')
        ax.set_facecolor('#FAFAFA')

        # Get the Pagerank of each node
        bts = {team: row['Bayes BT'] for team, row in self.team_df.iterrows()}
        primary = {team: row['Primary Colors'] for team, row in self.team_df.iterrows()}
        secondary = {team: row['Secondary Colors'] for team, row in self.team_df.iterrows()}
        max_bt = self.team_df['Bayes BT'].max()
        min_bt = self.team_df['Bayes BT'].min()
        bt_dev = statistics.stdev(self.team_df['Bayes BT'])
        subset = {team: row['Tier'] for team, row in self.team_df.iterrows()}

        nx.set_node_attributes(nfl, bts, 'Bayes BT')
        nx.set_node_attributes(nfl, primary, 'Primary')
        nx.set_node_attributes(nfl, secondary, 'Secondary')
        nx.set_node_attributes(nfl, subset, 'subset')

        images = {team: PIL.Image.open('Projects/nfl/NFL_Prediction/Pred/resources/logos/' + team + '.png')
                  for team, row in self.team_df.iterrows()}
        nx.set_node_attributes(nfl, images, 'image')

        pos = nx.multipartite_layout(nfl, align='horizontal')

        if divisional_edges_only:
            edge_list = [(t1, t2) for t1, t2 in nfl.edges() if
                         league_structure.get_division(t1) == league_structure.get_division(t2)]
        else:
            edge_list = nfl.edges()

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
        icon_size['Bears'] = icon_size.get('Bears') * .8
        icon_center = {team: size / 2.0 for team, size in icon_size.items()}

        for n in nfl.nodes:
            xa, ya = fig.transFigure.inverted().transform(ax.transData.transform(pos[n]))
            a = plt.axes([xa - icon_center.get(n), ya - icon_center.get(n), icon_size.get(n), icon_size.get(n)])
            a.set_aspect('auto')
            a.imshow(nfl.nodes[n]['image'], alpha=1.0)
            a.axis("off")

        plt.savefig('D:\\Colin\\Documents\\Programming\\Python\\PythonProjects\\Projects\\nfl\\'
                    'NFL_Prediction\\Pred\\resources\\LeagueGraph.png', dpi=300)

        # Show the graph
        plt.show()

    def surprises(self):
        last_year_wins = {'Ravens': 13,
                          'Bills': 11,
                          'Chiefs': 11,
                          'Dolphins': 11,
                          'Browns': 11,
                          'Steelers': 10,
                          'Texans': 10,
                          'Bengals': 9,
                          'Jaguars': 9,
                          'Colts': 9,
                          'Raiders': 8,
                          'Broncos': 8,
                          'Jets': 7,
                          'Titans': 6,
                          'Chargers': 5,
                          'Patriots': 4,

                          '49ers': 12,
                          'Cowboys': 12,
                          'Lions': 12,
                          'Eagles': 11,
                          'Rams': 10,
                          'Seahawks': 9,
                          'Buccaneers': 9,
                          'Packers': 9,
                          'Saints': 9,
                          'Vikings': 7,
                          'Bears': 7,
                          'Falcons': 7,
                          'Giants': 6,
                          'Commanders': 4,
                          'Cardinals': 4,
                          'Panthers': 2}

        pp = PlayoffPredictor(team_df=self.team_df, graph=self.graph)

        surprise_dict = dict()
        for team in self.team_df.index:
            team_wins = last_year_wins.get(team)
            team_wp = team_wins / 17

            proj_wins, proj_losses, proj_ties = pp.get_proj_record(team)
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

    def get_schedule_difficulties(self):
        schedule = league_structure.load_schedule()
        weeks = schedule.get('weeks')

        team_schedule = dict()
        for team in self.team_df.index:
            opponents = list()
            for week in weeks:
                for game in week:
                    if game.get('away') == team:
                        opponents.append(game.get('home'))

                    if game.get('home') == team:
                        opponents.append(game.get('away'))

            opponent_bts = [self.team_df.at[opponent, 'Bayes BT'] for opponent in opponents]
            team_schedule[team] = opponent_bts

        team_opponent_bts = {team: statistics.mean(bts) for team, bts in team_schedule.items()}
        team_opponent_bts = {k: v for k, v in sorted(team_opponent_bts.items(), key=lambda item: item[1], reverse=True)}

        team_win_chances = {team: [1 / (1 + math.exp(bt)) for bt in bts] for team, bts in team_schedule.items()}
        team_average_wins = {team: round(sum(chances)) for team, chances in team_win_chances.items()}

        self.team_df = self.team_df.sort_values(by='Bayes BT', kind='mergesort', ascending=False)
        table = PrettyTable(['Rank', 'Name', 'Record', 'Avg. Opponent BT', 'Games Above Average'])
        table.float_format = '0.3'

        green = '\033[32m'
        red = '\033[31m'
        stop = '\033[0m'

        pp = PlayoffPredictor(team_df=self.team_df, graph=self.graph)
        for index, team_info in enumerate(team_opponent_bts.items()):
            team, avg_opp_bt = team_info
            table_row = list()

            wins = self.team_df.at[team, 'Wins']
            losses = self.team_df.at[team, 'Losses']
            ties = self.team_df.at[team, 'Ties']

            record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]]) + ' - ' + str(int(ties))

            average_wins = team_average_wins.get(team)
            expected_wins, expected_losses, expected_ties = pp.get_proj_record(team)
            games_above = (expected_wins - average_wins) / 2
            color = green if games_above > 1 else red if games_above < -1 else ''
            games_above = color + str(games_above).rjust(4) + stop

            avg_record = ' - '.join([str(val).rjust(2) for val in [average_wins, 17 - average_wins, 0]])

            rank = index + 1

            table_row.append(rank)
            table_row.append(team)
            table_row.append(record)
            # table_row.append(team_df.at[team, 'Bayes BT'])
            table_row.append(avg_opp_bt)
            table_row.append(games_above)

            table.add_row(table_row)

        # Print the table
        print('Team Schedule Difficulties')
        print(table)
        print()

    def plot_matchup(self, team1, team2, team1_chance, team1_spread):
        # Setup
        plt.rcParams['font.family'] = 'monospace'

        is_favorite = team1_spread < 0
        bets = Bettor(self.team_df, self.individual_df, self.graph)
        if is_favorite:
            cover_chance, push_chance, fail_chance = bets.get_spread_chance(team1, team2, team1_spread)
        else:
            t2_cover_chance, push_chance, t2_fail_chance = bets.get_spread_chance(team2, team1, -team1_spread)
            cover_chance = 1 - push_chance - t2_fail_chance
            fail_chance = 1 - push_chance - t2_cover_chance

        team1_spread = math.floor(team1_spread)
        team1_spread = -team1_spread

        team1_logo = PIL.Image.open('Projects/nfl/NFL_Prediction/Pred/resources/logos/' + team1 + '.png')
        team2_logo = PIL.Image.open('Projects/nfl/NFL_Prediction/Pred/resources/logos/' + team2 + '.png')

        ordinal_suffix_map = {1: 'st',
                              2: 'nd',
                              3: 'rd',
                              21: 'st',
                              22: 'nd',
                              31: 'st',
                              32: 'nd'}

        # Skellam PMF
        intercept = self.team_df.at[team1, 'Points Intercept']
        team1_off_coef = self.team_df.at[team1, 'Bayes Points Coef']
        team2_off_coef = self.team_df.at[team2, 'Bayes Points Coef']
        team1_def_coef = self.team_df.at[team1, 'Bayes Points Allowed Coef']
        team2_def_coef = self.team_df.at[team2, 'Bayes Points Allowed Coef']

        team1_lambda = math.exp(intercept + team1_off_coef + team2_def_coef)
        team2_lambda = math.exp(intercept + team2_off_coef + team1_def_coef)

        helper = Helper(self.team_df, self.individual_df, self.graph)
        average_drives = helper.predict_drives(team1, team2)
        team1_lambda = team1_lambda * average_drives
        team2_lambda = team2_lambda * average_drives

        skel = skellam(team1_lambda, team2_lambda)
        win_chance = skel.sf(0)
        tie_chance = skel.pmf(0)
        win_chance = win_chance / (1 - tie_chance)
        loss_chance = 1 - win_chance

        min_end = skel.ppf(.001)
        max_end = skel.ppf(.999)

        x_range = np.arange(min_end, max_end, 1.0)
        pmf_range = [skel.pmf(x) for x in x_range]

        fig, ax = plt.subplots(figsize=(20, 10))

        ax.set_xlabel(team1 + ' Margin of Victory')
        ax.set_ylabel('Chance')
        ax.set_facecolor('#FAFAFA')

        plt.plot(x_range, pmf_range)
        plt.ylim((0.0, 0.09))

        # Projected score
        common_team1_points = helper.get_common_score(team1_lambda)
        common_team2_points = helper.get_common_score(team2_lambda)
        is_ot = False
        if common_team1_points == common_team2_points:
            is_ot = True
            if team1_lambda >= team2_lambda:
                common_team1_points = common_team1_points + 3
            else:
                common_team2_points = common_team2_points + 3
        winner = team1 if common_team1_points > common_team2_points else team2
        winner_common_points = common_team1_points if common_team1_points > common_team2_points else common_team2_points
        loser_common_points = common_team2_points if common_team1_points > common_team2_points else common_team1_points

        end = '(OT)' if is_ot else ''
        expected_score = str(winner_common_points) + ' - ' + str(loser_common_points)
        title = 'Projected Score: ' + expected_score + ' ' + winner + ' ' + end
        ax.set_title(title.strip(), fontsize=35)

        # Fill PMF
        def add_pct_text(range_start, range_end, chance):
            range_size = range_end - range_start
            if (range_size <= 5 and chance < .15) or range_size <= 2:
                return
            pmfs = list()
            for x in range(int(range_start), int(range_end)):
                prob = skel.pmf(x)
                pmfs.append(prob)
            total = sum(pmfs)

            total_prob = 0
            for x in range(int(range_start), int(range_end)):
                prob = skel.pmf(x)
                total_prob = total_prob + prob
                if total_prob >= total / 2:
                    r1_center = x
                    break

            r1_height = skel.pmf(r1_center) / 2
            r1_size = r1_height * 2 / .06 * 40
            r1_size = max([r1_size, 10])
            r1_size = min([r1_size, 35])
            if 3 < range_size <= 5:
                r1_size = min([r1_size, 20])
            ax.annotate(f"{chance * 100:.1f}" + '%', xy=(r1_center, r1_height), ha='center', fontsize=r1_size)

        if is_favorite:
            r1 = np.arange(min_end, 1, 1.0)
            add_pct_text(min_end, 1, loss_chance)

            r2 = np.arange(0, team1_spread + 1, 1.0)
            add_pct_text(0, team1_spread + 1, fail_chance - loss_chance)

            r3 = np.arange(team1_spread, max_end, 1.0)
            add_pct_text(team1_spread, max_end, cover_chance)
        else:
            r1 = np.arange(min_end, team1_spread + 1, 1.0)
            add_pct_text(min_end, team1_spread + 1, cover_chance)  # TODO Verify chance

            r2 = np.arange(team1_spread, 1, 1.0)
            add_pct_text(team1_spread, 1, fail_chance)  # TODO Verify chance

            r3 = np.arange(0, max_end, 1.0)
            add_pct_text(0, max_end, 1 - loss_chance)  # TODO Verify chance

        plt.fill_between(x=r1,
                         y1=[0 for x in range(len(r1))],
                         y2=[skel.pmf(x) for x in r1],
                         color='r' if is_favorite else 'r',
                         alpha=.3)

        plt.fill_between(x=r2,
                         y1=[0 for x in range(len(r2))],
                         y2=[skel.pmf(x) for x in r2],
                         color='yellowgreen' if is_favorite else 'g',
                         alpha=.3)

        plt.fill_between(x=r3,
                         y1=[0 for x in range(len(r3))],
                         y2=[skel.pmf(x) for x in r3],
                         color='g' if is_favorite else 'darkgreen',
                         alpha=.3)

        # Mean margin of victory
        skel_mean = team1_lambda - team2_lambda
        ax.vlines(x=skel_mean,
                  ymin=0,
                  ymax=skel.pmf(math.floor(skel_mean)),
                  linestyles='dashed',
                  label='mean',
                  color='k',
                  alpha=.8)
        new_x_ticks = [tick for tick in list(ax.get_xticks()) if abs(tick - skel_mean) > 2]
        ax.set_xticks(new_x_ticks + [round(skel_mean, 1)])

        a = plt.axes([-.125, .69, .6, .16])
        a.set_aspect('auto')
        a.axis("off")
        a.imshow(team1_logo, origin='upper', extent=(0, 1, 0, 1), alpha=0.7)

        wins = self.team_df.at[team1, 'Wins']
        losses = self.team_df.at[team1, 'Losses']
        ties = self.team_df.at[team1, 'Ties']
        team1_record = ' - '.join([str(int(val)) for val in [wins, losses]]) + ' - ' + str(int(ties))

        pp = PlayoffPredictor(self.team_df, self.graph)
        proj_record = pp.get_proj_record(team1)
        ties = proj_record[-1]
        team1_proj_record = ' - '.join([str(val) for val in proj_record[:-1]]) + ' - ' + str(int(ties))

        team1_bt = self.team_df.at[team1, 'Bayes BT']
        team1_bt_pct = self.team_df.at[team1, 'BT Pct'] * 100
        team1_off = self.team_df.at[team1, 'Bayes Adjusted Points']
        team1_def = self.team_df.at[team1, 'Bayes Adjusted Points Allowed']

        team1_bt_index = list(self.team_df['Bayes BT']).index(team1_bt)
        team1_bt_rank = rankdata(self.team_df['Bayes BT'], method='max')[team1_bt_index]
        team1_bt_rank = 33 - team1_bt_rank

        team1_off_index = list(self.team_df['Bayes Adjusted Points']).index(team1_off)
        team1_off_rank = rankdata(self.team_df['Bayes Adjusted Points'], method='max')[team1_off_index]
        team1_off_rank = 33 - team1_off_rank

        team1_def_index = list(self.team_df['Bayes Adjusted Points Allowed']).index(team1_def)
        team1_def_rank = rankdata(self.team_df['Bayes Adjusted Points Allowed'], method='max')[team1_def_index]

        team1_stats = 'Rec: ' + team1_record + ' (Proj ' + team1_proj_record + ')\n' + \
                      'Ovr: ' + str(round(team1_bt_pct, 1)).rjust(4) + ' (' + str(
            team1_bt_rank) + ordinal_suffix_map.get(
            team1_bt_rank, 'th') + ')\n' + \
                      'Off: ' + str(round(team1_off, 1)).rjust(4) + ' (' + str(team1_off_rank) + ordinal_suffix_map.get(
            team1_off_rank, 'th') + ')\n' + \
                      'Def: ' + str(round(team1_def, 1)).rjust(4) + ' (' + str(team1_def_rank) + ordinal_suffix_map.get(
            team1_def_rank, 'th') + ')'

        plt.text(x=1.075, y=.2, s=team1_stats, fontsize=16)

        # Team 2 Logo and Stats
        a = plt.axes([-.125, .52, .6, .16])
        a.set_aspect('auto')
        a.axis("off")
        a.imshow(team2_logo, origin='upper', extent=(0, 1, 0, 1), alpha=0.7)

        wins = self.team_df.at[team2, 'Wins']
        losses = self.team_df.at[team2, 'Losses']
        ties = self.team_df.at[team2, 'Ties']
        team2_record = ' - '.join([str(int(val)) for val in [wins, losses]]) + ' - ' + str(int(ties))

        proj_record = pp.get_proj_record(team2)
        ties = proj_record[-1]
        team2_proj_record = ' - '.join([str(val) for val in proj_record[:-1]]) + ' - ' + str(int(ties))

        team2_bt = self.team_df.at[team2, 'Bayes BT']
        team2_bt_pct = self.team_df.at[team2, 'BT Pct'] * 100
        team2_off = self.team_df.at[team2, 'Bayes Adjusted Points']
        team2_def = self.team_df.at[team2, 'Bayes Adjusted Points Allowed']

        team2_bt_index = list(self.team_df['Bayes BT']).index(team2_bt)
        team2_bt_rank = rankdata(self.team_df['Bayes BT'], method='max')[team2_bt_index]
        team2_bt_rank = 33 - team2_bt_rank

        team2_off_index = list(self.team_df['Bayes Adjusted Points']).index(team2_off)
        team2_off_rank = rankdata(self.team_df['Bayes Adjusted Points'], method='max')[team2_off_index]
        team2_off_rank = 33 - team2_off_rank

        team2_def_index = list(self.team_df['Bayes Adjusted Points Allowed']).index(team2_def)
        team2_def_rank = rankdata(self.team_df['Bayes Adjusted Points Allowed'], method='max')[team2_def_index]

        team2_stats = 'Rec: ' + team2_record + ' (Proj ' + team2_proj_record + ')\n' + \
                      'Ovr: ' + str(round(team2_bt_pct, 1)).rjust(4) + ' (' + str(
            team2_bt_rank) + ordinal_suffix_map.get(
            team2_bt_rank, 'th') + ')\n' + \
                      'Off: ' + str(round(team2_off, 1)).rjust(4) + ' (' + str(team2_off_rank) + ordinal_suffix_map.get(
            team2_off_rank, 'th') + ')\n' + \
                      'Def: ' + str(round(team2_def, 1)).rjust(4) + ' (' + str(team2_def_rank) + ordinal_suffix_map.get(
            team2_def_rank, 'th') + ')'

        plt.text(x=1.075, y=.2, s=team2_stats, fontsize=16)

        # Pie charts
        team2_chance = 1 - team1_chance
        a = plt.axes([.73, .67, .14, .18])
        a.pie([team1_chance, team2_chance],
              labels=[team1, team2],
              colors=[self.team_df.at[team1, 'Primary Colors'], self.team_df.at[team2, 'Primary Colors']],
              autopct='%1.1f%%',
              startangle=90,
              wedgeprops={'alpha': 0.8})
        a.set_aspect('equal')
        a.set_title('Overall Chance', pad=-5)
        a.axis("off")

        team1_bt = self.team_df.at[team1, 'Bayes BT']
        team2_bt = self.team_df.at[team2, 'Bayes BT']

        team1_bt_chance = math.exp(team1_bt) / (math.exp(team1_bt) + math.exp(team2_bt))
        team2_bt_chance = 1 - team1_bt_chance

        a = plt.axes([.73, .5, .14, .18])
        a.pie([team1_bt_chance, team2_bt_chance],
              labels=[team1, team2],
              colors=[self.team_df.at[team1, 'Primary Colors'], self.team_df.at[team2, 'Primary Colors']],
              autopct='%1.1f%%',
              startangle=90,
              wedgeprops={'alpha': 0.8})
        a.set_aspect('equal')
        a.set_title('BT Chance', y=0, pad=-5)
        a.axis("off")

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))

        plt.show()

    def plot_score_heatmap(self, team1, team2):
        intercept = self.team_df.at[team1, 'Points Intercept']
        team1_off_coef = self.team_df.at[team1, 'Bayes Points Coef']
        team2_off_coef = self.team_df.at[team2, 'Bayes Points Coef']
        team1_def_coef = self.team_df.at[team1, 'Bayes Points Allowed Coef']
        team2_def_coef = self.team_df.at[team2, 'Bayes Points Allowed Coef']

        team1_lambda = math.exp(intercept + team1_off_coef + team2_def_coef)
        team2_lambda = math.exp(intercept + team2_off_coef + team1_def_coef)

        helper = Helper(self.team_df, self.individual_df, self.graph)
        average_drives = helper.predict_drives(team1, team2)
        team1_lambda = team1_lambda * average_drives
        team2_lambda = team2_lambda * average_drives

        team1_poisson = poisson(team1_lambda)
        team2_poisson = poisson(team2_lambda)

        rows = list()
        team1_max_score = int(team1_poisson.ppf(.999))
        team2_max_score = int(team2_poisson.ppf(.999))
        for team1_score in range(team1_max_score + 1):
            for team2_score in range(team2_max_score + 1):
                team1_chance = team1_poisson.pmf(team1_score)
                team2_chance = team2_poisson.pmf(team2_score)
                total_chance = team1_chance * team2_chance
                row_dict = {team1 + ' Score': team1_score,
                            team1 + ' Chance': team1_chance,
                            team2 + ' Score': team2_score,
                            team2 + ' Chance': team2_chance,
                            'Total Chance': total_chance}
                rows.append(row_dict)
        score_df = pd.DataFrame(rows)

        score_df[team1 + ' Common Score'] = score_df.apply(
            lambda r: league_structure.get_common_score(r[team1 + ' Score']), axis=1)
        score_df[team2 + ' Common Score'] = score_df.apply(
            lambda r: league_structure.get_common_score(r[team2 + ' Score']), axis=1)
        score_df = score_df.groupby(by=[team1 + ' Common Score', team2 + ' Common Score']).sum()
        for index, row in score_df.iterrows():
            team1_score, team2_score = index
            score_df.at[index, team1 + ' Score'] = team1_score
            score_df.at[index, team2 + ' Score'] = team2_score

        score_df = score_df.pivot(index=team1 + ' Score', columns=team2 + ' Score', values='Total Chance')

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(score_df, annot=True, fmt=".1%", ax=ax)
        fig.show()

    def plot_score_joint(self, team1, team2):
        intercept = self.team_df.at[team1, 'Points Intercept']
        team1_off_coef = self.team_df.at[team1, 'Bayes Points Coef']
        team2_off_coef = self.team_df.at[team2, 'Bayes Points Coef']
        team1_def_coef = self.team_df.at[team1, 'Bayes Points Allowed Coef']
        team2_def_coef = self.team_df.at[team2, 'Bayes Points Allowed Coef']

        team1_lambda = math.exp(intercept + team1_off_coef + team2_def_coef)
        team2_lambda = math.exp(intercept + team2_off_coef + team1_def_coef)

        helper = Helper(self.team_df, self.individual_df, self.graph)
        average_drives = helper.predict_drives(team1, team2)
        team1_lambda = team1_lambda * average_drives
        team2_lambda = team2_lambda * average_drives

        team1_poisson = poisson(team1_lambda)
        team2_poisson = poisson(team2_lambda)

        score_df = pd.DataFrame(columns=[team1 + ' Score', team2 + ' Score'])
        score_df[team1 + ' Score'] = [league_structure.get_common_score(score) for score in
                                      team1_poisson.rvs(size=10_000)]
        score_df[team2 + ' Score'] = [league_structure.get_common_score(score) for score in
                                      team2_poisson.rvs(size=10_000)]

        team1_max_score = int(team1_poisson.ppf(.999))
        team2_max_score = int(team2_poisson.ppf(.999))
        max_score = max([team1_max_score, team2_max_score])

        # TODO figure out how to display
        fig, ax = plt.subplots(figsize=(20, 10))
        g = sns.JointGrid(data=score_df, x=team1 + ' Score', y=team2 + ' Score', space=0)
        g.plot_joint(sns.kdeplot,
                     fill=True, clip=((0, max_score), (0, max_score)),
                     thresh=0, levels=100, cmap="rocket")
        g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)
        # g.savefig('temp.png')
