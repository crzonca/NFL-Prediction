import os
import statistics

import matplotlib.pyplot as plt
import maya
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from scipy import stats
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import check_consistent_length

from Projects.nfl.NFL_Prediction import StatsHelper as Stats

game_data_dir = '..\\Projects\\nfl\\NFL_Prediction\\Game Data\\'
other_dir = '..\\Projects\\nfl\\NFL_Prediction\\Other\\'


def run_all():
    frames = get_all_data_frames()
    frames = add_point_diff_and_results(frames)
    frames = add_team_records(frames)
    # best_reg, best_k = evaluate_elo(frames)
    # frames = add_team_elos(frames, regression_factor=best_reg, k_factor=best_k)
    frames = add_team_elos(frames)
    frames = add_season_totals(frames)
    frames = add_season_averages(frames)
    frames = add_advanced_stats(frames)
    frames = add_stat_differences(frames)
    all_games = combine_frames(frames)

    # over_sampled = oversample_data()

    won_series = plot_corr()
    best_features = get_best_features()

    won_series = won_series.filter(best_features).sort_values(kind='quicksort', ascending=False)
    best_features = list(won_series.index)

    evaluate_model_parameters(best_features)
    voting_classifier = get_voting_classifier(best_features)
    evaluate_2018_season()


def get_all_data_frames():
    """Gets all of the csv files in the directory and converts them to pandas DataFrames."""
    frames = list()

    for file in os.listdir(game_data_dir):
        if (os.path.splitext(file)[1] == '.csv'
                and os.path.splitext(file)[0] != '20022018'
                and os.path.splitext(file)[0] != '20022017'):
            with open(game_data_dir + file, 'r') as season_csv:
                df = pd.read_csv(season_csv, encoding='utf-8')
                df = df.rename(index=str, columns={'ï»¿home_team': 'home_team'})
                frames.append(df)

    return frames


def add_point_diff_and_results(frames):
    """Adds the point differential for each game as well as columns indicating if the home team won or tied."""
    for df in frames:
        df['game_point_diff'] = df.apply(lambda row: Stats.game_point_diff(row), axis=1)
        df['home_victory'] = df.apply(lambda row: Stats.home_victory(row), axis=1)
        df['home_draw'] = df.apply(lambda row: Stats.home_draw(row), axis=1)

    return frames


def add_team_records(frames):
    """Adds columns for the record of each team going into the game.
    Adds a column for the win percentage of each team going into the game."""
    # For every data frame
    for df in frames:

        # Have a dictionary for each team to their wins, losses and ties
        team_wins = dict()
        team_losses = dict()
        team_ties = dict()

        # Go through all teams and add them to each dictionary with 0 wins, losses and ties
        team_ids = df['home_team'].unique()
        for team in team_ids:
            if not team_wins.get(team):
                team_wins[team] = 0

            if not team_losses.get(team):
                team_losses[team] = 0

            if not team_ties.get(team):
                team_ties[team] = 0

        # Go through each row
        rows = list(df.values)
        for index, row in enumerate(rows):
            row_num = index

            # Get the home and away teams
            team = row[df.columns.get_loc('home_team')]
            opponent = row[df.columns.get_loc('away_team')]

            # Update the home teams record
            df.loc[df.index[row_num], 'home_wins'] = int(team_wins.get(team))
            df.loc[df.index[row_num], 'home_losses'] = int(team_losses.get(team))
            df.loc[df.index[row_num], 'home_ties'] = int(team_ties.get(team))

            # Update the away teams record
            df.loc[df.index[row_num], 'away_wins'] = int(team_wins.get(opponent))
            df.loc[df.index[row_num], 'away_losses'] = int(team_losses.get(opponent))
            df.loc[df.index[row_num], 'away_ties'] = int(team_ties.get(opponent))

            # Increment the records based on the game result
            if row[df.columns.get_loc('home_victory')] == 1:
                team_wins[team] = team_wins.get(team) + 1
                team_losses[opponent] = team_losses.get(opponent) + 1
            elif row[df.columns.get_loc('home_draw')] == 1:
                team_ties[team] = team_ties.get(team) + 1
                team_ties[opponent] = team_ties.get(opponent) + 1
            else:
                team_wins[opponent] = team_wins.get(opponent) + 1
                team_losses[team] = team_losses.get(team) + 1

        # Set the wins, losses and ties column to the int type
        df['home_wins'] = df['home_wins'].astype(np.int64)
        df['home_losses'] = df['home_losses'].astype(np.int64)
        df['home_ties'] = df['home_ties'].astype(np.int64)

        df['away_wins'] = df['away_wins'].astype(np.int64)
        df['away_losses'] = df['away_losses'].astype(np.int64)
        df['away_ties'] = df['away_ties'].astype(np.int64)

        # Calculate the teams win percentage
        df['home_win_pct'] = df.apply(lambda r: Stats.home_win_pct(r), axis=1)
        df['away_win_pct'] = df.apply(lambda r: Stats.away_win_pct(r), axis=1)

    return frames


def add_team_elos(frames, regression_factor=.41, k_factor=42):
    """Adds columns for the Elo for each team going into the game.
    Elo is calculated from the prior Elo of each team and the result of the game.
    If the team does not have an Elo, it is set to a default value of 1500.
    Team Elos are regressed towards 1500 at the end of each season.
    https://en.wikipedia.org/wiki/Elo_rating_system"""
    # Create a dictionary for team elos
    team_elos = dict()

    # For every data frame
    for df in frames:
        # Go through all teams and add them to the dictionary with 1500 elo if they are not there
        team_ids = df['home_team'].unique()
        for team in team_ids:
            if not team_elos.get(team):
                team_elos[team] = 1500

            # Regress each teams elo
            team_elos[team] = team_elos.get(team) * (1 - regression_factor) + 1500 * regression_factor

        # Go through each row
        rows = list(df.values)
        for index, row in enumerate(rows):
            row_num = index

            # Get the home and away teams
            team = row[df.columns.get_loc('home_team')]
            opponent = row[df.columns.get_loc('away_team')]

            # Update each teams elo
            df.loc[df.index[row_num], 'home_elo'] = float(team_elos.get(team))
            df.loc[df.index[row_num], 'away_elo'] = float(team_elos.get(opponent))

            # Calculate each teams new elo based on the game result
            home_elo, away_elo = get_new_elos(team_elos.get(team),
                                              team_elos.get(opponent),
                                              row[df.columns.get_loc('home_victory')] == 1,
                                              row[df.columns.get_loc('home_draw')] == 1,
                                              k_factor)

            # Update the dictionary
            team_elos[team] = home_elo
            team_elos[opponent] = away_elo

    return frames


def get_new_elos(home_elo, away_elo, home_victory, home_draw, k_factor):
    """Calculates the new Elos of each team given their previous Elos and the outcome of the game."""
    q_home = 10 ** (home_elo / 400)
    q_away = 10 ** (away_elo / 400)

    e_home = q_home / (q_home + q_away)
    e_away = q_away / (q_home + q_away)

    if home_victory:
        home_elo = home_elo + k_factor * (1 - e_home)
        away_elo = away_elo + k_factor * (0 - e_away)
    elif home_draw:
        home_elo = home_elo + k_factor * (.5 - e_home)
        away_elo = away_elo + k_factor * (.5 - e_away)
    else:
        home_elo = home_elo + k_factor * (0 - e_home)
        away_elo = away_elo + k_factor * (1 - e_away)

    return home_elo, away_elo


def evaluate_season_elo(df):
    """Evaluates the accuracy of the parameters of the Elo function for a given season
    based off the brier loss function.  Returns the brier loss score for that season."""

    # Get the elo and result columns
    home_elo_column = df['home_elo']
    away_elo_column = df['away_elo']
    results = df['home_victory'].values

    # Calculate the expectations for each game
    home_expectations = list()
    for index in range(len(home_elo_column)):
        home_elo = home_elo_column[index]
        away_elo = away_elo_column[index]

        q_home = 10 ** (home_elo / 400)
        q_away = 10 ** (away_elo / 400)

        e_home = q_home / (q_home + q_away)
        home_expectations.append(e_home)

    # Evaluate the forecasts based on brier loss
    forecasts = np.asarray(home_expectations)
    loss = brier_score_loss(results, forecasts)

    # Return the seasons loss
    return loss


def evaluate_elo(frames):
    """Evaluates the accuracy of all regression factors and K factors
    of the Elo function for averaged over all seasons. Returns the best
    combination of regression factor and K factor."""

    # Benchmark for best brier and features
    best_brier = 1
    best_reg = -1
    best_k = -1

    # List of all the briers for all k factors for all regression factors
    reg_briers = list()

    # For each regression factor between 0 and 1
    for regression_factor in range(0, 101):
        regression_factor = regression_factor / 100

        # Print the regression factor
        print('Regression Factor:', regression_factor)

        # List of all the briers for all k factors 
        k_briers = list()

        # For each k factor between 0 and 50
        for k_factor in range(0, 51):

            # Print the k factor
            print('K Factor:', k_factor)

            # List of all the briers for each season with the given factors
            season_briers = list()

            # Update the elos based on the factors
            frames = add_team_elos(frames, regression_factor, k_factor)

            # For each season 
            for df in frames:
                # Evaluate the seasons loss
                loss = evaluate_season_elo(df)

                # Add the brier to the list of briers for the season
                season_briers.append(loss)

            # Get the average brier loss for the season
            avg = statistics.mean(season_briers)

            # If there is less loss than the previous best
            if avg < best_brier:
                # Update the best features
                best_brier = avg
                best_reg = regression_factor
                best_k = k_factor

            # Get the standard deviation on the brier losses for the season
            std_dev = statistics.stdev(season_briers)

            # Add the season average brier loss to a list of average losses for the k factor
            k_briers.append(str(avg))

            # Print the brier and deviation for the features
            print('Average Brier:', avg)
            print('Brier Deviation:', std_dev)
            print()

        # Add the joined briers for the k factor to a list of briers for the regression factor
        reg_briers.append(','.join(k_briers))

    # Print the joined list of briers for the k factors for the regression factors
    print('\n'.join(reg_briers))

    # Print the best brier and features
    print('Best Brier:', best_brier)
    print(best_reg)
    print(best_k)

    return best_reg, best_k


def add_season_totals(frames):
    """Adds a running total for each stat for each team prior to the game."""
    # For each season
    for df in frames:

        # Get all the team names in the season
        team_ids = df['home_team'].unique()

        # For each team
        for team in team_ids:
            # Get all the games that team played in
            relevant_games = df.loc[(df['home_team'] == team) | (df['away_team'] == team)]
            home = pd.Series(relevant_games['home_team'] == team)

            # Get each stat and get a cumulative summation shifted by 1 index
            points_for = pd.Series(relevant_games.lookup(relevant_games.index,
                                                         home.map({True: 'home_score',
                                                                   False: 'away_score'})))
            total_points_for = pd.Series(pd.Series([0]).append(points_for.cumsum(), ignore_index=True).head(-1))

            points_against = pd.Series(relevant_games.lookup(relevant_games.index,
                                                             home.map({True: 'away_score',
                                                                       False: 'home_score'})))
            total_points_against = pd.Series(pd.Series([0]).append(points_against.cumsum(), ignore_index=True).head(-1))

            first_downs = pd.Series(relevant_games.lookup(relevant_games.index,
                                                          home.map({True: 'home_first_downs',
                                                                    False: 'away_first_downs'})))
            total_first_downs = pd.Series(pd.Series([0]).append(first_downs.cumsum(), ignore_index=True).head(-1))

            rush_attempts = pd.Series(relevant_games.lookup(relevant_games.index,
                                                            home.map({True: 'home_rush_attempts',
                                                                      False: 'away_rush_attempts'})))
            total_rush_attempts = pd.Series(pd.Series([0]).append(rush_attempts.cumsum(), ignore_index=True).head(-1))

            rushing_yards = pd.Series(relevant_games.lookup(relevant_games.index,
                                                            home.map({True: 'home_rushing_yards',
                                                                      False: 'away_rushing_yards'})))
            total_rushing_yards = pd.Series(pd.Series([0]).append(rushing_yards.cumsum(), ignore_index=True).head(-1))

            rushing_touchdowns = pd.Series(relevant_games.lookup(relevant_games.index,
                                                                 home.map({True: 'home_rushing_touchdowns',
                                                                           False: 'away_rushing_touchdowns'})))
            total_rushing_touchdowns = pd.Series(pd.Series([0]).append(rushing_touchdowns.cumsum(),
                                                                       ignore_index=True).head(-1))

            pass_completions = pd.Series(relevant_games.lookup(relevant_games.index,
                                                               home.map({True: 'home_pass_completions',
                                                                         False: 'away_pass_completions'})))
            total_pass_completions = pd.Series(pd.Series([0]).append(pass_completions.cumsum(),
                                                                     ignore_index=True).head(-1))

            pass_attempts = pd.Series(relevant_games.lookup(relevant_games.index,
                                                            home.map({True: 'home_pass_attempts',
                                                                      False: 'away_pass_attempts'})))
            total_pass_attempts = pd.Series(pd.Series([0]).append(pass_attempts.cumsum(), ignore_index=True).head(-1))

            passing_yards = pd.Series(relevant_games.lookup(relevant_games.index,
                                                            home.map({True: 'home_passing_yards',
                                                                      False: 'away_passing_yards'})))
            total_passing_yards = pd.Series(pd.Series([0]).append(passing_yards.cumsum(), ignore_index=True).head(-1))

            passing_touchdowns = pd.Series(relevant_games.lookup(relevant_games.index,
                                                                 home.map({True: 'home_passing_touchdowns',
                                                                           False: 'away_passing_touchdowns'})))
            total_passing_touchdowns = pd.Series(pd.Series([0]).append(passing_touchdowns.cumsum(),
                                                                       ignore_index=True).head(-1))

            interceptions_thrown = pd.Series(relevant_games.lookup(relevant_games.index,
                                                                   home.map({True: 'home_interceptions_thrown',
                                                                             False: 'away_interceptions_thrown'})))
            total_interceptions_thrown = pd.Series(pd.Series([0]).append(interceptions_thrown.cumsum(),
                                                                         ignore_index=True).head(-1))

            times_sacked = pd.Series(relevant_games.lookup(relevant_games.index,
                                                           home.map({True: 'home_times_sacked',
                                                                     False: 'away_times_sacked'})))
            total_times_sacked = pd.Series(pd.Series([0]).append(times_sacked.cumsum(), ignore_index=True).head(-1))

            sacked_yards = pd.Series(relevant_games.lookup(relevant_games.index,
                                                           home.map({True: 'home_sacked_yards',
                                                                     False: 'away_sacked_yards'})))
            total_sacked_yards = pd.Series(pd.Series([0]).append(sacked_yards.cumsum(), ignore_index=True).head(-1))

            net_passing_yards = pd.Series(relevant_games.lookup(relevant_games.index,
                                                                home.map({True: 'home_net_passing_yards',
                                                                          False: 'away_net_passing_yards'})))
            total_net_passing_yards = pd.Series(pd.Series([0]).append(net_passing_yards.cumsum(),
                                                                      ignore_index=True).head(-1))

            total_yards = pd.Series(relevant_games.lookup(relevant_games.index,
                                                          home.map({True: 'home_total_yards',
                                                                    False: 'away_total_yards'})))
            total_total_yards = pd.Series(pd.Series([0]).append(total_yards.cumsum(), ignore_index=True).head(-1))

            fumbles = pd.Series(relevant_games.lookup(relevant_games.index,
                                                      home.map({True: 'home_fumbles',
                                                                False: 'away_fumbles'})))
            total_fumbles = pd.Series(pd.Series([0]).append(fumbles.cumsum(), ignore_index=True).head(-1))

            fumbles_lost = pd.Series(relevant_games.lookup(relevant_games.index,
                                                           home.map({True: 'home_fumbles_lost',
                                                                     False: 'away_fumbles_lost'})))
            total_fumbles_lost = pd.Series(pd.Series([0]).append(fumbles_lost.cumsum(), ignore_index=True).head(-1))

            turnovers = pd.Series(relevant_games.lookup(relevant_games.index,
                                                        home.map({True: 'home_turnovers',
                                                                  False: 'away_turnovers'})))
            total_turnovers = pd.Series(pd.Series([0]).append(turnovers.cumsum(), ignore_index=True).head(-1))

            penalties = pd.Series(relevant_games.lookup(relevant_games.index,
                                                        home.map({True: 'home_penalties',
                                                                  False: 'away_penalties'})))
            total_penalties = pd.Series(pd.Series([0]).append(penalties.cumsum(), ignore_index=True).head(-1))

            penalty_yards = pd.Series(relevant_games.lookup(relevant_games.index,
                                                            home.map({True: 'home_penalty_yards',
                                                                      False: 'away_penalty_yards'})))
            total_penalty_yards = pd.Series(pd.Series([0]).append(penalty_yards.cumsum(), ignore_index=True).head(-1))

            third_down_conversions = pd.Series(relevant_games.lookup(relevant_games.index,
                                                                     home.map({True: 'home_third_down_conversions',
                                                                               False: 'away_third_down_conversions'})))
            total_third_down_conversions = pd.Series(pd.Series([0]).append(third_down_conversions.cumsum(),
                                                                           ignore_index=True).head(-1))

            third_downs = pd.Series(relevant_games.lookup(relevant_games.index,
                                                          home.map({True: 'home_third_downs',
                                                                    False: 'away_third_downs'})))
            total_third_downs = pd.Series(pd.Series([0]).append(third_downs.cumsum(), ignore_index=True).head(-1))

            fourth_down_conversions = pd.Series(relevant_games.lookup(relevant_games.index,
                                                                      home.map({True: 'home_fourth_down_conversions',
                                                                                False: 'away_fourth_down_conversions'}))
                                                )
            total_fourth_down_conversions = pd.Series(pd.Series([0]).append(fourth_down_conversions.cumsum(),
                                                                            ignore_index=True).head(-1))

            fourth_downs = pd.Series(relevant_games.lookup(relevant_games.index,
                                                           home.map({True: 'home_fourth_downs',
                                                                     False: 'away_fourth_downs'})))
            total_fourth_downs = pd.Series(pd.Series([0]).append(fourth_downs.cumsum(), ignore_index=True).head(-1))

            time_of_possession = pd.Series(relevant_games.lookup(relevant_games.index,
                                                                 home.map({True: 'home_time_of_possession',
                                                                           False: 'away_time_of_possession'})))
            total_time_of_possession = pd.Series(pd.Series([0]).append(time_of_possession.cumsum(),
                                                                       ignore_index=True).head(-1))

            yards_allowed = pd.Series(relevant_games.lookup(relevant_games.index,
                                                            home.map({True: 'home_yards_allowed',
                                                                      False: 'away_yards_allowed'})))
            total_yards_allowed = pd.Series(pd.Series([0]).append(yards_allowed.cumsum(), ignore_index=True).head(-1))

            interceptions = pd.Series(relevant_games.lookup(relevant_games.index,
                                                            home.map({True: 'home_interceptions',
                                                                      False: 'away_interceptions'})))
            total_interceptions = pd.Series(pd.Series([0]).append(interceptions.cumsum(), ignore_index=True).head(-1))

            fumbles_forced = pd.Series(relevant_games.lookup(relevant_games.index,
                                                             home.map({True: 'home_fumbles_forced',
                                                                       False: 'away_fumbles_forced'})))
            total_fumbles_forced = pd.Series(pd.Series([0]).append(fumbles_forced.cumsum(), ignore_index=True).head(-1))

            fumbles_recovered = pd.Series(relevant_games.lookup(relevant_games.index,
                                                                home.map({True: 'home_fumbles_recovered',
                                                                          False: 'away_fumbles_recovered'})))
            total_fumbles_recovered = pd.Series(pd.Series([0]).append(fumbles_recovered.cumsum(),
                                                                      ignore_index=True).head(-1))

            turnovers_forced = pd.Series(relevant_games.lookup(relevant_games.index,
                                                               home.map({True: 'home_turnovers_forced',
                                                                         False: 'away_turnovers_forced'})))
            total_turnovers_forced = pd.Series(pd.Series([0]).append(turnovers_forced.cumsum(),
                                                                     ignore_index=True).head(-1))

            sacks = pd.Series(relevant_games.lookup(relevant_games.index,
                                                    home.map({True: 'home_sacks',
                                                              False: 'away_sacks'})))
            total_sacks = pd.Series(pd.Series([0]).append(sacks.cumsum(), ignore_index=True).head(-1))

            sack_yards_forced = pd.Series(relevant_games.lookup(relevant_games.index,
                                                                home.map({True: 'home_sack_yards_forced',
                                                                          False: 'away_sack_yards_forced'})))
            total_sack_yards_forced = pd.Series(pd.Series([0]).append(sack_yards_forced.cumsum(),
                                                                      ignore_index=True).head(-1))

            # For each game the team played in
            for index in range(len(home)):
                # Add the teams running total prior to each game
                if home.iloc[index]:
                    row_num = df.loc[(df['home_team'] == team) & (df['home_games_played'] == index)].index
                    df.loc[row_num, 'home_total_points_for'] = total_points_for.iloc[index]
                    df.loc[row_num, 'home_total_points_against'] = total_points_against.iloc[index]
                    df.loc[row_num, 'home_total_first_downs'] = total_first_downs.iloc[index]
                    df.loc[row_num, 'home_total_rush_attempts'] = total_rush_attempts.iloc[index]
                    df.loc[row_num, 'home_total_rushing_yards'] = total_rushing_yards.iloc[index]
                    df.loc[row_num, 'home_total_rushing_touchdowns'] = total_rushing_touchdowns.iloc[index]
                    df.loc[row_num, 'home_total_pass_completions'] = total_pass_completions.iloc[index]
                    df.loc[row_num, 'home_total_pass_attempts'] = total_pass_attempts.iloc[index]
                    df.loc[row_num, 'home_total_passing_yards'] = total_passing_yards.iloc[index]
                    df.loc[row_num, 'home_total_passing_touchdowns'] = total_passing_touchdowns.iloc[index]
                    df.loc[row_num, 'home_total_interceptions_thrown'] = total_interceptions_thrown.iloc[index]
                    df.loc[row_num, 'home_total_times_sacked'] = total_times_sacked.iloc[index]
                    df.loc[row_num, 'home_total_sacked_yards'] = total_sacked_yards.iloc[index]
                    df.loc[row_num, 'home_total_net_passing_yards'] = total_net_passing_yards.iloc[index]
                    df.loc[row_num, 'home_total_total_yards'] = total_total_yards.iloc[index]
                    df.loc[row_num, 'home_total_fumbles'] = total_fumbles.iloc[index]
                    df.loc[row_num, 'home_total_fumbles_lost'] = total_fumbles_lost.iloc[index]
                    df.loc[row_num, 'home_total_turnovers'] = total_turnovers.iloc[index]
                    df.loc[row_num, 'home_total_penalties'] = total_penalties.iloc[index]
                    df.loc[row_num, 'home_total_penalty_yards'] = total_penalty_yards.iloc[index]
                    df.loc[row_num, 'home_total_third_down_conversions'] = total_third_down_conversions.iloc[index]
                    df.loc[row_num, 'home_total_third_downs'] = total_third_downs.iloc[index]
                    df.loc[row_num, 'home_total_fourth_down_conversions'] = total_fourth_down_conversions.iloc[index]
                    df.loc[row_num, 'home_total_fourth_downs'] = total_fourth_downs.iloc[index]
                    df.loc[row_num, 'home_total_time_of_possession'] = total_time_of_possession.iloc[index]
                    df.loc[row_num, 'home_total_yards_allowed'] = total_yards_allowed.iloc[index]
                    df.loc[row_num, 'home_total_interceptions'] = total_interceptions.iloc[index]
                    df.loc[row_num, 'home_total_fumbles_forced'] = total_fumbles_forced.iloc[index]
                    df.loc[row_num, 'home_total_fumbles_recovered'] = total_fumbles_recovered.iloc[index]
                    df.loc[row_num, 'home_total_turnovers_forced'] = total_turnovers_forced.iloc[index]
                    df.loc[row_num, 'home_total_sacks'] = total_sacks.iloc[index]
                    df.loc[row_num, 'home_total_sack_yards_forced'] = total_sack_yards_forced.iloc[index]
                else:
                    row_num = df.loc[(df['away_team'] == team) & (df['away_games_played'] == index)].index
                    df.loc[row_num, 'away_total_points_for'] = total_points_for.iloc[index]
                    df.loc[row_num, 'away_total_points_against'] = total_points_against.iloc[index]
                    df.loc[row_num, 'away_total_first_downs'] = total_first_downs.iloc[index]
                    df.loc[row_num, 'away_total_rush_attempts'] = total_rush_attempts.iloc[index]
                    df.loc[row_num, 'away_total_rushing_yards'] = total_rushing_yards.iloc[index]
                    df.loc[row_num, 'away_total_rushing_touchdowns'] = total_rushing_touchdowns.iloc[index]
                    df.loc[row_num, 'away_total_pass_completions'] = total_pass_completions.iloc[index]
                    df.loc[row_num, 'away_total_pass_attempts'] = total_pass_attempts.iloc[index]
                    df.loc[row_num, 'away_total_passing_yards'] = total_passing_yards.iloc[index]
                    df.loc[row_num, 'away_total_passing_touchdowns'] = total_passing_touchdowns.iloc[index]
                    df.loc[row_num, 'away_total_interceptions_thrown'] = total_interceptions_thrown.iloc[index]
                    df.loc[row_num, 'away_total_times_sacked'] = total_times_sacked.iloc[index]
                    df.loc[row_num, 'away_total_sacked_yards'] = total_sacked_yards.iloc[index]
                    df.loc[row_num, 'away_total_net_passing_yards'] = total_net_passing_yards.iloc[index]
                    df.loc[row_num, 'away_total_total_yards'] = total_total_yards.iloc[index]
                    df.loc[row_num, 'away_total_fumbles'] = total_fumbles.iloc[index]
                    df.loc[row_num, 'away_total_fumbles_lost'] = total_fumbles_lost.iloc[index]
                    df.loc[row_num, 'away_total_turnovers'] = total_turnovers.iloc[index]
                    df.loc[row_num, 'away_total_penalties'] = total_penalties.iloc[index]
                    df.loc[row_num, 'away_total_penalty_yards'] = total_penalty_yards.iloc[index]
                    df.loc[row_num, 'away_total_third_down_conversions'] = total_third_down_conversions.iloc[index]
                    df.loc[row_num, 'away_total_third_downs'] = total_third_downs.iloc[index]
                    df.loc[row_num, 'away_total_fourth_down_conversions'] = total_fourth_down_conversions.iloc[index]
                    df.loc[row_num, 'away_total_fourth_downs'] = total_fourth_downs.iloc[index]
                    df.loc[row_num, 'away_total_time_of_possession'] = total_time_of_possession.iloc[index]
                    df.loc[row_num, 'away_total_yards_allowed'] = total_yards_allowed.iloc[index]
                    df.loc[row_num, 'away_total_interceptions'] = total_interceptions.iloc[index]
                    df.loc[row_num, 'away_total_fumbles_forced'] = total_fumbles_forced.iloc[index]
                    df.loc[row_num, 'away_total_fumbles_recovered'] = total_fumbles_recovered.iloc[index]
                    df.loc[row_num, 'away_total_turnovers_forced'] = total_turnovers_forced.iloc[index]
                    df.loc[row_num, 'away_total_sacks'] = total_sacks.iloc[index]
                    df.loc[row_num, 'away_total_sack_yards_forced'] = total_sack_yards_forced.iloc[index]

        # Change the data types to integers
        df['home_total_points_for'] = df['home_total_points_for'].astype(np.int64)
        df['home_total_points_against'] = df['home_total_points_against'].astype(np.int64)
        df['home_total_first_downs'] = df['home_total_first_downs'].astype(np.int64)
        df['home_total_rush_attempts'] = df['home_total_rush_attempts'].astype(np.int64)
        df['home_total_rushing_yards'] = df['home_total_rushing_yards'].astype(np.int64)
        df['home_total_rushing_touchdowns'] = df['home_total_rushing_touchdowns'].astype(np.int64)
        df['home_total_pass_completions'] = df['home_total_pass_completions'].astype(np.int64)
        df['home_total_pass_attempts'] = df['home_total_pass_attempts'].astype(np.int64)
        df['home_total_passing_yards'] = df['home_total_passing_yards'].astype(np.int64)
        df['home_total_passing_touchdowns'] = df['home_total_passing_touchdowns'].astype(np.int64)
        df['home_total_interceptions_thrown'] = df['home_total_interceptions_thrown'].astype(np.int64)
        df['home_total_times_sacked'] = df['home_total_times_sacked'].astype(np.int64)
        df['home_total_sacked_yards'] = df['home_total_sacked_yards'].astype(np.int64)
        df['home_total_net_passing_yards'] = df['home_total_net_passing_yards'].astype(np.int64)
        df['home_total_total_yards'] = df['home_total_total_yards'].astype(np.int64)
        df['home_total_fumbles'] = df['home_total_fumbles'].astype(np.int64)
        df['home_total_fumbles_lost'] = df['home_total_fumbles_lost'].astype(np.int64)
        df['home_total_turnovers'] = df['home_total_turnovers'].astype(np.int64)
        df['home_total_penalties'] = df['home_total_penalties'].astype(np.int64)
        df['home_total_penalty_yards'] = df['home_total_penalty_yards'].astype(np.int64)
        df['home_total_third_down_conversions'] = df['home_total_third_down_conversions'].astype(np.int64)
        df['home_total_third_downs'] = df['home_total_third_downs'].astype(np.int64)
        df['home_total_fourth_down_conversions'] = df['home_total_fourth_down_conversions'].astype(np.int64)
        df['home_total_fourth_downs'] = df['home_total_fourth_downs'].astype(np.int64)
        df['home_total_time_of_possession'] = df['home_total_time_of_possession'].astype(np.int64)
        df['home_total_yards_allowed'] = df['home_total_yards_allowed'].astype(np.int64)
        df['home_total_interceptions'] = df['home_total_interceptions'].astype(np.int64)
        df['home_total_fumbles_forced'] = df['home_total_fumbles_forced'].astype(np.int64)
        df['home_total_fumbles_recovered'] = df['home_total_fumbles_recovered'].astype(np.int64)
        df['home_total_turnovers_forced'] = df['home_total_turnovers_forced'].astype(np.int64)
        df['home_total_sacks'] = df['home_total_sacks'].astype(np.int64)
        df['home_total_sack_yards_forced'] = df['home_total_sack_yards_forced'].astype(np.int64)

        df['away_total_points_for'] = df['away_total_points_for'].astype(np.int64)
        df['away_total_points_against'] = df['away_total_points_against'].astype(np.int64)
        df['away_total_first_downs'] = df['away_total_first_downs'].astype(np.int64)
        df['away_total_rush_attempts'] = df['away_total_rush_attempts'].astype(np.int64)
        df['away_total_rushing_yards'] = df['away_total_rushing_yards'].astype(np.int64)
        df['away_total_rushing_touchdowns'] = df['away_total_rushing_touchdowns'].astype(np.int64)
        df['away_total_pass_completions'] = df['away_total_pass_completions'].astype(np.int64)
        df['away_total_pass_attempts'] = df['away_total_pass_attempts'].astype(np.int64)
        df['away_total_passing_yards'] = df['away_total_passing_yards'].astype(np.int64)
        df['away_total_passing_touchdowns'] = df['away_total_passing_touchdowns'].astype(np.int64)
        df['away_total_interceptions_thrown'] = df['away_total_interceptions_thrown'].astype(np.int64)
        df['away_total_times_sacked'] = df['away_total_times_sacked'].astype(np.int64)
        df['away_total_sacked_yards'] = df['away_total_sacked_yards'].astype(np.int64)
        df['away_total_net_passing_yards'] = df['away_total_net_passing_yards'].astype(np.int64)
        df['away_total_total_yards'] = df['away_total_total_yards'].astype(np.int64)
        df['away_total_fumbles'] = df['away_total_fumbles'].astype(np.int64)
        df['away_total_fumbles_lost'] = df['away_total_fumbles_lost'].astype(np.int64)
        df['away_total_turnovers'] = df['away_total_turnovers'].astype(np.int64)
        df['away_total_penalties'] = df['away_total_penalties'].astype(np.int64)
        df['away_total_penalty_yards'] = df['away_total_penalty_yards'].astype(np.int64)
        df['away_total_third_down_conversions'] = df['away_total_third_down_conversions'].astype(np.int64)
        df['away_total_third_downs'] = df['away_total_third_downs'].astype(np.int64)
        df['away_total_fourth_down_conversions'] = df['away_total_fourth_down_conversions'].astype(np.int64)
        df['away_total_fourth_downs'] = df['away_total_fourth_downs'].astype(np.int64)
        df['away_total_time_of_possession'] = df['away_total_time_of_possession'].astype(np.int64)
        df['away_total_yards_allowed'] = df['away_total_yards_allowed'].astype(np.int64)
        df['away_total_interceptions'] = df['away_total_interceptions'].astype(np.int64)
        df['away_total_fumbles_forced'] = df['away_total_fumbles_forced'].astype(np.int64)
        df['away_total_fumbles_recovered'] = df['away_total_fumbles_recovered'].astype(np.int64)
        df['away_total_turnovers_forced'] = df['away_total_turnovers_forced'].astype(np.int64)
        df['away_total_sacks'] = df['away_total_sacks'].astype(np.int64)
        df['away_total_sack_yards_forced'] = df['away_total_sack_yards_forced'].astype(np.int64)

    return frames


def add_season_averages(frames):
    """Adds a running average for each stat for each team prior to the game."""
    for df in frames:
        df['home_average_points_for'] = df.apply(lambda row: Stats.home_average_points_for(row), axis=1)
        df['home_average_points_against'] = df.apply(lambda row: Stats.home_average_points_against(row), axis=1)
        df['home_average_first_downs'] = df.apply(lambda row: Stats.home_average_first_downs(row), axis=1)
        df['home_average_rush_attempts'] = df.apply(lambda row: Stats.home_average_rush_attempts(row), axis=1)
        df['home_average_rushing_yards'] = df.apply(lambda row: Stats.home_average_rushing_yards(row), axis=1)
        df['home_average_rushing_touchdowns'] = df.apply(lambda row: Stats.home_average_rushing_touchdowns(row), axis=1)
        df['home_average_pass_completions'] = df.apply(lambda row: Stats.home_average_pass_completions(row), axis=1)
        df['home_average_pass_attempts'] = df.apply(lambda row: Stats.home_average_pass_attempts(row), axis=1)
        df['home_average_passing_yards'] = df.apply(lambda row: Stats.home_average_passing_yards(row), axis=1)
        df['home_average_passing_touchdowns'] = df.apply(lambda row: Stats.home_average_passing_touchdowns(row), axis=1)
        df['home_average_interceptions_thrown'] = df.apply(lambda row: Stats.home_average_interceptions_thrown(row),
                                                           axis=1)
        df['home_average_times_sacked'] = df.apply(lambda row: Stats.home_average_times_sacked(row), axis=1)
        df['home_average_sacked_yards'] = df.apply(lambda row: Stats.home_average_sacked_yards(row), axis=1)
        df['home_average_net_passing_yards'] = df.apply(lambda row: Stats.home_average_net_passing_yards(row), axis=1)
        df['home_average_total_yards'] = df.apply(lambda row: Stats.home_average_total_yards(row), axis=1)
        df['home_average_fumbles'] = df.apply(lambda row: Stats.home_average_fumbles(row), axis=1)
        df['home_average_fumbles_lost'] = df.apply(lambda row: Stats.home_average_fumbles_lost(row), axis=1)
        df['home_average_turnovers'] = df.apply(lambda row: Stats.home_average_turnovers(row), axis=1)
        df['home_average_penalties'] = df.apply(lambda row: Stats.home_average_penalties(row), axis=1)
        df['home_average_penalty_yards'] = df.apply(lambda row: Stats.home_average_penalty_yards(row), axis=1)
        df['home_average_third_down_pct'] = df.apply(lambda row: Stats.home_average_third_down_pct(row), axis=1)
        df['home_average_fourth_down_pct'] = df.apply(lambda row: Stats.home_average_fourth_down_pct(row), axis=1)
        df['home_average_time_of_possession'] = df.apply(lambda row: Stats.home_average_time_of_possession(row), axis=1)
        df['home_average_yards_allowed'] = df.apply(lambda row: Stats.home_average_yards_allowed(row), axis=1)
        df['home_average_interceptions'] = df.apply(lambda row: Stats.home_average_interceptions(row), axis=1)
        df['home_average_fumbles_forced'] = df.apply(lambda row: Stats.home_average_fumbles_forced(row), axis=1)
        df['home_average_fumbles_recovered'] = df.apply(lambda row: Stats.home_average_fumbles_recovered(row), axis=1)
        df['home_average_turnovers_forced'] = df.apply(lambda row: Stats.home_average_turnovers_forced(row), axis=1)
        df['home_average_sacks'] = df.apply(lambda row: Stats.home_average_sacks(row), axis=1)
        df['home_average_sack_yards_forced'] = df.apply(lambda row: Stats.home_average_sack_yards_forced(row), axis=1)

        df['away_average_points_for'] = df.apply(lambda row: Stats.away_average_points_for(row), axis=1)
        df['away_average_points_against'] = df.apply(lambda row: Stats.away_average_points_against(row), axis=1)
        df['away_average_first_downs'] = df.apply(lambda row: Stats.away_average_first_downs(row), axis=1)
        df['away_average_rush_attempts'] = df.apply(lambda row: Stats.away_average_rush_attempts(row), axis=1)
        df['away_average_rushing_yards'] = df.apply(lambda row: Stats.away_average_rushing_yards(row), axis=1)
        df['away_average_rushing_touchdowns'] = df.apply(lambda row: Stats.away_average_rushing_touchdowns(row), axis=1)
        df['away_average_pass_completions'] = df.apply(lambda row: Stats.away_average_pass_completions(row), axis=1)
        df['away_average_pass_attempts'] = df.apply(lambda row: Stats.away_average_pass_attempts(row), axis=1)
        df['away_average_passing_yards'] = df.apply(lambda row: Stats.away_average_passing_yards(row), axis=1)
        df['away_average_passing_touchdowns'] = df.apply(lambda row: Stats.away_average_passing_touchdowns(row), axis=1)
        df['away_average_interceptions_thrown'] = df.apply(lambda row: Stats.away_average_interceptions_thrown(row),
                                                           axis=1)
        df['away_average_times_sacked'] = df.apply(lambda row: Stats.away_average_times_sacked(row), axis=1)
        df['away_average_sacked_yards'] = df.apply(lambda row: Stats.away_average_sacked_yards(row), axis=1)
        df['away_average_net_passing_yards'] = df.apply(lambda row: Stats.away_average_net_passing_yards(row), axis=1)
        df['away_average_total_yards'] = df.apply(lambda row: Stats.away_average_total_yards(row), axis=1)
        df['away_average_fumbles'] = df.apply(lambda row: Stats.away_average_fumbles(row), axis=1)
        df['away_average_fumbles_lost'] = df.apply(lambda row: Stats.away_average_fumbles_lost(row), axis=1)
        df['away_average_turnovers'] = df.apply(lambda row: Stats.away_average_turnovers(row), axis=1)
        df['away_average_penalties'] = df.apply(lambda row: Stats.away_average_penalties(row), axis=1)
        df['away_average_penalty_yards'] = df.apply(lambda row: Stats.away_average_penalty_yards(row), axis=1)
        df['away_average_third_down_pct'] = df.apply(lambda row: Stats.away_average_third_down_pct(row), axis=1)
        df['away_average_fourth_down_pct'] = df.apply(lambda row: Stats.away_average_fourth_down_pct(row), axis=1)
        df['away_average_time_of_possession'] = df.apply(lambda row: Stats.away_average_time_of_possession(row), axis=1)
        df['away_average_yards_allowed'] = df.apply(lambda row: Stats.away_average_yards_allowed(row), axis=1)
        df['away_average_interceptions'] = df.apply(lambda row: Stats.away_average_interceptions(row), axis=1)
        df['away_average_fumbles_forced'] = df.apply(lambda row: Stats.away_average_fumbles_forced(row), axis=1)
        df['away_average_fumbles_recovered'] = df.apply(lambda row: Stats.away_average_fumbles_recovered(row), axis=1)
        df['away_average_turnovers_forced'] = df.apply(lambda row: Stats.away_average_turnovers_forced(row), axis=1)
        df['away_average_sacks'] = df.apply(lambda row: Stats.away_average_sacks(row), axis=1)
        df['away_average_sack_yards_forced'] = df.apply(lambda row: Stats.away_average_sack_yards_forced(row), axis=1)

    return frames


def add_advanced_stats(frames):
    """Adds additional stats for each team prior to the game."""
    for df in frames:
        df['home_average_yards_per_rush_attempt'] = df.apply(lambda row:
                                                             Stats.home_average_yards_per_rush_attempt(row), axis=1)
        df['home_average_rushing_touchdowns_per_attempt'] = df.apply(
            lambda row: Stats.home_average_rushing_touchdowns_per_attempt(row), axis=1)
        df['home_average_yards_per_pass_attempt'] = df.apply(lambda row:
                                                             Stats.home_average_yards_per_pass_attempt(row), axis=1)
        df['home_average_passing_touchdowns_per_attempt'] = df.apply(
            lambda row: Stats.home_average_passing_touchdowns_per_attempt(row), axis=1)
        df['home_average_yards_per_pass_completion'] = df.apply(
            lambda row: Stats.home_average_yards_per_pass_completion(row), axis=1)
        df['home_average_rushing_play_pct'] = df.apply(lambda row: Stats.home_average_rushing_play_pct(row), axis=1)
        df['home_average_passing_play_pct'] = df.apply(lambda row: Stats.home_average_passing_play_pct(row), axis=1)
        df['home_average_rushing_yards_pct'] = df.apply(lambda row: Stats.home_average_rushing_yards_pct(row), axis=1)
        df['home_average_passing_yards_pct'] = df.apply(lambda row: Stats.home_average_passing_yards_pct(row), axis=1)
        df['home_average_completion_pct'] = df.apply(lambda row: Stats.home_average_completion_pct(row), axis=1)
        df['home_average_sacked_pct'] = df.apply(lambda row: Stats.home_average_sacked_pct(row), axis=1)
        df['home_average_passer_rating'] = df.apply(lambda row: Stats.home_average_passer_rating(row), axis=1)
        df['home_average_touchdowns'] = df.apply(lambda row: Stats.home_average_touchdowns(row), axis=1)
        df['home_average_yards_per_point'] = df.apply(lambda row: Stats.home_average_yards_per_point(row), axis=1)
        df['home_average_scoring_margin'] = df.apply(lambda row: Stats.home_average_scoring_margin(row), axis=1)
        df['home_average_turnover_margin'] = df.apply(lambda row: Stats.home_average_turnover_margin(row), axis=1)

        df['away_average_yards_per_rush_attempt'] = df.apply(lambda row:
                                                             Stats.away_average_yards_per_rush_attempt(row), axis=1)
        df['away_average_rushing_touchdowns_per_attempt'] = df.apply(
            lambda row: Stats.away_average_rushing_touchdowns_per_attempt(row), axis=1)
        df['away_average_yards_per_pass_attempt'] = df.apply(lambda row:
                                                             Stats.away_average_yards_per_pass_attempt(row), axis=1)
        df['away_average_passing_touchdowns_per_attempt'] = df.apply(
            lambda row: Stats.away_average_passing_touchdowns_per_attempt(row), axis=1)
        df['away_average_yards_per_pass_completion'] = df.apply(
            lambda row: Stats.away_average_yards_per_pass_completion(row), axis=1)
        df['away_average_rushing_play_pct'] = df.apply(lambda row: Stats.away_average_rushing_play_pct(row), axis=1)
        df['away_average_passing_play_pct'] = df.apply(lambda row: Stats.away_average_passing_play_pct(row), axis=1)
        df['away_average_rushing_yards_pct'] = df.apply(lambda row: Stats.away_average_rushing_yards_pct(row), axis=1)
        df['away_average_passing_yards_pct'] = df.apply(lambda row: Stats.away_average_passing_yards_pct(row), axis=1)
        df['away_average_completion_pct'] = df.apply(lambda row: Stats.away_average_completion_pct(row), axis=1)
        df['away_average_sacked_pct'] = df.apply(lambda row: Stats.away_average_sacked_pct(row), axis=1)
        df['away_average_passer_rating'] = df.apply(lambda row: Stats.away_average_passer_rating(row), axis=1)
        df['away_average_touchdowns'] = df.apply(lambda row: Stats.away_average_touchdowns(row), axis=1)
        df['away_average_yards_per_point'] = df.apply(lambda row: Stats.away_average_yards_per_point(row), axis=1)
        df['away_average_scoring_margin'] = df.apply(lambda row: Stats.away_average_scoring_margin(row), axis=1)
        df['away_average_turnover_margin'] = df.apply(lambda row: Stats.away_average_turnover_margin(row), axis=1)

    return frames


def add_stat_differences(frames):
    """Calculates the margin between the home and away teams for the averages and advanced stats columns.
    Differences are from the home team's perspective (home_stat - away_stat)."""
    for df in frames:
        df['win_pct_diff'] = df.apply(lambda row: Stats.win_pct_diff(row), axis=1)
        df['elo_diff'] = df.apply(lambda row: Stats.elo_diff(row), axis=1)
        df['average_points_for_diff'] = df.apply(lambda row: Stats.average_points_for_diff(row), axis=1)
        df['average_points_against_diff'] = df.apply(lambda row: Stats.average_points_against_diff(row), axis=1)
        df['average_first_downs_diff'] = df.apply(lambda row: Stats.average_first_downs_diff(row), axis=1)
        df['average_rush_attempts_diff'] = df.apply(lambda row: Stats.average_rush_attempts_diff(row), axis=1)
        df['average_rushing_yards_diff'] = df.apply(lambda row: Stats.average_rushing_yards_diff(row), axis=1)
        df['average_rushing_touchdowns_diff'] = df.apply(lambda row: Stats.average_rushing_touchdowns_diff(row), axis=1)
        df['average_pass_completions_diff'] = df.apply(lambda row: Stats.average_pass_completions_diff(row), axis=1)
        df['average_pass_attempts_diff'] = df.apply(lambda row: Stats.average_pass_attempts_diff(row), axis=1)
        df['average_passing_yards_diff'] = df.apply(lambda row: Stats.average_passing_yards_diff(row), axis=1)
        df['average_passing_touchdowns_diff'] = df.apply(lambda row: Stats.average_passing_touchdowns_diff(row), axis=1)
        df['average_interceptions_thrown_diff'] = df.apply(lambda row: Stats.average_interceptions_thrown_diff(row),
                                                           axis=1)
        df['average_times_sacked_diff'] = df.apply(lambda row: Stats.average_times_sacked_diff(row), axis=1)
        df['average_sacked_yards_diff'] = df.apply(lambda row: Stats.average_sacked_yards_diff(row), axis=1)
        df['average_net_passing_yards_diff'] = df.apply(lambda row: Stats.average_net_passing_yards_diff(row), axis=1)
        df['average_total_yards_diff'] = df.apply(lambda row: Stats.average_total_yards_diff(row), axis=1)
        df['average_fumbles_diff'] = df.apply(lambda row: Stats.average_fumbles_diff(row), axis=1)
        df['average_fumbles_lost_diff'] = df.apply(lambda row: Stats.average_fumbles_lost_diff(row), axis=1)
        df['average_turnovers_diff'] = df.apply(lambda row: Stats.average_turnovers_diff(row), axis=1)
        df['average_penalties_diff'] = df.apply(lambda row: Stats.average_penalties_diff(row), axis=1)
        df['average_penalty_yards_diff'] = df.apply(lambda row: Stats.average_penalty_yards_diff(row), axis=1)
        df['average_third_down_pct_diff'] = df.apply(lambda row: Stats.average_third_down_pct_diff(row), axis=1)
        df['average_fourth_down_pct_diff'] = df.apply(lambda row: Stats.average_fourth_down_pct_diff(row), axis=1)
        df['average_time_of_possession_diff'] = df.apply(lambda row: Stats.average_time_of_possession_diff(row), axis=1)
        df['average_yards_allowed_diff'] = df.apply(lambda row: Stats.average_yards_allowed_diff(row), axis=1)
        df['average_interceptions_diff'] = df.apply(lambda row: Stats.average_interceptions_diff(row), axis=1)
        df['average_fumbles_forced_diff'] = df.apply(lambda row: Stats.average_fumbles_forced_diff(row), axis=1)
        df['average_fumbles_recovered_diff'] = df.apply(lambda row: Stats.average_fumbles_recovered_diff(row), axis=1)
        df['average_turnovers_forced_diff'] = df.apply(lambda row: Stats.average_turnovers_forced_diff(row), axis=1)
        df['average_sacks_diff'] = df.apply(lambda row: Stats.average_sacks_diff(row), axis=1)
        df['average_sack_yards_forced_diff'] = df.apply(lambda row: Stats.average_sack_yards_forced_diff(row), axis=1)
        df['average_yards_per_rush_attempt_diff'] = df.apply(lambda row: Stats.average_yards_per_rush_attempt_diff(row),
                                                             axis=1)
        df['average_rushing_touchdowns_per_attempt_diff'] = df.apply(
            lambda row: Stats.average_rushing_touchdowns_per_attempt_diff(row), axis=1)
        df['average_yards_per_pass_attempt_diff'] = df.apply(lambda row: Stats.average_yards_per_pass_attempt_diff(row),
                                                             axis=1)
        df['average_passing_touchdowns_per_attempt_diff'] = df.apply(
            lambda row: Stats.average_passing_touchdowns_per_attempt_diff(row), axis=1)
        df['average_yards_per_pass_completion_diff'] = df.apply(
            lambda row: Stats.average_yards_per_pass_completion_diff(row), axis=1)
        df['average_rushing_play_pct_diff'] = df.apply(lambda row: Stats.average_rushing_play_pct_diff(row), axis=1)
        df['average_passing_play_pct_diff'] = df.apply(lambda row: Stats.average_passing_play_pct_diff(row), axis=1)
        df['average_rushing_yards_pct_diff'] = df.apply(lambda row: Stats.average_rushing_yards_pct_diff(row), axis=1)
        df['average_passing_yards_pct_diff'] = df.apply(lambda row: Stats.average_passing_yards_pct_diff(row), axis=1)
        df['average_completion_pct_diff'] = df.apply(lambda row: Stats.average_completion_pct_diff(row), axis=1)
        df['average_sacked_pct_diff'] = df.apply(lambda row: Stats.average_sacked_pct_diff(row), axis=1)
        df['average_passer_rating_diff'] = df.apply(lambda row: Stats.average_passer_rating_diff(row), axis=1)
        df['average_touchdowns_diff'] = df.apply(lambda row: Stats.average_touchdowns_diff(row), axis=1)
        df['average_yards_per_point_diff'] = df.apply(lambda row: Stats.average_yards_per_point_diff(row), axis=1)
        df['average_scoring_margin_diff'] = df.apply(lambda row: Stats.average_scoring_margin_diff(row), axis=1)
        df['average_turnover_margin_diff'] = df.apply(lambda row: Stats.average_turnover_margin_diff(row), axis=1)

    return frames


def combine_frames(frames):
    """Combines all of the season data frames into a total frame and writes it to a csv file."""

    # Check that all seasons have the same number of columns
    check_consistent_length(*[frame.T for frame in frames])

    # Check that all seasons have the same number of rows
    check_consistent_length(*frames)

    # Combine all seasons into one dataframe
    combined = pd.concat(frames, sort=False)
    print(combined.shape)
    print()

    # Get the percentage of the target variable that is true
    num_obs = len(combined)
    num_wins = len(combined.loc[combined['home_victory'] == 1])
    num_losses = len(combined.loc[combined['home_victory'] == 0])
    num_ties = len(combined.loc[combined['home_draw'] == 1])
    print('Number of wins:  {0} ({1:2.2f}%)'.format(num_wins, (num_wins / num_obs) * 100))
    print('Number of losses: {0} ({1:2.2f}%)'.format(num_losses, (num_losses / num_obs) * 100))
    print('Number of ties: {0} ({1:2.2f}%)'.format(num_ties, (num_ties / num_obs) * 100))
    print()

    # Get the percentage of the target variable that is true during regular season games
    regular_season_games = combined.loc[(combined['week'] < 18)]
    num_obs = len(regular_season_games)
    num_wins = len(regular_season_games.loc[regular_season_games['home_victory'] == 1])
    num_losses = len(regular_season_games.loc[regular_season_games['home_victory'] == 0])
    num_ties = len(regular_season_games.loc[regular_season_games['home_draw'] == 1])
    print('Number of regular season wins:  {0} ({1:2.2f}%)'.format(num_wins, (num_wins / num_obs) * 100))
    print('Number of regular season losses: {0} ({1:2.2f}%)'.format(num_losses, (num_losses / num_obs) * 100))
    print('Number of regular season ties: {0} ({1:2.2f}%)'.format(num_ties, (num_ties / num_obs) * 100))
    print()

    # Get the percentage of the target variable that is true during post season games (excluding the superbowls)
    playoff_games = combined.loc[(combined['week'] == 18) | (combined['week'] == 19) | (combined['week'] == 20)]
    num_obs = len(playoff_games)
    num_wins = len(playoff_games.loc[playoff_games['home_victory'] == 1])
    num_losses = len(playoff_games.loc[playoff_games['home_victory'] == 0])
    num_ties = len(playoff_games.loc[playoff_games['home_draw'] == 1])
    print('Number of playoff wins:  {0} ({1:2.2f}%)'.format(num_wins, (num_wins / num_obs) * 100))
    print('Number of playoff losses: {0} ({1:2.2f}%)'.format(num_losses, (num_losses / num_obs) * 100))
    print('Number of playoff ties: {0} ({1:2.2f}%)'.format(num_ties, (num_ties / num_obs) * 100))
    print()

    # Get the percentage of the target variable that is true during the superbowl
    superbowl_games = combined.loc[(combined['week'] == 21)]
    num_obs = len(superbowl_games)
    num_wins = len(superbowl_games.loc[superbowl_games['home_victory'] == 1])
    num_losses = len(superbowl_games.loc[superbowl_games['home_victory'] == 0])
    num_ties = len(superbowl_games.loc[superbowl_games['home_draw'] == 1])
    print('Number of superbowl wins:  {0} ({1:2.2f}%)'.format(num_wins, (num_wins / num_obs) * 100))
    print('Number of superbowl losses: {0} ({1:2.2f}%)'.format(num_losses, (num_losses / num_obs) * 100))
    print('Number of superbowl ties: {0} ({1:2.2f}%)'.format(num_ties, (num_ties / num_obs) * 100))
    print()

    """It's clear that the teams chance of victory is increased when the are playing at home (57.3% chance at home 
       vs 42.7% chance on the road). This however is not a reflection of the quality of the home teams vs away teams.
       On average, during the regular season, the strength of the home team is equal to the strength of the away team 
       (since all teams play an equal number of home and road games). During the post season, the stronger team almost 
       always has home field advantage. These games however are of minimal impact due to their rarity.  Almost 96% of 
       the home victories are during the regular season (between teams that are on average of equal strength)."""

    combined.to_csv(game_data_dir + '20022018.csv', index=False)

    return combined


def oversample_data():
    """Over samples the data to get an even number of wins and losses.  This causes the number of predicted losses to
    increase, having an improvement on the negative prediction rate.  However, this also caused the overall brier to
    be significantly lower in testing against the 2018 season."""
    # Get the data frame for all seasons
    original_df = pd.read_csv(game_data_dir + '20022018.csv')

    # Over sample the data to get an even number of wins and losses
    random_over_sampler = RandomOverSampler()
    oversampled, y_ros = random_over_sampler.fit_sample(original_df, original_df['home_victory'])

    df = pd.DataFrame(data=oversampled, columns=original_df.columns)

    # Get the percentage of the target variable that is true
    num_obs = len(df)
    num_wins = len(df.loc[df['home_victory'] == 1])
    num_losses = len(df.loc[df['home_victory'] == 0])
    num_ties = len(df.loc[df['home_draw'] == 1])
    print('Number of wins:  {0} ({1:2.2f}%)'.format(num_wins, (num_wins / num_obs) * 100))
    print('Number of losses: {0} ({1:2.2f}%)'.format(num_losses, (num_losses / num_obs) * 100))
    print('Number of ties: {0} ({1:2.2f}%)'.format(num_ties, (num_ties / num_obs) * 100))
    print()

    # Get the percentage of the target variable that is true during regular season games
    regular_season_games = df.loc[(df['week'] < 18)]
    num_obs = len(regular_season_games)
    num_wins = len(regular_season_games.loc[regular_season_games['home_victory'] == 1])
    num_losses = len(regular_season_games.loc[regular_season_games['home_victory'] == 0])
    num_ties = len(regular_season_games.loc[regular_season_games['home_draw'] == 1])
    print('Number of regular season wins:  {0} ({1:2.2f}%)'.format(num_wins, (num_wins / num_obs) * 100))
    print('Number of regular season losses: {0} ({1:2.2f}%)'.format(num_losses, (num_losses / num_obs) * 100))
    print('Number of regular season ties: {0} ({1:2.2f}%)'.format(num_ties, (num_ties / num_obs) * 100))
    print()

    # Get the percentage of the target variable that is true during post season games (excluding the superbowls)
    playoff_games = df.loc[(df['week'] == 18) | (df['week'] == 19) | (df['week'] == 20)]
    num_obs = len(playoff_games)
    num_wins = len(playoff_games.loc[playoff_games['home_victory'] == 1])
    num_losses = len(playoff_games.loc[playoff_games['home_victory'] == 0])
    num_ties = len(playoff_games.loc[playoff_games['home_draw'] == 1])
    print('Number of playoff wins:  {0} ({1:2.2f}%)'.format(num_wins, (num_wins / num_obs) * 100))
    print('Number of playoff losses: {0} ({1:2.2f}%)'.format(num_losses, (num_losses / num_obs) * 100))
    print('Number of playoff ties: {0} ({1:2.2f}%)'.format(num_ties, (num_ties / num_obs) * 100))
    print()

    # Get the percentage of the target variable that is true during the superbowl
    superbowl_games = df.loc[(df['week'] == 21)]
    num_obs = len(superbowl_games)
    num_wins = len(superbowl_games.loc[superbowl_games['home_victory'] == 1])
    num_losses = len(superbowl_games.loc[superbowl_games['home_victory'] == 0])
    num_ties = len(superbowl_games.loc[superbowl_games['home_draw'] == 1])
    print('Number of superbowl wins:  {0} ({1:2.2f}%)'.format(num_wins, (num_wins / num_obs) * 100))
    print('Number of superbowl losses: {0} ({1:2.2f}%)'.format(num_losses, (num_losses / num_obs) * 100))
    print('Number of superbowl ties: {0} ({1:2.2f}%)'.format(num_ties, (num_ties / num_obs) * 100))
    print()

    # print(df.dtypes)
    df['home_spread'] = df['home_spread'].astype(np.int64)
    df['home_victory'] = df['home_victory'].astype(np.int64)
    df['win_pct_diff'] = df['win_pct_diff'].astype(np.float64)
    df['elo_diff'] = df['elo_diff'].astype(np.float64)
    df['average_points_for_diff'] = df['average_points_for_diff'].astype(np.float64)
    df['average_points_against_diff'] = df['average_points_against_diff'].astype(np.float64)
    df['average_first_downs_diff'] = df['average_first_downs_diff'].astype(np.float64)
    df['average_rush_attempts_diff'] = df['average_rush_attempts_diff'].astype(np.float64)
    df['average_rushing_yards_diff'] = df['average_rushing_yards_diff'].astype(np.float64)
    df['average_rushing_touchdowns_diff'] = df['average_rushing_touchdowns_diff'].astype(np.float64)
    df['average_pass_completions_diff'] = df['average_pass_completions_diff'].astype(np.float64)
    df['average_pass_attempts_diff'] = df['average_pass_attempts_diff'].astype(np.float64)
    df['average_passing_yards_diff'] = df['average_passing_yards_diff'].astype(np.float64)
    df['average_passing_touchdowns_diff'] = df['average_passing_touchdowns_diff'].astype(np.float64)
    df['average_interceptions_thrown_diff'] = df['average_interceptions_thrown_diff'].astype(np.float64)
    df['average_times_sacked_diff'] = df['average_times_sacked_diff'].astype(np.float64)
    df['average_sacked_yards_diff'] = df['average_sacked_yards_diff'].astype(np.float64)
    df['average_net_passing_yards_diff'] = df['average_net_passing_yards_diff'].astype(np.float64)
    df['average_total_yards_diff'] = df['average_total_yards_diff'].astype(np.float64)
    df['average_fumbles_diff'] = df['average_fumbles_diff'].astype(np.float64)
    df['average_fumbles_lost_diff'] = df['average_fumbles_lost_diff'].astype(np.float64)
    df['average_turnovers_diff'] = df['average_turnovers_diff'].astype(np.float64)
    df['average_penalties_diff'] = df['average_penalties_diff'].astype(np.float64)
    df['average_penalty_yards_diff'] = df['average_penalty_yards_diff'].astype(np.float64)
    df['average_third_down_pct_diff'] = df['average_third_down_pct_diff'].astype(np.float64)
    df['average_fourth_down_pct_diff'] = df['average_fourth_down_pct_diff'].astype(np.float64)
    df['average_time_of_possession_diff'] = df['average_time_of_possession_diff'].astype(np.float64)
    df['average_yards_allowed_diff'] = df['average_yards_allowed_diff'].astype(np.float64)
    df['average_interceptions_diff'] = df['average_interceptions_diff'].astype(np.float64)
    df['average_fumbles_forced_diff'] = df['average_fumbles_forced_diff'].astype(np.float64)
    df['average_fumbles_recovered_diff'] = df['average_fumbles_recovered_diff'].astype(np.float64)
    df['average_turnovers_forced_diff'] = df['average_turnovers_forced_diff'].astype(np.float64)
    df['average_sacks_diff'] = df['average_sacks_diff'].astype(np.float64)
    df['average_sack_yards_forced_diff'] = df['average_sack_yards_forced_diff'].astype(np.float64)
    df['average_yards_per_rush_attempt_diff'] = df['average_yards_per_rush_attempt_diff'].astype(np.float64)
    df['average_rushing_touchdowns_per_attempt_diff'] = df['average_rushing_touchdowns_per_attempt_diff'].astype(
        np.float64)
    df['average_yards_per_pass_attempt_diff'] = df['average_yards_per_pass_attempt_diff'].astype(np.float64)
    df['average_passing_touchdowns_per_attempt_diff'] = df['average_passing_touchdowns_per_attempt_diff'].astype(
        np.float64)
    df['average_yards_per_pass_completion_diff'] = df['average_yards_per_pass_completion_diff'].astype(np.float64)
    df['average_rushing_play_pct_diff'] = df['average_rushing_play_pct_diff'].astype(np.float64)
    df['average_passing_play_pct_diff'] = df['average_passing_play_pct_diff'].astype(np.float64)
    df['average_rushing_yards_pct_diff'] = df['average_rushing_yards_pct_diff'].astype(np.float64)
    df['average_passing_yards_pct_diff'] = df['average_passing_yards_pct_diff'].astype(np.float64)
    df['average_completion_pct_diff'] = df['average_completion_pct_diff'].astype(np.float64)
    df['average_sacked_pct_diff'] = df['average_sacked_pct_diff'].astype(np.float64)
    df['average_passer_rating_diff'] = df['average_passer_rating_diff'].astype(np.float64)
    df['average_touchdowns_diff'] = df['average_touchdowns_diff'].astype(np.float64)
    df['average_yards_per_point_diff'] = df['average_yards_per_point_diff'].astype(np.float64)
    df['average_scoring_margin_diff'] = df['average_scoring_margin_diff'].astype(np.float64)
    df['average_turnover_margin_diff'] = df['average_turnover_margin_diff'].astype(np.float64)

    return df


def plot_corr(df=None):
    """Gets the correlation between all relevant features and the home_victory label."""
    if df is None:
        # Get the data frame for all seasons
        df = pd.read_csv(game_data_dir + '20022018.csv')

    # Print a description of all the games
    games_description = df.describe()
    games_description.to_csv(other_dir + '20022018Description.csv')
    print(games_description.to_string())
    print()

    # Drop all columns that arent the label, the spread or a team difference
    columns_to_keep = list()
    columns_to_keep.extend(list(filter(lambda f: 'home_victory' in f, df.columns.values)))
    columns_to_keep.extend(list(filter(lambda f: 'diff' in f, df.columns.values)))
    columns_to_keep.extend(list(filter(lambda f: 'home_spread' in f, df.columns.values)))
    columns_to_drop = list(set(df.columns.values) - set(columns_to_keep))
    df = df.drop(columns=columns_to_drop)

    df = df.drop('game_point_diff', axis=1)

    # Get the correlation values between all features
    corr = df.corr().abs()

    # Unstack them into a series of pairs
    pairs = corr.unstack()

    # Sort the pairs based on highest correlation
    sorted_pairs = pairs.sort_values(kind='quicksort', ascending=False)

    # Get all pairs with the home_victory label
    won_series = sorted_pairs['home_victory']

    # Remove the home_victory to home_victory correlation
    corr_vals = list(won_series.index)
    corr_vals.remove('home_victory')
    won_series = won_series.filter(corr_vals)

    # Print each features correlation to the home_victory label
    print('Features most correlated with a victory:')
    print(won_series)
    print()

    # Plot each features correlation with each other feature
    size = len(df.columns)
    pd.set_option('expand_frame_repr', False)
    pd.set_option("display.max_columns", size + 2)

    fig, ax = plt.subplots(figsize=(size, size))

    # Color code the rectangles by correlation value
    ax.matshow(corr)

    # Draw x tick marks
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.xticks(rotation=90)

    # Draw y tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

    # Return the series containing the sorted correlations between each feature and the home_victory label
    return won_series


def get_best_features(df=None):
    """Gets the set of at least 10 features that explains the variance for the y label."""
    if df is None:
        # Get the data frame for all seasons
        df = pd.read_csv(game_data_dir + '20022018.csv')

    # Drop all columns that arent the label, the spread or a team difference
    columns_to_keep = list()
    columns_to_keep.extend(list(filter(lambda f: 'home_victory' in f, df.columns.values)))
    columns_to_keep.extend(list(filter(lambda f: 'diff' in f, df.columns.values)))
    columns_to_keep.extend(list(filter(lambda f: 'home_spread' in f, df.columns.values)))
    columns_to_drop = list(set(df.columns.values) - set(columns_to_keep))
    df = df.drop(columns=columns_to_drop)

    df = df.drop('game_point_diff', axis=1)

    # Get the feature column names and predicted label name
    feature_col_names = list(set(df.columns.values) - {'home_victory'})
    predicted_class_name = ['home_victory']

    # Create data frames with the X and y values
    X = df[feature_col_names].values
    y = df[predicted_class_name].values

    # Fit the PCA
    n = .997
    pca = PCA(n_components=n).fit(X)

    # Plot the total percentage of variance explained by the number of features used
    ratios = pca.explained_variance_ratio_
    ratios = np.cumsum(ratios)
    plt.plot(ratios)
    plt.xlabel('Dimension')
    plt.ylabel('Ratio')
    plt.show()

    # Get the number of features required to explain the variance
    print('\n' + str(n * 100) + '% coverage: ' + str(len(pca.explained_variance_)) + ' features')
    num_comp = len(pca.explained_variance_)

    # Select the top features that explain the variance
    skb = SelectKBest(f_classif, k=num_comp)
    skb.fit(X, y.ravel())

    # Get the mask array
    features = list(skb.get_support())

    # For each feature in the list
    contributing_features = list()
    for i in range(0, len(features)):
        # If the feature is one of the top features, add it to the list
        if features[i]:
            contributing_features.append(feature_col_names[i])

    # Print the list of the top features
    print('The top ' + str(num_comp) + ' features are: ')
    for feature in contributing_features:
        print(feature)
    print()

    # Correlation matrix revealed high correlation between points for and touchdowns
    contributing_features.remove('average_points_for_diff')

    # Plot the correlation between the top 8 features
    columns_to_drop = list(set(feature_col_names) - set(contributing_features))
    relevant = df.drop(columns=columns_to_drop)
    corrmat = relevant.corr().abs()
    f, ax = plt.subplots(figsize=(9, 9))
    sns.set(font_scale=0.9)
    sns.heatmap(corrmat, vmax=.8, square=True, annot=True, fmt='.2f', cmap='winter')
    plt.show()

    # Check the distribution of each feature
    for feature in contributing_features:
        series = df[feature]
        (mu, sigma) = norm.fit(series)
        sns.distplot(series, fit=norm)
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title(feature + ' Distribution')

        print(feature)
        print('Skewness: {:.2f}'.format(series.skew()))
        print('Kurtosis: {:.2f}'.format(series.kurt()))

        print('mu: {:.2f}'.format(mu))
        print('sigma: {:.2f}'.format(sigma))
        print()

        # Get the QQ-plot
        fig = plt.figure()
        res = stats.probplot(series, plot=plt)
        plt.show()

    return contributing_features


def evaluate_model_parameters(contributing_features, df=None):
    """Does a grid search on 6 different models to find the best parameters,
    evaluates each set of parameters on brier loss score and accuracy."""
    if df is None:
        # Get the data frame for all seasons
        df = pd.read_csv(game_data_dir + '20022018.csv')

    # Filter the columns to keep
    columns_to_keep = list()
    for feature in contributing_features:
        columns_to_keep.extend(list(filter(lambda f: f == feature, df.columns.values)))

    # Drop all other columns (except the home_victory label)
    columns_to_keep.extend(list(filter(lambda f: 'home_victory' in f, df.columns.values)))
    columns_to_drop = list(set(df.columns.values) - set(columns_to_keep))
    df = df.drop(columns=columns_to_drop)

    # Get the X and y frames
    feature_col_names = contributing_features
    predicted_class_name = ['home_victory']

    # Print the feature column names by order of importance
    print()
    print('Top ' + str(len(feature_col_names)) + ' features ranked by importance are:')
    for feature in feature_col_names:
        print(feature)
    print()

    # Get the feature and label data sets
    X = df[feature_col_names].values
    y = df[predicted_class_name].values

    # Standardize the X values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Using Stratified K-Fold cross validation with 5 folds
    skf = StratifiedKFold(n_splits=5)

    # Tune the parameters to find the best models based on brier loss and accuracy
    scores = ['brier_score_loss', 'accuracy']

    # Logistic Regression               -0.21185    66.557
    tune_logistic_regression(X, y, skf, scores)

    # C Support Vector Classifier       -0.21214    66.424
    tune_svc_classifier(X, y, skf, scores)

    # Random Forest                     -0.21281    66.314
    tune_random_forest(X, y, feature_col_names, skf, scores)

    # K Nearest Neighbors               -0.21657    65.345
    tune_k_nearest_neighbors(X, y, skf, scores)

    # Bernoulli Naive Bayes             -0.22297    64.353
    tune_bernoulli_naive_bayes(X, y, skf, scores)

    # Gaussian Naive Bayes              -0.22866    61.974
    tune_gauss_naive_bayes(X, y, skf, scores)


def tune_logistic_regression(X, y, skf, scores):
    """Does a grid search over different parameters for a logistic regression model.
    Returns the model with the least brier loss and the most accurate model."""

    # Create a list of dicts to try parameter combinations over
    print('Logistic Regression')
    logistic_regression_parameters = [{'penalty': ['l2'],
                                       'tol': [1e-3, 1e-4, 1e-5],
                                       'C': [x / 100.0 for x in range(1, 20)],
                                       'solver': ['newton-cg', 'lbfgs', 'sag'],
                                       'max_iter': range(900, 1100, 10),
                                       'multi_class': ['ovr', 'multinomial'],
                                       'class_weight': [None, 'balanced'],
                                       'random_state': [42]},
                                      {'penalty': ['l1', 'l2'],
                                       'tol': [1e-3, 1e-4, 1e-5],
                                       'C': [x / 100.0 for x in range(1, 20)],
                                       'solver': ['saga'],
                                       'multi_class': ['ovr', 'multinomial'],
                                       'class_weight': [None, 'balanced'],
                                       'random_state': [42]},
                                      {'penalty': ['l1'],
                                       'tol': [1e-3, 1e-4, 1e-5],
                                       'C': [x / 100.0 for x in range(1, 20)],
                                       'solver': ['liblinear'],
                                       'multi_class': ['auto'],
                                       'class_weight': [None, 'balanced'],
                                       'random_state': [42]}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(LogisticRegression(),
                           logistic_regression_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           verbose=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'logistic_regression_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def tune_gauss_naive_bayes(X, y, skf, scores):
    """Does a grid search over different parameters for a gaussian naive bayes model.
    Returns the model with the least brier loss and the most accurate model."""

    # Create a list of dicts to try parameter combinations over
    print('Gaussian Naive Bayes')
    naive_bayes_parameters = [{'var_smoothing': [1 * 10 ** x for x in range(3, -20, -1)]}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(GaussianNB(),
                           naive_bayes_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           verbose=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'gaussian_naive_bayes_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def tune_bernoulli_naive_bayes(X, y, skf, scores):
    """Does a grid search over different parameters for a bernoulli naive bayes model.
    Returns the model with the least brier loss and the most accurate model."""

    # Create a list of dicts to try parameter combinations over
    print('Bernoulli Naive Bayes')
    naive_bayes_parameters = [{'alpha': [x for x in range(5, 20000, 10)],
                               'fit_prior': [True, False]}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(BernoulliNB(),
                           naive_bayes_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           verbose=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'bernoulli_naive_bayes_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def tune_random_forest(X, y, feature_col_names, skf, scores):
    """Does a grid search over different parameters for a random forest model.
    Returns the model with the least brier loss and the most accurate model."""

    # Create a list of dicts to try parameter combinations over
    print('Random Forest')
    random_forest_parameters = [{'n_estimators': [500, 1000],
                                 'random_state': [42],
                                 'max_features': range(1, len(feature_col_names) + 1),
                                 'max_depth': range(1, 20)},
                                {'n_estimators': [500, 1000],
                                 'random_state': [42],
                                 'max_features': range(1, len(feature_col_names) + 1),
                                 'min_samples_split': range(2, 101)},
                                {'n_estimators': [500, 1000],
                                 'random_state': [42],
                                 'max_features': range(1, len(feature_col_names) + 1),
                                 'min_samples_leaf': range(1, 101)}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(RandomForestClassifier(),
                           random_forest_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           verbose=2,
                           n_jobs=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'random_forest_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def tune_k_nearest_neighbors(X, y, skf, scores):
    """Does a grid search over different parameters for a K nearest neighbors model.
    Returns the model with the least brier loss and the most accurate model."""

    # Create a list of dicts to try parameter combinations over
    print('K Nearest Neighbors')
    k_neighbors_parameters = [{'n_neighbors': range(3, 71, 2),
                               'weights': ['uniform', 'distance'],
                               'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                               'leaf_size': range(1, 31),
                               'p': [1, 2]}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(KNeighborsClassifier(),
                           k_neighbors_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           verbose=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'knn_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def tune_svc_classifier(X, y, skf, scores):
    """Does a grid search over different parameters for a C support vector classification model.
    Returns the model with the least brier loss and the most accurate model."""

    # Create a list of dicts to try parameter combinations over
    print('C Support Vector Classifier')
    support_vector_parameters = [{'kernel': ['rbf', 'sigmoid'],
                                  'probability': [True],
                                  'C': [1 * 10 ** x for x in range(-1, 3)],
                                  'gamma': [1 * 10 ** x for x in range(-3, -8, -1)]},
                                 {'kernel': ['rbf', 'sigmoid'],
                                  'probability': [True],
                                  'C': [1 * 10 ** x for x in range(-1, 3)],
                                  'gamma': ['auto', 'scale']},
                                 {'kernel': ['linear'],
                                  'probability': [True],
                                  'C': [1 * 10 ** x for x in range(-1, 3)]}]

    best_brier_model = None
    best_accuracy_model = None

    # For each scoring method
    for score in scores:
        # State the scoring method
        print('# Tuning parameters for %s' % score)

        # Log the start time
        start = maya.now()
        print('Started at: ' + str(start))

        # Do a grid search over all parameter combinations
        clf = GridSearchCV(SVC(),
                           support_vector_parameters,
                           cv=skf,
                           scoring='%s' % score,
                           verbose=2,
                           n_jobs=2)
        clf.fit(X, y.ravel())

        # Print the results of the grid search
        print_grid_search_details(clf, 'svc_' + score + '.txt')

        # Get the best model for each scoring method
        if score == 'brier_score_loss':
            best_brier_model = clf.best_estimator_
        else:
            best_accuracy_model = clf.best_estimator_

    return best_brier_model, best_accuracy_model


def print_grid_search_details(clf, filename):
    """Prints the results of the grid search to a file and the console."""

    # Set the directory to write files to
    filename = other_dir + '7 Features\\Scores\\' + filename

    # Print the best parameter found in the search along with its score
    print('Best parameters set found on development set:')
    print('Best parameters set found on development set:', file=open(filename, 'a'))

    print('%0.5f (+/-%0.03f) for %r' % (clf.best_score_,
                                        clf.cv_results_['std_test_score'][clf.best_index_] * 2,
                                        clf.best_params_))
    print('%0.5f (+/-%0.03f) for %r' % (clf.best_score_,
                                        clf.cv_results_['std_test_score'][clf.best_index_] * 2,
                                        clf.best_params_), file=open(filename, 'a'))

    # Print the results of all parameter combinations, sorted from best score to worst
    print('\nGrid scores on development set:')
    print('\nGrid scores on development set:', file=open(filename, 'a'))

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    results = list(zip(means, stds, clf.cv_results_['params'], clf.cv_results_['rank_test_score']))
    for mean, std, params, rank in sorted(results, key=lambda tup: tup[3]):
        print('%0.5f (+/-%0.03f) for %r' % (mean, std * 2, params))
        print('%0.5f (+/-%0.03f) for %r' % (mean, std * 2, params), file=open(filename, 'a'))

    print()
    print('', file=open(filename, 'a'))


def get_best_knn():
    """Gets the SVC model that yielded the best result."""

    return KNeighborsClassifier(algorithm='kd_tree',
                                leaf_size=30,
                                n_neighbors=63,
                                p=1,
                                weights='uniform')


def get_best_logistic_regression():
    """Gets the logistic regression model that yielded the best result."""

    return LogisticRegression(C=0.04,
                              class_weight=None,
                              multi_class='ovr',
                              penalty='l1',
                              random_state=42,
                              solver='saga',
                              tol=0.0001)


def get_best_svc():
    """Gets the SVC model that yielded the best result."""

    return SVC(C=10,
               gamma=0.001,
               kernel='sigmoid',
               probability=True)


def get_best_random_forest():
    """Gets the random forest model that yielded the best result."""

    return RandomForestClassifier(n_estimators=500,
                                  max_features=5,
                                  max_depth=3,
                                  random_state=42)


def get_voting_classifier(contributing_features, df=None):
    """Creates a voting classifier based on the top 3 estimators,
    estimators are weighted by the normalized inverse of their respective briers."""

    if df is None:
        # Get the data frame for all seasons
        df = pd.read_csv(game_data_dir + '20022018.csv')

    # Drop all columns except for the most important features, and the predicted label
    columns_to_keep = list()
    for feature in contributing_features:
        columns_to_keep.extend(list(filter(lambda f: f == feature, df.columns.values)))

    columns_to_keep.extend(list(filter(lambda f: 'home_victory' in f, df.columns.values)))
    columns_to_drop = list(set(df.columns.values) - set(columns_to_keep))
    df = df.drop(columns=columns_to_drop)

    # Get a list of all the feature names
    feature_col_names = contributing_features
    predicted_class_name = ['home_victory']

    # Print the feature column names by order of importance
    print()
    print('Top ' + str(len(feature_col_names)) + ' features ranked by importance are:')
    for feature in feature_col_names:
        print(feature)

    # Get the feature and label data sets
    X = df[feature_col_names].values
    y = df[predicted_class_name].values

    # Standardize the X values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Pickle the scaler
    joblib.dump(scaler, other_dir + '7 Features\\2018Scaler.pkl')

    # Get the classification models
    logistic_regression = get_best_logistic_regression()
    svc = get_best_svc()
    random_forest = get_best_random_forest()

    # Create voting classifier from the 3 estimators, weighted by unit vector of the inverse of the briers, soft voting
    voting_classifier = VotingClassifier(estimators=[('Logistic Regression', logistic_regression),
                                                     ('SVC', svc),
                                                     ('Random Forest', random_forest)],
                                         weights=[0.578447, 0.577656, 0.575945],
                                         voting='soft',
                                         flatten_transform=False)

    # Fit the voting classifier
    voting_classifier.fit(X, y.ravel())

    # Pickle the voting classifier
    joblib.dump(voting_classifier, other_dir + '7 Features\\2018VotingClassifier.pkl')

    return voting_classifier


def evaluate_2018_season():
    # Set the directory to write files to
    filename = other_dir + '7 Features\\Scores\\2018Confusion.txt'

    voting_classifier = joblib.load(other_dir + '7 Features\\2017VotingClassifier.pkl')
    scaler = joblib.load(other_dir + '7 Features\\2017Scaler.pkl')

    last_season = pd.read_csv(game_data_dir + '20022018.csv').values[-267:]
    last_season = pd.DataFrame(last_season)

    results = pd.DataFrame(columns=['rf_prob', 'svc_prob', 'lr_prob', 'vote_prob', 'outcome'])
    for game in last_season.values:
        home_victory = game[71]
        home_spread = game[7]
        elo_diff = game[240]
        average_scoring_margin_diff = game[285]
        win_pct_diff = game[239]
        average_touchdowns_diff = game[283]
        average_passer_rating_diff = game[282]
        average_total_yards_diff = game[255]

        game_features = (home_spread,
                         elo_diff,
                         average_scoring_margin_diff,
                         win_pct_diff,
                         average_touchdowns_diff,
                         average_passer_rating_diff,
                         average_total_yards_diff)

        # Convert the features to a data frame and scale it
        game = pd.DataFrame([game_features])
        game = scaler.transform(game)

        # Get the voting classifier probability
        vote_prob = voting_classifier.predict_proba(game)[0][1]

        # Get the individual estimator probabilities
        estimator_probs = voting_classifier.transform(game)
        lr_prob = estimator_probs[0][0][1]
        svc_prob = estimator_probs[1][0][1]
        rf_prob = estimator_probs[2][0][1]

        game_df = pd.DataFrame([[rf_prob, svc_prob, lr_prob, vote_prob, home_victory]],
                               columns=['rf_prob', 'svc_prob', 'lr_prob', 'vote_prob', 'outcome'])

        results = results.append(game_df)

    outcome = results['outcome']
    rf = results['rf_prob']
    svc = results['svc_prob']
    lr = results['lr_prob']
    vote = results['vote_prob']

    rf_brier = brier_score_loss(outcome, rf)
    print('Random Forest Brier Score Loss:', round(rf_brier, 4))
    print('Random Forest Brier Score Loss:', round(rf_brier, 4), file=open(filename, 'a'))

    svc_brier = brier_score_loss(outcome, svc)
    print('SVC Brier Score Loss:', round(svc_brier, 4))
    print('SVC Brier Score Loss:', round(svc_brier, 4), file=open(filename, 'a'))

    lr_brier = brier_score_loss(outcome, lr)
    print('Logistic Regression Brier Score Loss:', round(lr_brier, 4))
    print('Logistic Regression Brier Score Loss:', round(lr_brier, 4), file=open(filename, 'a'))

    vote_brier = brier_score_loss(outcome, vote)
    print('Voting Classifier Brier Score Loss:', round(vote_brier, 4))
    print('Voting Classifier Brier Score Loss:', round(vote_brier, 4), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    print('Random Forest:', round((.25 - rf_brier) * 26700, 2))
    print('SVC:', round((.25 - svc_brier) * 26700, 2))
    print('Logistic Regression:', round((.25 - lr_brier) * 26700, 2))
    print('Voting Classifier:', round((.25 - vote_brier) * 26700, 2))

    print('Random Forest:', round((.25 - rf_brier) * 26700, 2), file=open(filename, 'a'))
    print('SVC:', round((.25 - svc_brier) * 26700, 2), file=open(filename, 'a'))
    print('Logistic Regression:', round((.25 - lr_brier) * 26700, 2), file=open(filename, 'a'))
    print('Voting Classifier:', round((.25 - vote_brier) * 26700, 2), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    # results.to_csv(other_dir + '7 Features\\Scores\\2018Predictions.csv', index=False)

    rounded_rf = rf.apply(lambda row: round(row))
    rounded_svc = svc.apply(lambda row: round(row))
    rounded_lr = lr.apply(lambda row: round(row))
    rounded_vote = vote.apply(lambda row: round(row))

    print()
    print('Random Forest')
    print('-' * 120)
    print('Random Forest', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_rf, filename)

    print('SVC')
    print('-' * 120)
    print('SVC', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_svc, filename)

    print('Linear Regression')
    print('-' * 120)
    print('Linear Regression', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_lr, filename)

    print('Voting Classifier')
    print('-' * 120)
    print('Voting Classifier', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_vote, filename)

    # visualize_2018_season()


def get_metrics(y_true, y_pred, filename):
    y_true = pd.to_numeric(y_true)
    outcome_counts = y_true.value_counts()
    outcome_positive = outcome_counts.loc[1]
    outcome_negative = outcome_counts.loc[0]
    total_games = outcome_positive + outcome_negative
    print('Actual home victories:', outcome_positive)
    print('Actual home defeats:', outcome_negative)
    print('Actual home victories:', outcome_positive, file=open(filename, 'a'))
    print('Actual home defeats:', outcome_negative, file=open(filename, 'a'))

    prevalence = outcome_positive / (outcome_positive + outcome_negative)
    print('Home victory prevalence:', round(prevalence * 100, 2),
          str(outcome_positive) + '/' + str(outcome_positive + outcome_negative))
    print()
    print('Home victory prevalence:', round(prevalence * 100, 2),
          str(outcome_positive) + '/' + str(outcome_positive + outcome_negative), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    predicted_counts = y_pred.value_counts()
    predicted_positive = predicted_counts.loc[1]
    predicted_negative = predicted_counts.loc[0]
    print('Predicted home victories:', predicted_positive)
    print('Predicted home defeats:', predicted_negative)
    print('Predicted home victories:', predicted_positive, file=open(filename, 'a'))
    print('Predicted home defeats:', predicted_negative, file=open(filename, 'a'))
    print()
    print('', file=open(filename, 'a'))

    confusion = confusion_matrix(y_true, y_pred)
    true_positive = confusion[1][1]
    false_positive = confusion[0][1]
    false_negative = confusion[1][0]
    true_negative = confusion[0][0]

    print('Correctly predicted home victories:', true_positive)
    print('Home victories predicted as defeats:', false_negative)
    print('Correctly predicted home defeats:', true_negative)
    print('Home defeats predicted as victories:', false_positive)
    print()

    print('Correctly predicted home victories:', true_positive, file=open(filename, 'a'))
    print('Home victories predicted as defeats:', false_negative, file=open(filename, 'a'))
    print('Correctly predicted home defeats:', true_negative, file=open(filename, 'a'))
    print('Home defeats predicted as victories:', false_positive, file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print('Prediction accuracy:', round(accuracy * 100, 2),
          str(true_positive + true_negative) + '/' + str(total_games))
    print()

    print('Prediction precision:', round(precision * 100, 2),
          str(true_positive) + '/' + str(predicted_positive))

    print('Prediction accuracy:', round(accuracy * 100, 2),
          str(true_positive + true_negative) + '/' + str(total_games), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    print('Prediction precision:', round(precision * 100, 2),
          str(true_positive) + '/' + str(predicted_positive), file=open(filename, 'a'))

    false_discovery = false_positive / predicted_positive
    print('False discovery rate:', round(false_discovery * 100, 2),
          str(false_positive) + '/' + str(predicted_positive))

    print('False discovery rate:', round(false_discovery * 100, 2),
          str(false_positive) + '/' + str(predicted_positive), file=open(filename, 'a'))

    negative_prediction = true_negative / predicted_negative
    print('Negative prediction rate:', round(negative_prediction * 100, 2),
          str(true_negative) + '/' + str(predicted_negative))

    print('Negative prediction rate:', round(negative_prediction * 100, 2),
          str(true_negative) + '/' + str(predicted_negative), file=open(filename, 'a'))

    false_omission = false_negative / predicted_negative
    print('False omission rate:', round(false_omission * 100, 2),
          str(false_negative) + '/' + str(predicted_negative))
    print()

    print('Prediction recall:', round(recall * 100, 2),
          str(true_positive) + '/' + str(outcome_positive))

    print('False omission rate:', round(false_omission * 100, 2),
          str(false_negative) + '/' + str(predicted_negative), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    print('Prediction recall:', round(recall * 100, 2),
          str(true_positive) + '/' + str(outcome_positive), file=open(filename, 'a'))

    miss_rate = false_negative / outcome_positive
    print('Miss rate:', round(miss_rate * 100, 2),
          str(false_negative) + '/' + str(outcome_positive))

    print('Miss rate:', round(miss_rate * 100, 2),
          str(false_negative) + '/' + str(outcome_positive), file=open(filename, 'a'))

    specificity = true_negative / outcome_negative
    print('Specificity:', round(specificity * 100, 2),
          str(true_negative) + '/' + str(outcome_negative))

    print('Specificity:', round(specificity * 100, 2),
          str(true_negative) + '/' + str(outcome_negative), file=open(filename, 'a'))

    fall_out = false_positive / outcome_negative
    print('Fall out:', round(fall_out * 100, 2),
          str(false_positive) + '/' + str(outcome_negative))
    print()

    print('Fall out:', round(fall_out * 100, 2),
          str(false_positive) + '/' + str(outcome_negative), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    positive_likelihood = recall / fall_out
    print('Positive likelihood ratio:', round(positive_likelihood, 4),
          str(round(recall * 100, 2)) + '/' + str(round(fall_out * 100, 2)))

    print('Positive likelihood ratio:', round(positive_likelihood, 4),
          str(round(recall * 100, 2)) + '/' + str(round(fall_out * 100, 2)), file=open(filename, 'a'))

    negative_likelihood = miss_rate / specificity
    print('Negative likelihood ratio:', round(negative_likelihood, 4),
          str(round(miss_rate * 100, 2)) + '/' + str(round(specificity * 100, 2)))
    print()

    print('Negative likelihood ratio:', round(negative_likelihood, 4),
          str(round(miss_rate * 100, 2)) + '/' + str(round(specificity * 100, 2)), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    diagnostic_odds = positive_likelihood / negative_likelihood
    print('Diagnostic odds ratio:', round(diagnostic_odds, 4),
          str(round(positive_likelihood, 4)) + '/' + str(round(negative_likelihood, 4)))

    print('Diagnostic odds ratio:', round(diagnostic_odds, 4),
          str(round(positive_likelihood, 4)) + '/' + str(round(negative_likelihood, 4)), file=open(filename, 'a'))

    f1 = f1_score(y_true, y_pred)
    print('F1 score:', round(f1, 4))
    print()
    print('F1 score:', round(f1, 4), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))


def visualize_2018_season():
    predictions = pd.read_csv(other_dir + '7 Features\\Scores\\2018Predictions.csv')

    for num, game in enumerate(predictions.values):
        vote_prob = int(round(game[3] * 100))
        home_victory = int(round(game[4] * 100))
        if home_victory == 0:
            vote_prob = 100 - vote_prob
            home_victory = 100

        brier = round((game[3] - game[4]) ** 2, 2)

        slider = ['.' for pct in range(100)]

        slider[49] = slider[49].replace('.', '') + '|'
        slider[vote_prob - 1] = slider[vote_prob - 1].replace('.', '') + 'V'
        slider[home_victory - 1] = slider[home_victory - 1].replace('.', '') + 'X'

        if vote_prob < 50:
            print('\033[31m' + str(num + 2).zfill(3) + ' ' + ''.join(slider) + ' ' + str(brier) + '\033[0m')
        else:
            print('\033[32m' + str(num + 2).zfill(3) + ' ' + ''.join(slider) + ' ' + str(brier) + '\033[0m')
