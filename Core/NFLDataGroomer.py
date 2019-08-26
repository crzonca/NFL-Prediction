import os
import statistics

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss
from sklearn.utils import check_consistent_length

from Projects.nfl.NFL_Prediction.Core import StatsHelper as Stats

game_data_dir = '..\\Projects\\nfl\\NFL_Prediction\\Game Data\\'
other_dir = '..\\Projects\\nfl\\NFL_Prediction\\Other\\'


def groom_data():
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
    all_games = remove_outliers(all_games)
    return all_games


def get_all_data_frames():
    """
    Gets all of the csv files in the directory and converts them to pandas DataFrames.

    :return: A list of data frames, one for each season
    """

    frames = list()

    for file in os.listdir(game_data_dir):
        if (os.path.splitext(file)[1] == '.csv'
                and os.path.splitext(file)[0] != '20022018'
                and os.path.splitext(file)[0] != '19952018'
                and os.path.splitext(file)[0] != '20022017'):
            with open(game_data_dir + file, 'r') as season_csv:
                df = pd.read_csv(season_csv, encoding='utf-8')
                df = df.rename(index=str, columns={'ï»¿home_team': 'home_team'})
                frames.append(df)

    return frames


def add_point_diff_and_results(frames):
    """
    Adds the point differential for each game as well as columns indicating if the home team won or tied.

    :param frames: A list of data frames containing the game info, one for each season
    :return: An updated list of data frames with point differential and results, one for each season
    """

    for df in frames:
        df['game_point_diff'] = df.apply(lambda row: Stats.game_point_diff(row), axis=1)
        df['home_victory'] = df.apply(lambda row: Stats.home_victory(row), axis=1)
        df['home_draw'] = df.apply(lambda row: Stats.home_draw(row), axis=1)

    return frames


def add_team_records(frames):
    """
    Adds columns for the record of each team going into the game. Adds a column for the win percentage of each team
    going into the game.

    :param frames: A list of data frames containing the game info, one for each season
    :return: An updated list of data frames with records and win percentages, one for each season
    """

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


def add_team_elos(frames, regression_factor=.44, k_factor=42):
    """
    Adds columns for the Elo for each team going into the game.
    Elo is calculated from the prior Elo of each team and the result of the game.
    If the team does not have an Elo, it is set to a default value of 1500.
    Team Elos are regressed towards 1500 at the end of each season.
    https://en.wikipedia.org/wiki/Elo_rating_system

    :param frames: A list of data frames containing the game info, one for each season
    :param regression_factor: The percent to regress each teams elo towards the mean by
    :param k_factor: The K-factor used in the Elo function calculation
    :return: An updated list of data frames with team Elos, one for each season
    """

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
    """
    Calculates the new Elos of each team given their previous Elos and the outcome of the game.

    :param home_elo: The elo of the home team
    :param away_elo: The elo of the away team
    :param home_victory: If the home team was victorious
    :param home_draw: If the game resulted in a draw
    :param k_factor: The K-factor used in the Elo function calculation
    :return: The updated elo for each team, based of the game outcome
    """

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
    """
    Evaluates the accuracy of the parameters of the Elo function for a given season based off the brier loss function.
    Returns the brier loss score for that season.

    :param df: A data frame containing the game info for a season
    :return: The loss of the elo predictions, using the brier loss function
    """

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
    """
    Evaluates the accuracy of all regression factors and K factors of the Elo function for averaged over all seasons.
    Returns the best combination of regression factor and K factor.

    :param frames: A list of data frames containing the game info, one for each season
    :return: The regression factor and K factor that yeild the least loss
    """

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
    """
    Adds a running total for each stat for each team prior to the game.

    :param frames: A list of data frames containing the game info, one for each season
    :return: An updated list of data frames with running totals for all stats, one for each season
    """

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
    """
    Adds a running average for each stat for each team prior to the game.

    :param frames: A list of data frames containing the game info, one for each season
    :return: An updated list of data frames with running averages for all stats, one for each season
    """

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
    """
    Adds additional stats for each team prior to the game.

    :param frames: A list of data frames containing the game info, one for each season
    :return: An updated list of data frames with additional stats for each game, one for each season
    """

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
    """
    Calculates the margin between the home and away teams for the averages and advanced stats columns.
    Differences are from the home team's perspective (home_stat - away_stat).

    :param frames: A list of data frames containing the game info, one for each season
    :return: An updated list of data frames with stat differences for each game, one for each season
    """

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
    """
    Combines all of the season data frames into a total frame and writes it to a csv file.

    :param frames:  A list of data frames containing the game info, one for each season
    :return: One data frame containing the game info for all games since 1995
    """

    # Check that all seasons have the same number of columns
    check_consistent_length(*[frame.T for frame in frames])

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

    combined.to_csv(game_data_dir + '19952018.csv', index=False)

    return combined


def remove_outliers(df):
    relevant_columns = list()
    relevant_columns.extend(list(filter(lambda f: 'home_victory' in f, df.columns.values)))
    relevant_columns.extend(list(filter(lambda f: 'diff' in f, df.columns.values)))
    relevant_columns.extend(list(filter(lambda f: 'home_spread' in f, df.columns.values)))

    # Check the distribution of each feature
    for feature in relevant_columns:
        series = df[feature]
        (mean, dev) = norm.fit(series)

        print(feature)
        print('Skewness: {:.3f}'.format(series.skew()))
        print('Kurtosis: {:.3f}'.format(series.kurt()))

        print('Mean: {:.3f}'.format(mean))
        print('Std. Dev.: {:.3f}'.format(dev))
        print()

    # Remove extreme outliers
    print('Removing outliers')
    for feature in relevant_columns:
        series = df[feature]
        (mean, dev) = norm.fit(series)
        df = df.loc[df[feature] < mean + 4 * dev]
        df = df.loc[df[feature] > mean - 4 * dev]

    # Check the distribution of each feature
    for feature in relevant_columns:
        series = df[feature]
        (mean, dev) = norm.fit(series)

        print(feature)
        print('Skewness: {:.3f}'.format(series.skew()))
        print('Kurtosis: {:.3f}'.format(series.kurt()))

        print('Mean: {:.3f}'.format(mean))
        print('Std. Dev.: {:.3f}'.format(dev))
        print()

    return df


def get_all_games_no_outliers():
    all_games = pd.read_csv(game_data_dir + '19952018.csv')
    all_games = remove_outliers(all_games)
    return all_games
