import warnings

import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.externals import joblib

import Projects.nfl.NFL_Prediction.Core.NFLDataGroomer as NFL

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
base_dir = '..\\Projects\\nfl\\NFL_Prediction\\'


def predict_game(home_info, away_info, home_spread=0):
    """
    Predicts the percent chance of the outcome of the game.
    
    :param home_info: Stats for the home team
    :param away_info: Stats for the away team
    :param home_spread: The spread of the game, from the home team's perspective
    :return: The voting classifier probability and the individual estimator probabilities
    """

    # Load the classifier and scaler
    voting_classifier = joblib.load(base_dir + 'Other\\7 Features\\2018VotingClassifier.pkl')
    scaler = joblib.load(base_dir + 'Other\\7 Features\\2018Scaler.pkl')

    # Get the home team's info
    home_wins = home_info[1]
    home_losses = home_info[2]
    home_ties = home_info[3]
    home_elo = home_info[4]
    home_average_points_for = home_info[5]
    home_average_points_against = home_info[6]
    home_average_tds = home_info[7]
    home_total_net_pass_yards = home_info[8]
    home_total_pass_completions = home_info[9]
    home_total_pass_attempts = home_info[10]
    home_total_pass_tds = home_info[11]
    home_total_interceptions_thrown = home_info[12]
    home_average_total_yards = home_info[13]

    # Get the away team's info
    away_wins = away_info[1]
    away_losses = away_info[2]
    away_ties = away_info[3]
    away_elo = away_info[4]
    away_average_points_for = away_info[5]
    away_average_points_against = away_info[6]
    away_average_tds = away_info[7]
    away_total_net_pass_yards = away_info[8]
    away_total_pass_completions = away_info[9]
    away_total_pass_attempts = away_info[10]
    away_total_pass_tds = away_info[11]
    away_total_interceptions_thrown = away_info[12]
    away_average_total_yards = away_info[13]

    # Calculate the features for the prediction
    home_games_played = home_wins + home_losses + home_ties
    away_games_played = away_wins + away_losses + away_ties

    elo_diff = home_elo - away_elo

    home_average_scoring_margin = home_average_points_for - home_average_points_against
    away_average_scoring_margin = away_average_points_for - away_average_points_against
    average_scoring_margin_diff = home_average_scoring_margin - away_average_scoring_margin

    home_win_pct = home_wins / home_games_played if home_games_played > 0 else 0
    away_win_pct = away_wins / away_games_played if away_games_played > 0 else 0
    win_pct_diff = home_win_pct - away_win_pct

    average_touchdowns_diff = home_average_tds - away_average_tds

    home_average_completion_pct = home_total_pass_completions / home_total_pass_attempts \
        if home_total_pass_attempts > 0 else 0
    home_a = (home_average_completion_pct - .3) * 5

    home_average_yards_per_pass_attempt = home_total_net_pass_yards / home_total_pass_attempts \
        if home_total_pass_attempts > 0 else 0
    home_b = (home_average_yards_per_pass_attempt - 3) * .25

    home_average_passing_touchdowns_per_attempt = home_total_pass_tds / home_total_pass_attempts \
        if home_total_pass_attempts > 0 else 0
    home_c = home_average_passing_touchdowns_per_attempt * 20

    home_average_ints_per_attempt = home_total_interceptions_thrown / home_total_pass_attempts \
        if home_total_pass_attempts > 0 else 0
    home_d = 2.375 - (home_average_ints_per_attempt * 25)

    home_average_passer_rating = ((home_a + home_b + home_c + home_d) / 6) * 100

    away_average_completion_pct = away_total_pass_completions / away_total_pass_attempts \
        if away_total_pass_attempts > 0 else 0
    away_a = (away_average_completion_pct - .3) * 5

    away_average_yards_per_pass_attempt = away_total_net_pass_yards / away_total_pass_attempts \
        if away_total_pass_attempts > 0 else 0
    away_b = (away_average_yards_per_pass_attempt - 3) * .25

    away_average_passing_touchdowns_per_attempt = away_total_pass_tds / away_total_pass_attempts \
        if away_total_pass_attempts > 0 else 0
    away_c = away_average_passing_touchdowns_per_attempt * 20

    away_average_ints_per_attempt = away_total_interceptions_thrown / away_total_pass_attempts \
        if away_total_pass_attempts > 0 else 0
    away_d = 2.375 - (away_average_ints_per_attempt * 25)

    away_average_passer_rating = ((away_a + away_b + away_c + away_d) / 6) * 100

    average_passer_rating_diff = home_average_passer_rating - away_average_passer_rating

    average_total_yards_diff = home_average_total_yards - away_average_total_yards

    # Organize the features
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
    vote_prob = voting_classifier.predict_proba(game)[0]

    # Get the individual estimator probabilities
    estimator_probs = voting_classifier.transform(game)
    lr_prob = estimator_probs[0][0]
    svc_prob = estimator_probs[1][0]
    rf_prob = estimator_probs[2][0]

    return vote_prob, lr_prob, svc_prob, rf_prob


def predict_home_victory(home_info, away_info, home_spread=0):
    """
    Predicts the percent chance of the home team defeating the away team.
    
    :param home_info: Stats for the home team
    :param away_info: Stats for the away team
    :param home_spread: The spread of the game, from the home team's perspective
    :return: The voting classifier probability and the individual estimator probabilities for the home team
    """

    vote_prob, lr_prob, svc_prob, rf_prob = predict_game(home_info, away_info, home_spread)
    return vote_prob[1], lr_prob[1], svc_prob[1], rf_prob[1]


def predict_away_victory(home_info, away_info, home_spread=0):
    """
    Predicts the percent chance of the away team defeating the home team.

    :param home_info: Stats for the home team
    :param away_info: Stats for the away team
    :param home_spread: The spread of the game, from the home team's perspective
    :return: The voting classifier probability and the individual estimator probabilities for the away team
    """

    vote_prob, lr_prob, svc_prob, rf_prob = predict_game(home_info, away_info, home_spread)
    return vote_prob[0], lr_prob[0], svc_prob[0], rf_prob[0]


def predict_game_outcome(teams, home_name, away_name, home_spread, neutral_location=False, verbose=False):
    """
    Predicts the percent chance of the outcome of the game.  Factors in the possibility of a neutral location game.

    :param teams: A list of all the teams in the league
    :param home_name: The name of the home team
    :param away_name: The name of the away team
    :param home_spread: The spread of the game, from the home team's perspective
    :param neutral_location: If the game is at a neutral location
    :param verbose: If verbose output should be included
    :return: The favored teams chance of winning and a message
    """

    # Get each team
    home = get_team(teams, home_name)
    away = get_team(teams, away_name)

    # Get each teams probability of victory according to each estimator
    vote_probs, lr_probs, svc_probs, rf_probs = predict_game(home, away, home_spread)

    home_vote_prob = vote_probs[1]
    home_lr_prob = lr_probs[1]
    home_svc_prob = svc_probs[1]
    home_rf_prob = rf_probs[1]

    away_vote_prob = vote_probs[0]
    away_lr_prob = lr_probs[0]
    away_svc_prob = svc_probs[0]
    away_rf_prob = rf_probs[0]

    if neutral_location:
        vote_probs, lr_probs, svc_probs, rf_probs = predict_game(away, home, -home_spread)

        home_vote_prob = (home_vote_prob + vote_probs[0]) / 2
        home_lr_prob = (home_lr_prob + lr_probs[0]) / 2
        home_svc_prob = (home_svc_prob + svc_probs[0]) / 2
        home_rf_prob = (home_rf_prob + rf_probs[0]) / 2

        away_vote_prob = (away_vote_prob + vote_probs[1]) / 2
        away_lr_prob = (away_lr_prob + lr_probs[1]) / 2
        away_svc_prob = (away_svc_prob + svc_probs[1]) / 2
        away_rf_prob = (away_rf_prob + rf_probs[1]) / 2

    # Write the favored teams chance of winning
    home_vote_prob_formatted = round(home_vote_prob * 100, 2)
    away_vote_prob_formatted = round(away_vote_prob * 100, 2)
    if home_vote_prob >= away_vote_prob:
        message = 'The ' + home_name + ' have a ' + str(home_vote_prob_formatted) + '% chance to beat the ' + away_name
    else:
        message = 'The ' + away_name + ' have a ' + str(away_vote_prob_formatted) + '% chance to beat the ' + home_name

    # If verbose output
    if verbose:
        # Print each teams probability of winning according to each estimator
        home_lr_prob_formatted = round(home_lr_prob * 100, 2)
        home_svc_prob_formatted = round(home_svc_prob * 100, 2)
        home_rf_prob_formatted = round(home_rf_prob * 100, 2)
        away_lr_prob_formatted = round(away_lr_prob * 100, 2)
        away_svc_prob_formatted = round(away_svc_prob * 100, 2)
        away_rf_prob_formatted = round(away_rf_prob * 100, 2)

        print('*' * 120, '\n')

        home_spread = round(home_spread * 2) / 2

        if home_spread > 0:
            print('>> The ' + away_name + ' are favored to beat the ' + home_name + ' by ' +
                  str(home_spread) + ' points')
        elif home_spread < 0:
            print('>> The ' + home_name + ' are favored to beat the ' + away_name + ' by ' +
                  str(-home_spread) + ' points')
        else:
            print('>> The spread is even')

        print('>> ' + message)
        print()
        print('>> Logistic Regression ' + home_name + ' Victory Probability: ' + str(home_lr_prob_formatted) + '%')
        print('>> Logistic Regression ' + away_name + ' Victory Probability: ' + str(away_lr_prob_formatted) + '%')
        print()
        print('>> SVC ' + home_name + ' Victory Probability: ' + str(home_svc_prob_formatted) + '%')
        print('>> SVC ' + away_name + ' Victory Probability: ' + str(away_svc_prob_formatted) + '%')
        print()
        print('>> Random Forest ' + home_name + ' Victory Probability: ' + str(home_rf_prob_formatted) + '%')
        print('>> Random Forest ' + away_name + ' Victory Probability: ' + str(away_rf_prob_formatted) + '%')
        print()

    # Return the favored teams chance of winning and a message
    return home_vote_prob if home_vote_prob >= away_vote_prob else away_vote_prob, message


def update_teams(teams, home_name, home_score, home_touchdowns, home_net_pass_yards, home_pass_completions,
                 home_pass_attempts, home_pass_tds, home_interceptions_thrown, home_total_yards,
                 away_name, away_score, away_touchdowns, away_net_pass_yards, away_pass_completions, away_pass_attempts,
                 away_pass_tds, away_interceptions_thrown, away_total_yards):
    """
    Updates each team with the stats from a game, converts game stats to part of team's rolling average stats.
    
    :param teams: A list of all the teams in the league
    :param home_name: The name of the home team
    :param home_score: The score of the home team
    :param home_touchdowns: The total number of offensive touchdowns of the home team
    :param home_net_pass_yards: The net passing yards of the home team
    :param home_pass_completions: The number of pass completions of the home team
    :param home_pass_attempts: The number of pass attempts of the home team
    :param home_pass_tds: The number of passing touchdowns of the home team
    :param home_interceptions_thrown: The number of interceptions thrown by the home team
    :param home_total_yards: The total number of offensive yards of the home team
    :param away_name: The name of the away team
    :param away_score: The score of the away team
    :param away_touchdowns: The total number of offensive touchdowns of the away team
    :param away_net_pass_yards: The net passing yards of the away team
    :param away_pass_completions: The number of pass completions of the away team
    :param away_pass_attempts: The number of pass attempts of the away team
    :param away_pass_tds: The number of passing touchdowns of the away team
    :param away_interceptions_thrown: The number of interceptions thrown by the away team
    :param away_total_yards: The total number of offensive yards of the away team
    :return: An updated list of all the teams in the league
    """

    # Get the home team info
    home = get_team(teams, home_name)
    home_wins = home[1]
    home_losses = home[2]
    home_ties = home[3]
    home_elo = home[4]
    home_average_points_for = home[5]
    home_average_points_against = home[6]
    home_average_tds = home[7]
    home_total_net_pass_yards = home[8]
    home_total_pass_completions = home[9]
    home_total_pass_attempts = home[10]
    home_total_pass_tds = home[11]
    home_total_interceptions_thrown = home[12]
    home_average_total_yards = home[13]

    # Get the away team info
    away = get_team(teams, away_name)
    away_wins = away[1]
    away_losses = away[2]
    away_ties = away[3]
    away_elo = away[4]
    away_average_points_for = away[5]
    away_average_points_against = away[6]
    away_average_tds = away[7]
    away_total_net_pass_yards = away[8]
    away_total_pass_completions = away[9]
    away_total_pass_attempts = away[10]
    away_total_pass_tds = away[11]
    away_total_interceptions_thrown = away[12]
    away_average_total_yards = away[13]

    # Get the game result
    home_victory = home_score > away_score
    draw = home_score == away_score

    # Get the number of games each team has played
    home_games_played = home_wins + home_losses + home_ties
    away_games_played = away_wins + away_losses + away_ties

    # Get the total values for averaged stats
    home_total_points_for = home_average_points_for * home_games_played
    home_total_points_against = home_average_points_against * home_games_played
    home_total_tds = home_average_tds * home_games_played
    home_total_total_yards = home_average_total_yards * home_games_played

    away_total_points_for = away_average_points_for * away_games_played
    away_total_points_against = away_average_points_against * away_games_played
    away_total_tds = away_average_tds * away_games_played
    away_total_total_yards = away_average_total_yards * away_games_played

    # Update each teams record and the number of games played
    if home_victory:
        home_wins = home_wins + 1
        away_losses = away_losses + 1
    elif draw:
        home_ties = home_ties + 1
        away_ties = away_ties + 1
    else:
        home_losses = home_losses + 1
        away_wins = away_wins + 1

    home_games_played = home_games_played + 1
    away_games_played = away_games_played + 1

    # Update each teams elo
    home_elo, away_elo = NFL.get_new_elos(home_elo,
                                          away_elo,
                                          home_victory,
                                          draw,
                                          42)

    # Update the totals for all other stats for the home team
    home_total_points_for = home_total_points_for + home_score
    home_total_points_against = home_total_points_against + away_score
    home_total_tds = home_total_tds + home_touchdowns
    home_total_net_pass_yards = home_total_net_pass_yards + home_net_pass_yards
    home_total_pass_completions = home_total_pass_completions + home_pass_completions
    home_total_pass_attempts = home_total_pass_attempts + home_pass_attempts
    home_total_pass_tds = home_total_pass_tds + home_pass_tds
    home_total_interceptions_thrown = home_total_interceptions_thrown + home_interceptions_thrown
    home_total_total_yards = home_total_total_yards + home_total_yards

    # Update the totals for all other stats for the away team
    away_total_points_for = away_total_points_for + away_score
    away_total_points_against = away_total_points_against + home_score
    away_total_tds = away_total_tds + away_touchdowns
    away_total_net_pass_yards = away_total_net_pass_yards + away_net_pass_yards
    away_total_pass_completions = away_total_pass_completions + away_pass_completions
    away_total_pass_attempts = away_total_pass_attempts + away_pass_attempts
    away_total_pass_tds = away_total_pass_tds + away_pass_tds
    away_total_interceptions_thrown = away_total_interceptions_thrown + away_interceptions_thrown
    away_total_total_yards = away_total_total_yards + away_total_yards

    # Average the home teams averaged stats
    home_average_points_for = home_total_points_for / home_games_played
    home_average_points_against = home_total_points_against / home_games_played
    home_average_tds = home_total_tds / home_games_played
    home_average_total_yards = home_total_total_yards / home_games_played

    # Average the away teams averaged stats
    away_average_points_for = away_total_points_for / away_games_played
    away_average_points_against = away_total_points_against / away_games_played
    away_average_tds = away_total_tds / away_games_played
    away_average_total_yards = away_total_total_yards / away_games_played

    # Create new teams with the updated stats
    new_home = (home[0], home_wins, home_losses, home_ties, home_elo, home_average_points_for,
                home_average_points_against, home_average_tds, home_total_net_pass_yards, home_total_pass_completions,
                home_total_pass_attempts, home_total_pass_tds, home_total_interceptions_thrown,
                home_average_total_yards)

    new_away = (away[0], away_wins, away_losses, away_ties, away_elo, away_average_points_for,
                away_average_points_against, away_average_tds, away_total_net_pass_yards, away_total_pass_completions,
                away_total_pass_attempts, away_total_pass_tds, away_total_interceptions_thrown,
                away_average_total_yards)

    # Update each team in the list and return the list
    teams = [new_home if team == home else team for team in teams]
    teams = [new_away if team == away else team for team in teams]

    return teams


def get_week_probabilities(teams, games, verbose=False):
    """
    Gets all the game outcome probabilities for each game in a week.

    :param teams: A list of all the teams in the league
    :param games: A list of all the games in the week
    :param verbose: If verbose output is desired
    :return: The list of probabilities for each game
    """

    # For each game in the list of games
    probabilities = list()
    for game in games:
        # Predict the probability and add it to a list
        probabilities.append(predict_game_outcome(teams, game[0][0], game[0][1], game[1], neutral_location=game[2],
                                                  verbose=verbose))

    # Sort the list of probabilities of each game from most likely to least likely
    probabilities.sort(key=lambda outcome: outcome[0], reverse=True)

    # Print a message for each game
    for game in probabilities:
        print(game[1])
    print()

    # Return the list of probabilities for each game
    return probabilities


def get_team(teams, team_name):
    """
    Gets a specific team in the league based on the team name.

    :param teams: The list of all the teams in the league
    :param team_name: The name of the team to get
    :return: The team with the given name
    """

    for team in teams:
        if team[0] == team_name:
            return team
