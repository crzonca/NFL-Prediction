import pandas as pd
import Projects.nfl.NFL_Prediction.NFL as Nfl

from sklearn.externals import joblib

base_dir = '..\\Projects\\nfl\\NFL_Prediction\\'


def predict_game(home_info, away_info, home_spread=0):
    # Load the classifier and scaler
    voting_classifier = joblib.load(base_dir + 'Other\\2018VotingClassifier.pkl')
    scaler = joblib.load(base_dir + 'Other\\2018Scaler.pkl')

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
    home_average_first_downs = home_info[14]
    home_total_third_down_conversions = home_info[15]
    home_total_third_downs = home_info[16]

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
    away_average_first_downs = away_info[14]
    away_total_third_down_conversions = away_info[15]
    away_total_third_downs = away_info[16]

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

    average_points_for_diff = home_average_points_for - away_average_points_for
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
    average_first_downs_diff = home_average_first_downs - away_average_first_downs
    average_yards_per_pass_attempt_diff = home_average_yards_per_pass_attempt - away_average_yards_per_pass_attempt

    home_average_third_down_pct = home_total_third_down_conversions / home_total_third_downs \
        if home_total_third_downs > 0 else 0
    away_average_third_down_pct = away_total_third_down_conversions / away_total_third_downs \
        if away_total_third_downs > 0 else 0
    average_third_down_pct_diff = home_average_third_down_pct - away_average_third_down_pct

    # Organize the features
    game_features = (home_spread,
                     elo_diff,
                     average_scoring_margin_diff,
                     win_pct_diff,
                     average_points_for_diff,
                     average_touchdowns_diff,
                     average_passer_rating_diff,
                     average_total_yards_diff,
                     average_first_downs_diff,
                     average_yards_per_pass_attempt_diff,
                     average_third_down_pct_diff)

    # Convert the features to a data frame and scale it
    game = pd.DataFrame([game_features])
    game = scaler.transform(game)

    # Get the voting classifier probability
    vote_prob = voting_classifier.predict_proba([game])[0]

    # Get the individual estimator probabilities
    estimator_probs = voting_classifier.transform([game])
    lr_prob = estimator_probs[0][0]
    svc_prob = estimator_probs[1][0]
    rf_prob = estimator_probs[2][0]

    return vote_prob, lr_prob, svc_prob, rf_prob


def predict_home_victory(home_info, away_info, home_spread=0):
    vote_prob, lr_prob, svc_prob, rf_prob = predict_game(home_info, away_info, home_spread)
    return vote_prob[1], lr_prob[1], svc_prob[1], rf_prob[1]


def predict_away_victory(home_info, away_info, home_spread=0):
    vote_prob, lr_prob, svc_prob, rf_prob = predict_game(home_info, away_info, home_spread)
    return vote_prob[0], lr_prob[0], svc_prob[0], rf_prob[0]


def predict_game_outcome(teams, home_name, away_name, home_spread, verbose=False):
    # Get each team
    home = get_team(teams, home_name)
    away = get_team(teams, away_name)

    # Get each teams probability of victory according to each estimator
    home_vote_prob, home_lr_prob, home_svc_prob, home_rf_prob = predict_home_victory(home, away, home_spread)
    away_vote_prob, away_lr_prob, away_svc_prob, away_rf_prob = predict_away_victory(away, away, home_spread)

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
        print(message)
        print('Logistic Regression Home Victory Probability: ' + str(home_lr_prob_formatted) + '%')
        print('Logistic Regression Away Victory Probability: ' + str(away_lr_prob_formatted) + '%')
        print('SVC Home Victory Probability: ' + str(home_svc_prob_formatted) + '%')
        print('SVC Away Victory Probability: ' + str(away_svc_prob_formatted) + '%')
        print('Random Forest Home Victory Probability: ' + str(home_rf_prob_formatted) + '%')
        print('Random Forest Away Victory Probability: ' + str(away_rf_prob_formatted) + '%')
        print()

    # Return the favored teams chance of winning and a message
    return home_vote_prob if home_vote_prob >= away_vote_prob else away_vote_prob, message


def update_teams(teams, away_name, away_score, home_name, home_score,
                 home_touchdowns, home_net_pass_yards, home_pass_completions, home_pass_attempts, home_pass_tds,
                 home_interceptions_thrown, home_total_yards, home_first_downs, home_third_down_conversions,
                 home_third_downs,
                 away_touchdowns, away_net_pass_yards, away_pass_completions, away_pass_attempts, away_pass_tds,
                 away_interceptions_thrown, away_total_yards, away_first_downs, away_third_down_conversions,
                 away_third_downs):
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
    home_average_first_downs = home[14]
    home_total_third_down_conversions = home[15]
    home_total_third_downs = home[16]

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
    away_average_first_downs = away[14]
    away_total_third_down_conversions = away[15]
    away_total_third_downs = away[16]

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
    home_total_first_downs = home_average_first_downs * home_games_played

    away_total_points_for = away_average_points_for * away_games_played
    away_total_points_against = away_average_points_against * away_games_played
    away_total_tds = away_average_tds * away_games_played
    away_total_total_yards = away_average_total_yards * away_games_played
    away_total_first_downs = away_average_first_downs * away_games_played

    # Update each teams record and the number of games played
    if home_victory:
        home_wins = home_wins + 1
        away_losses = away_losses + 1
    else:
        home_losses = home_losses + 1
        away_wins = away_wins + 1

    if draw:
        home_ties = home_ties + 1
        away_ties = away_ties + 1

    home_games_played = home_games_played + 1
    away_games_played = away_games_played + 1

    # Update each teams elo
    home_elo, away_elo = Nfl.get_new_elos(home_elo,
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
    home_total_first_downs = home_total_first_downs + home_first_downs
    home_total_third_down_conversions = home_total_third_down_conversions + home_third_down_conversions
    home_total_third_downs = home_total_third_downs + home_third_downs

    # Update the totals for all other stats for the away team
    away_total_points_for = away_total_points_for + away_score
    away_total_points_against = away_total_points_against + away_score
    away_total_tds = away_total_tds + away_touchdowns
    away_total_net_pass_yards = away_total_net_pass_yards + away_net_pass_yards
    away_total_pass_completions = away_total_pass_completions + away_pass_completions
    away_total_pass_attempts = away_total_pass_attempts + away_pass_attempts
    away_total_pass_tds = away_total_pass_tds + away_pass_tds
    away_total_interceptions_thrown = away_total_interceptions_thrown + away_interceptions_thrown
    away_total_total_yards = away_total_total_yards + away_total_yards
    away_total_first_downs = away_total_first_downs + away_first_downs
    away_total_third_down_conversions = away_total_third_down_conversions + away_third_down_conversions
    away_total_third_downs = away_total_third_downs + away_third_downs

    # Average the home teams averaged stats
    home_average_points_for = home_total_points_for / home_games_played
    home_average_points_against = home_total_points_against / home_games_played
    home_average_tds = home_total_tds / home_games_played
    home_average_total_yards = home_total_total_yards / home_games_played
    home_average_first_downs = home_total_first_downs / home_games_played

    # Average the away teams averaged stats
    away_average_points_for = away_total_points_for / away_games_played
    away_average_points_against = away_total_points_against / away_games_played
    away_average_tds = away_total_tds / away_games_played
    away_average_total_yards = away_total_total_yards / away_games_played
    away_average_first_downs = away_total_first_downs / away_games_played

    # Create new teams with the updated stats
    new_home = (home[0], home_wins, home_losses, home_ties, home_elo, home_average_points_for,
                home_average_points_against, home_average_tds, home_total_net_pass_yards, home_total_pass_completions,
                home_total_pass_attempts, home_total_pass_tds, home_total_interceptions_thrown,
                home_average_total_yards, home_average_first_downs, home_total_third_down_conversions,
                home_total_third_downs)

    new_away = (away[0], away_wins, away_losses, away_ties, away_elo, away_average_points_for,
                away_average_points_against, away_average_tds, away_total_net_pass_yards, away_total_pass_completions,
                away_total_pass_attempts, away_total_pass_tds, away_total_interceptions_thrown,
                away_average_total_yards, away_average_first_downs, away_total_third_down_conversions,
                away_total_third_downs)

    # Update each team in the list and return the list
    teams = [new_home if team == home else team for team in teams]
    teams = [new_away if team == away else team for team in teams]
    return teams


def get_team(teams, team_name):
    for team in teams:
        if team[0] == team_name:
            return team


def set_team(teams, new_team):
    for index, team in enumerate(teams):
        if team[0] == new_team[0]:
            teams[index] = new_team
