import pandas as pd

from sklearn.externals import joblib

base_dir = '..\\Projects\\nfl\\NFL_Prediction\\Game Data\\'


def predict_game(home_info, away_info, home_spread=0):
    # Load the classifier and scaler
    voting_classifier = joblib.load(base_dir + 'Other\\2018VotingClassifier.pkl')
    scaler = joblib.load(base_dir + 'Other\\Scaler.pkl')

    # Get the home team's info
    home_wins = home_info[1]
    home_losses = home_info[2]
    home_ties = home_info[3]
    home_elo = home_info[4]
    home_average_points_for = home_info[5]
    home_average_points_against = home_info[6]
    home_average_tds = home_info[7]
    home_net_pass_yards = home_info[8]
    home_pass_completions = home_info[9]
    home_pass_attempts = home_info[10]
    home_pass_tds = home_info[11]
    home_interceptions_thrown = home_info[12]
    home_average_total_yards = home_info[13]
    home_average_first_downs = home_info[14]
    home_third_down_conversions = home_info[15]
    home_third_downs = home_info[16]

    # Get the away team's info
    away_wins = away_info[1]
    away_losses = away_info[2]
    away_ties = away_info[3]
    away_elo = away_info[4]
    away_average_points_for = away_info[5]
    away_average_points_against = away_info[6]
    away_average_tds = away_info[7]
    away_net_pass_yards = away_info[8]
    away_pass_completions = away_info[9]
    away_pass_attempts = away_info[10]
    away_pass_tds = away_info[11]
    away_interceptions_thrown = away_info[12]
    away_average_total_yards = away_info[13]
    away_average_first_downs = away_info[14]
    away_third_down_conversions = away_info[15]
    away_third_downs = away_info[16]

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

    home_average_completion_pct = home_pass_completions / home_pass_attempts if home_pass_attempts > 0 else 0
    home_a = (home_average_completion_pct - .3) * 5

    home_average_yards_per_pass_attempt = home_net_pass_yards / home_pass_attempts if home_pass_attempts > 0 else 0
    home_b = (home_average_yards_per_pass_attempt - 3) * .25

    home_average_passing_touchdowns_per_attempt = home_pass_tds / home_pass_attempts if home_pass_attempts > 0 else 0
    home_c = home_average_passing_touchdowns_per_attempt * 20

    home_average_ints_per_attempt = home_interceptions_thrown / home_pass_attempts if home_pass_attempts > 0 else 0
    home_d = 2.375 - (home_average_ints_per_attempt * 25)

    home_average_passer_rating = ((home_a + home_b + home_c + home_d) / 6) * 100

    away_average_completion_pct = away_pass_completions / away_pass_attempts if away_pass_attempts > 0 else 0
    away_a = (away_average_completion_pct - .3) * 5

    away_average_yards_per_pass_attempt = away_net_pass_yards / away_pass_attempts if away_pass_attempts > 0 else 0
    away_b = (away_average_yards_per_pass_attempt - 3) * .25

    away_average_passing_touchdowns_per_attempt = away_pass_tds / away_pass_attempts if away_pass_attempts > 0 else 0
    away_c = away_average_passing_touchdowns_per_attempt * 20

    away_average_ints_per_attempt = away_interceptions_thrown / away_pass_attempts if away_pass_attempts > 0 else 0
    away_d = 2.375 - (away_average_ints_per_attempt * 25)

    away_average_passer_rating = ((away_a + away_b + away_c + away_d) / 6) * 100

    average_passer_rating_diff = home_average_passer_rating - away_average_passer_rating

    average_total_yards_diff = home_average_total_yards - away_average_total_yards
    average_first_downs_diff = home_average_first_downs - away_average_first_downs
    average_yards_per_pass_attempt_diff = home_average_yards_per_pass_attempt - away_average_yards_per_pass_attempt

    home_average_third_down_pct = home_third_down_conversions / home_third_downs if home_third_downs > 0 else 0
    away_average_third_down_pct = away_third_down_conversions / away_third_downs if away_third_downs > 0 else 0
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
    home = get_team(teams, home_name)
    away = get_team(teams, away_name)

    home_vote_prob, home_lr_prob, home_svc_prob, home_rf_prob = predict_home_victory(home, away, home_spread)
    away_vote_prob, away_lr_prob, away_svc_prob, away_rf_prob = predict_away_victory(away, away, home_spread)

    home_vote_prob_formatted = round(home_vote_prob * 100, 2)
    away_vote_prob_formatted = round(away_vote_prob * 100, 2)
    if home_vote_prob >= away_vote_prob:
        message = 'The ' + home_name + ' have a ' + str(home_vote_prob_formatted) + '% chance to beat the ' + away_name
    else:
        message = 'The ' + away_name + ' have a ' + str(away_vote_prob_formatted) + '% chance to beat the ' + home_name

    home_lr_prob_formatted = round(home_lr_prob * 100, 2)
    home_svc_prob_formatted = round(home_svc_prob * 100, 2)
    home_rf_prob_formatted = round(home_rf_prob * 100, 2)
    away_lr_prob_formatted = round(away_lr_prob * 100, 2)
    away_svc_prob_formatted = round(away_svc_prob * 100, 2)
    away_rf_prob_formatted = round(away_rf_prob * 100, 2)
    if verbose:
        print(message)
        print('Logistic Regression Home Victory Probability: ' + str(home_lr_prob_formatted) + '%')
        print('Logistic Regression Away Victory Probability: ' + str(away_lr_prob_formatted) + '%')
        print('SVC Home Victory Probability: ' + str(home_svc_prob_formatted) + '%')
        print('SVC Away Victory Probability: ' + str(away_svc_prob_formatted) + '%')
        print('Random Forest Home Victory Probability: ' + str(home_rf_prob_formatted) + '%')
        print('Random Forest Away Victory Probability: ' + str(away_rf_prob_formatted) + '%')

    return home_vote_prob if home_vote_prob >= away_vote_prob else away_vote_prob, message


def update_teams():
    # TODO
    pass


def get_team(teams, team_name):
    for team in teams:
        if team[0] == team_name:
            return team


def set_team(teams, new_team):
    for index, team in enumerate(teams):
        if team[0] == new_team[0]:
            teams[index] = new_team
