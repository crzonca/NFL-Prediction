import statistics
import sys

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

game_data_dir = '..\\Projects\\nfl\\NFL_Prediction\\Game Data\\'
other_dir = '..\\Projects\\nfl\\NFL_Prediction\\Other\\'


def dummy_classify_2018_season(feature_list):
    """
    Evaluates a voting classifier based on the 2018 season.

    :return: Void
    """

    df = pd.read_csv(game_data_dir + '20022018.csv')

    X = df[feature_list].values[:-267]
    y = df['home_victory'].values[:-267]

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Set the directory to write files to
    filename = other_dir + '7 Features\\Scores\\2017\\2018Confusion_Dummy.txt'

    classifier = DummyClassifier(strategy='prior', random_state=42)
    classifier.fit(X, y.ravel())

    last_season = pd.read_csv(game_data_dir + '20022018.csv').values[-267:]
    last_season = pd.DataFrame(last_season)

    results = pd.DataFrame(columns=['prediction', 'outcome'])
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
        prob = classifier.predict_proba(game)[0][1]

        game_df = pd.DataFrame([[prob, home_victory]], columns=['prediction', 'outcome'])

        results = results.append(game_df)

    outcome = results['outcome']
    prediction = results['prediction']

    dummy_brier = brier_score_loss(outcome, prediction)
    print('Dummy Classifier Brier Score Loss:', round(dummy_brier, 4))
    print('Dummy Classifier Brier Score Loss:', round(dummy_brier, 4), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    print('Dummy Classifier:', round((.25 - dummy_brier) * 26700, 2))
    print('Dummy Classifier:', round((.25 - dummy_brier) * 26700, 2), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    results.to_csv(other_dir + '7 Features\\Scores\\2017\\2018Predictions_Dummy.csv', index=False)

    rounded_prediction = prediction.apply(lambda row: round(row))

    print('Dummy Classifier')
    print('-' * 120)
    print('Dummy Classifier', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_prediction, filename)

    visualize_2018_season()


def evaluate_2018_season():
    """
    Evaluates a voting classifier based on the 2018 season.

    :return: Void
    """

    # Set the directory to write files to
    filename = other_dir + '7 Features\\Scores\\2017\\2018Confusion.txt'

    voting_classifier = joblib.load(other_dir + '7 Features\\Scores\\2017\\2017VotingClassifier.pkl')
    scaler = joblib.load(other_dir + '7 Features\\Scores\\2017\\2017Scaler.pkl')

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

    results.to_csv(other_dir + '7 Features\\Scores\\2017\\2018Predictions.csv', index=False)

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

    print('Logistic Regression')
    print('-' * 120)
    print('Logistic Regression', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_lr, filename)

    print('Voting Classifier')
    print('-' * 120)
    print('Voting Classifier', file=open(filename, 'a'))
    print('-' * 120, file=open(filename, 'a'))
    get_metrics(outcome, rounded_vote, filename)

    visualize_2018_season()


def get_metrics(y_true, y_pred, filename):
    """
    Creates a confusion matrix and prints detains about classification predictions.

    :param y_true: The actual game outcomes
    :param y_pred: The predicted game outcomes
    :param filename: The name of the file to write the reuslts to
    :return: Void
    """

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
    if 1 in predicted_counts.index:
        predicted_positive = predicted_counts.loc[1]
    else:
        predicted_positive = 0
    if 0 in predicted_counts.index:
        predicted_negative = predicted_counts.loc[0]
    else:
        predicted_negative = 0
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

    false_discovery = false_positive / predicted_positive if predicted_positive > 0 else 0
    print('False discovery rate:', round(false_discovery * 100, 2),
          str(false_positive) + '/' + str(predicted_positive))

    print('False discovery rate:', round(false_discovery * 100, 2),
          str(false_positive) + '/' + str(predicted_positive), file=open(filename, 'a'))

    negative_prediction = true_negative / predicted_negative if predicted_negative > 0 else 0
    print('Negative prediction rate:', round(negative_prediction * 100, 2),
          str(true_negative) + '/' + str(predicted_negative))

    print('Negative prediction rate:', round(negative_prediction * 100, 2),
          str(true_negative) + '/' + str(predicted_negative), file=open(filename, 'a'))

    false_omission = false_negative / predicted_negative if predicted_negative > 0 else 0
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

    positive_likelihood = recall / fall_out if fall_out > 0 else recall / sys.maxsize
    print('Positive likelihood ratio:', round(positive_likelihood, 4),
          str(round(recall * 100, 2)) + '/' + str(round(fall_out * 100, 2)))

    print('Positive likelihood ratio:', round(positive_likelihood, 4),
          str(round(recall * 100, 2)) + '/' + str(round(fall_out * 100, 2)), file=open(filename, 'a'))

    negative_likelihood = miss_rate / specificity if specificity > 0 else miss_rate / sys.maxsize
    print('Negative likelihood ratio:', round(negative_likelihood, 4),
          str(round(miss_rate * 100, 2)) + '/' + str(round(specificity * 100, 2)))
    print()

    print('Negative likelihood ratio:', round(negative_likelihood, 4),
          str(round(miss_rate * 100, 2)) + '/' + str(round(specificity * 100, 2)), file=open(filename, 'a'))
    print('', file=open(filename, 'a'))

    diagnostic_odds = positive_likelihood / negative_likelihood if negative_likelihood > 0 else sys.maxsize
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
    """
    Creates a visual representation of the prediction results for evaluating the 2018 season.

    :return: Void
    """
    predictions = pd.read_csv(other_dir + '7 Features\\Scores\\2017\\2018Predictions.csv')

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


def analyze_results():
    """
    Analyzes grid search results to determine best set of hyper parameters.

    :return: The sorted list of hyper parameters, sorted based on accuracy and loss
    """
    with open(other_dir + '7 Features\\Scores\\svc_brier_score_loss.txt') as brier:
        with open(other_dir + '7 Features\\Scores\\svc_f1.txt') as f1:
            brier_lines = brier.readlines()[4:]
            f1_lines = f1.readlines()[4:]

            tenth = int(len(brier_lines) * .99)
            if tenth > 10:
                brier_lines = brier_lines[:tenth]
                f1_lines = f1_lines[:tenth]

            briers = dict()
            for brier_line in brier_lines:
                brier_score = float(brier_line.split()[0])
                brier_params = brier_line.split(' for ')[-1].split(' probability ')[0]
                briers[brier_params] = brier_score

            f1s = dict()
            for f1_line in f1_lines:
                f1_score = float(f1_line.split()[0])
                f1_params = f1_line.split(' for ')[-1].split(' probability ')[0]
                f1s[f1_params] = f1_score

            brier_scores = briers.values()
            min_brier = min(brier_scores)
            max_brier = max(brier_scores)
            mean_brier = statistics.mean(brier_scores)
            brier_dev = statistics.stdev(brier_scores)

            for brier_key, brier_val in briers.items():
                briers[brier_key] = (brier_val - min_brier) / (max_brier - min_brier)

            f1s_scores = f1s.values()
            min_f1 = min(f1s_scores)
            max_f1 = max(f1s_scores)
            mean_f1 = statistics.mean(f1s_scores)
            f1_dev = statistics.stdev(f1s_scores)

            for f1_key, f1_val in f1s.items():
                f1s[f1_key] = (f1_val - min_f1) / (max_f1 - min_f1)

            params = dict()
            for brier_key, brier_val in briers.items():
                if f1s.get(brier_key) is not None:
                    params[brier_key] = brier_val + f1s.get(brier_key)

            sorted_params = sorted(params.items(), key=lambda kv: kv[1], reverse=True)
            return sorted_params
