import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

game_data_dir = '..\\Projects\\nfl\\NFL_Prediction\\Game Data\\'
other_dir = '..\\Projects\\nfl\\NFL_Prediction\\Other\\'


def get_best_features(df=None):
    won_series = plot_corr(df)
    best_features = pca_feature_selection(df)

    columns_to_keep = list()
    columns_to_keep.extend(list(filter(lambda f: 'diff' in f, df.columns.values)))
    columns_to_keep.extend(list(filter(lambda f: 'home_spread' in f, df.columns.values)))
    columns_to_keep.remove('game_point_diff')
    # best_features = recursive_feature_elimination(df, columns_to_keep)

    best_features = ['home_spread', 'elo_diff', 'average_scoring_margin_diff', 'win_pct_diff',
                     'average_touchdowns_diff', 'average_passer_rating_diff', 'average_total_yards_diff']

    won_series = won_series.filter(best_features).sort_values(kind='quicksort', ascending=False)
    best_features = list(won_series.index)
    return best_features


def plot_corr(df=None):
    """
    Gets the correlation between all relevant features and the home_victory label.

    :param df: The data frame containing the features to plot the correlation for
    :return: The series containing the sorted correlations between each feature and the home_victory label
    """

    if df is None:
        # Get the data frame for all seasons
        df = pd.read_csv(game_data_dir + '19952018.csv')

    # Print a description of all the games
    games_description = df.describe()
    games_description.to_csv(other_dir + '19952018Description.csv')
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


def plot_top_features_corr(df=None):
    """
    Get the correlation between the top features determined by recursive feature elimination and the home_victory label.

    :param df: The data frame containing the features to plot the correlation for
    :return: The series containing the sorted correlations between each feature and the home_victory label
    """

    if df is None:
        # Get the data frame for all seasons
        df = pd.read_csv(game_data_dir + '19952018.csv')

    # Print a description of all the games
    games_description = df.describe()
    games_description.to_csv(other_dir + '19952018Description.csv')
    print(games_description.to_string())
    print()

    # Drop all columns that arent the label or the top features
    df = df.filter(items=['home_spread', 'elo_diff', 'average_scoring_margin_diff', 'win_pct_diff',
                          'average_touchdowns_diff', 'average_passer_rating_diff', 'average_total_yards_diff',
                          'average_points_for_diff', 'average_points_against_diff', 'average_third_down_pct_diff',
                          'average_passing_touchdowns_per_attempt_diff', 'average_sacked_yards_diff',
                          'average_sacked_pct_diff', 'average_sack_yards_forced_diff', 'average_completion_pct_diff',
                          'average_penalty_yards_diff', 'average_yards_per_rush_attempt_diff'])

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


def pca_feature_selection(df=None):
    """
    Gets the set of at most 10 features that explains the variance for the y label.

    :param df: The data frame containing all games to get the best features from
    :return: The list of features the contribute most towards explaining the variance
    """

    if df is None:
        # Get the data frame for all seasons
        df = pd.read_csv(game_data_dir + '19952018.csv')

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
    n = .998
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

    # Print the top features
    contributing_features = print_top_features(skb, feature_col_names)

    # Correlation matrix revealed high correlation between points for and touchdowns
    if 'average_points_for_diff' in contributing_features:
        contributing_features.remove('average_points_for_diff')
    if 'average_points_against_diff' in contributing_features:
        contributing_features.remove('average_points_against_diff')

    # Plot the correlation between the top 8 features
    columns_to_drop = list(set(feature_col_names) - set(contributing_features))
    relevant = df.drop(columns=columns_to_drop)
    corrmat = relevant.corr()
    f, ax = plt.subplots(figsize=(9, 9))
    sns.set(font_scale=0.9)
    sns.heatmap(corrmat, vmax=.8, square=True, annot=True, fmt='.2f', cmap='winter')
    plt.show()

    # Get the correlations of each contributing feature
    correlations = corrmat.unstack()['home_victory']
    negatively_correlated = correlations[correlations < 0]
    positively_correlated = correlations[correlations > 0]

    # Get a list of which features are positively or negatively correlated
    negatively_correlated = list(negatively_correlated.index)
    positively_correlated = list(positively_correlated.index)
    positively_correlated.remove('home_victory')

    # Get description of relevant features
    contributing_feature_description = df.filter(contributing_features).describe()
    print(contributing_feature_description.to_string())
    print()

    return contributing_features


def recursive_feature_elimination(df, feature_list):
    """
    Recursively eliminates features from the feature set based on brier loss and f1 score.

    :param df: The data frame to train
    :param feature_list: The list of all the features to examine
    :return: Void
    """

    num_features = 1

    X = df[feature_list].values
    y = df['home_victory'].values

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    clf = RandomForestClassifier(n_estimators=100)
    skf = StratifiedKFold(n_splits=5)

    print('RFECV Brier')
    rfecv_br = RFECV(estimator=clf, min_features_to_select=num_features, cv=skf, scoring='brier_score_loss')
    rfecv_br.fit(X, y.ravel())
    brier_features = print_top_features(rfecv_br, feature_list)
    br_features_to_scores = zip(feature_list, rfecv_br.ranking_)
    br_features_to_scores = sorted(br_features_to_scores, key=lambda x: x[1])
    for item in br_features_to_scores:
        print(item[0], item[1])
    print()

    print('RFECV F1')
    rfecv_f1 = RFECV(estimator=clf, min_features_to_select=num_features, cv=skf, scoring='f1')
    rfecv_f1.fit(X, y.ravel())
    f1_features = print_top_features(rfecv_f1, feature_list)
    f1_features_to_scores = zip(feature_list, rfecv_f1.ranking_)
    f1_features_to_scores = sorted(f1_features_to_scores, key=lambda x: x[1])
    for item in f1_features_to_scores:
        print(item[0], item[1])
    print()

    return brier_features


def print_top_features(selector, feature_list):
    """
    Iterates through the support list and prints all the features that are included in the support.

    :param selector: The selector used for determining the top features
    :param feature_list: The list of all the features that were examined by the selector
    :return: The list of features that are deemed necessary by the selector
    """

    # Get the mask array
    feature_support = list(selector.get_support())

    num_features = sum(feature_support)

    # For each feature in the list
    contributing_features = list()
    for i in range(0, len(feature_support)):
        # If the feature is one of the top features, add it to the list
        if feature_support[i]:
            contributing_features.append(feature_list[i])

    # Print the list of the top features
    print('The top ' + str(num_features) + ' features are: ')
    for feature in contributing_features:
        print(feature)
    print()

    return contributing_features


def eval_feature_set(df, feature_list):
    """
    Gets the cross validation score of a set of features using a random forest algorithm.  Scored bby brier loss and f1.

    :param df: The data frame to train on
    :param feature_list: The list of all the features used in training
    :return: The average brier loss and f1 score
    """

    X = df[feature_list].values
    y = df['home_victory'].values

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    clf = RandomForestClassifier(n_estimators=100)
    skf = StratifiedKFold(n_splits=5)

    brier_score = cross_val_score(clf, X, y, scoring='brier_score_loss', cv=skf)
    f1_score = cross_val_score(clf, X, y, scoring='f1', cv=skf)

    return statistics.mean(brier_score), statistics.mean(f1_score)
