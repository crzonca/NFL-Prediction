import Projects.nfl.NFL_Prediction.NFLDataGroomer as Groomer
import Projects.nfl.NFL_Prediction.NFLFeatureSelection as FeatureSelection
import Projects.nfl.NFL_Prediction.NFLModelEvaluation as ModelEval
import Projects.nfl.NFL_Prediction.NFLParamOptimization as Params


def nfl():
    # Groom the scraped data
    # all_games = Groomer.groom_data()

    # Get the groomed games, without any outliers
    all_games = Groomer.get_all_games_no_outliers()

    # Select the best features
    features = FeatureSelection.get_best_features()

    # Tune each algorithm's hyper parameters
    Params.evaluate_model_parameters(features, all_games)

    # Compare results and choose best parameters
    ModelEval.analyze_results()

    # Create the voting classifier
    classifier = Params.get_voting_classifier(features, all_games)

    # Evaluate the results with 2018 as a validation set
    ModelEval.evaluate_2018_season()
