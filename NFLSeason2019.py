import statistics

from prettytable import PrettyTable

import Projects.nfl.NFL_Prediction.NFLPredictor as Predictor


def season():
    eliminated_teams = list()
    teams = handle_week(None, 'Preseason', set_up_teams, eliminated_teams)
    # teams = handle_week(teams, 'Week 1', week_1)
    # teams = handle_week(teams, 'Week 2', week_2)
    # teams = handle_week(teams, 'Week 3', week_3)
    # teams = handle_week(teams, 'Week 4', week_4)
    # teams = handle_week(teams, 'Week 5', week_5)
    # teams = handle_week(teams, 'Week 6', week_6)
    # teams = handle_week(teams, 'Week 7', week_7)
    # teams = handle_week(teams, 'Week 8', week_8)
    # teams = handle_week(teams, 'Week 9', week_9)
    # teams = handle_week(teams, 'Week 10', week_10)
    # teams = handle_week(teams, 'Week 11', week_11)
    # teams = handle_week(teams, 'Week 12', week_12)
    #
    # eliminated_teams.extend(['Raiders', '49ers'])
    # teams = handle_week(teams, 'Week 13', week_13, eliminated_teams)
    #
    # eliminated_teams.extend(['Bills', 'Cardinals', 'Jets', 'Jaguars'])
    # teams = handle_week(teams, 'Week 14', week_14, eliminated_teams=eliminated_teams)
    #
    # eliminated_teams.extend(['Broncos', 'Bengals', 'Giants', 'Packers', 'Lions', 'Buccaneers', 'Falcons'])
    # teams = handle_week(teams, 'Week 15', week_15, eliminated_teams=eliminated_teams)
    #
    # eliminated_teams.extend(['Browns', 'Panthers', 'Dolphins', 'Redskins'])
    # teams = handle_week(teams, 'Week 16', week_16, eliminated_teams=eliminated_teams)
    #
    # eliminated_teams.extend(['Steelers', 'Vikings', 'Titans'])
    # teams = handle_week(teams, 'Week 17', week_17, full_standings=True, eliminated_teams=eliminated_teams)
    #
    # eliminated_teams.extend(['Texans', 'Seahawks', 'Ravens', 'Bears'])
    # teams = handle_week(teams, 'Wildcard Weekend', wildcard, full_standings=True, eliminated_teams=eliminated_teams)
    #
    # eliminated_teams.extend(['Colts', 'Cowboys', 'Chargers', 'Eagles'])
    # teams = handle_week(teams, 'Divisional Round', divisional, full_standings=True, eliminated_teams=eliminated_teams)
    #
    # eliminated_teams.extend(['Saints', 'Chiefs'])
    # teams = handle_week(teams, 'Conference Finals', conference, full_standings=True, eliminated_teams=eliminated_teams)
    #
    # teams = handle_week(teams, 'Superbowl', superbowl, full_standings=True, eliminated_teams=eliminated_teams)

    print_league_details(teams, [], full_standings=True)


def set_up_teams():
    def create_base_team(team_name, elo):
        return team_name, 0, 0, 0, elo, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    teams = list()
    teams.append(create_base_team('Cardinals', 1414.907))
    teams.append(create_base_team('Falcons', 1501.919))
    teams.append(create_base_team('Ravens', 1537.051))
    teams.append(create_base_team('Bills', 1467.667))
    teams.append(create_base_team('Panthers', 1486.623))
    teams.append(create_base_team('Bears', 1547.207))
    teams.append(create_base_team('Bengals', 1450.022))
    teams.append(create_base_team('Browns', 1465.156))
    teams.append(create_base_team('Cowboys', 1555.546))
    teams.append(create_base_team('Broncos', 1447.034))
    teams.append(create_base_team('Lions', 1476.841))
    teams.append(create_base_team('Packers', 1463.462))
    teams.append(create_base_team('Texans', 1515.959))
    teams.append(create_base_team('Colts', 1533.559))
    teams.append(create_base_team('Jaguars', 1447.748))
    teams.append(create_base_team('Chiefs', 1562.643))
    teams.append(create_base_team('Chargers', 1566.415))
    teams.append(create_base_team('Rams', 1589.463))
    teams.append(create_base_team('Dolphins', 1458.772))
    teams.append(create_base_team('Vikings', 1532.134))
    teams.append(create_base_team('Patriots', 1608.685))
    teams.append(create_base_team('Saints', 1592.439))
    teams.append(create_base_team('Giants', 1442.666))
    teams.append(create_base_team('Jets', 1408.145))
    teams.append(create_base_team('Raiders', 1433.093))
    teams.append(create_base_team('Eagles', 1561.189))
    teams.append(create_base_team('Steelers', 1544.046))
    teams.append(create_base_team('49ers', 1431.803))
    teams.append(create_base_team('Seahawks', 1535.119))
    teams.append(create_base_team('Buccaneers', 1437.01))
    teams.append(create_base_team('Titans', 1520.108))
    teams.append(create_base_team('Redskins', 1465.572))
    return teams


def handle_week(teams, week_name, week, eliminated_teams, full_standings=False):
    print(week_name)
    if teams:
        teams = week(teams)
    else:
        teams = week()
    print_league_details(teams, eliminated_teams, full_standings=full_standings)
    return teams


# def week_1(teams):
#     # Games are listed as: Away Team, Home Team, Home - Away
#     # probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Falcons', 'Eagles', 1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bills', 'Ravens', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jaguars', 'Giants', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Buccaneers', 'Saints', 9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Texans', 'Patriots', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, '49ers', 'Vikings', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Titans', 'Dolphins', -1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bengals', 'Colts', 2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Steelers', 'Browns', -4.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chiefs', 'Chargers', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Seahawks', 'Broncos', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cowboys', 'Panthers', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Redskins', 'Cardinals', 1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bears', 'Packers', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jets', 'Lions', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', 'Raiders', -5.5))
#     #
#     # probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     # for game in probabilities:
#     #     print(game[1])
#     # print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Falcons', away_score=12, home_name='Eagles', home_score=18,
#                                    away_sacked_yards=26, home_sacked_yards=13, away_penalty_yards=135,
#                                    home_penalty_yards=101, away_convs=0, away_conv_attempts=1, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Bills', 3, 'Ravens', 47, 28, 8, 100, 78, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Jaguars', 20, 'Giants', 15, 8, 14, 119, 43, 0, 0, 0, 2)
#     teams = Predictor.update_teams(teams, 'Buccaneers', 48, 'Saints', 40, 0, 7, 70, 77, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Texans', 20, 'Patriots', 27, 18, 10, 44, 36, 0, 1, 1, 1)
#     teams = Predictor.update_teams(teams, '49ers', 16, 'Vikings', 24, 24, 17, 21, 52, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, 'Titans', 20, 'Dolphins', 27, 0, 8, 48, 51, 0, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Bengals', 34, 'Colts', 23, 14, 14, 94, 91, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Steelers', 21, 'Browns', 21, 22, 47, 116, 87, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, 'Chiefs', 38, 'Chargers', 28, 0, 6, 50, 45, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Seahawks', 24, 'Broncos', 27, 56, 5, 45, 60, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Cowboys', 8, 'Panthers', 16, 32, 15, 85, 80, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Redskins', 24, 'Cardinals', 6, 8, 8, 63, 67, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Bears', 23, 'Packers', 24, 16, 40, 35, 72, 0, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Jets', 48, 'Lions', 17, 18, 0, 49, 15, 2, 3, 0, 2)
#     teams = Predictor.update_teams(teams, 'Rams', 33, 'Raiders', 13, 8, 3, 70, 155, 0, 0, 0, 0)
#
#     return teams
#
#
# def week_2(teams):
#     # probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Ravens', 'Bengals', -1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Panthers', 'Falcons', 5.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Redskins', 6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Texans', 'Titans', -3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Buccaneers', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chiefs', 'Steelers', 4.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Dolphins', 'Jets', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Bills', -7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Vikings', 'Packers', -2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Browns', 'Saints', 9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Lions', '49ers', 6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cardinals', 'Rams', 13))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Jaguars', -1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Raiders', 'Broncos', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Giants', 'Cowboys', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Seahawks', 'Bears', 4.5))
#     #
#     # probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     # for game in probabilities:
#     #     print(game[1])
#     # print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Ravens', away_score=12, home_name='Bengals', home_score=18,
#                                    away_sacked_yards=26, home_sacked_yards=13, away_penalty_yards=135,
#                                    home_penalty_yards=101, away_convs=0, away_conv_attempts=1, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Panthers', 24, 'Falcons', 31, 17, 0, 49, 26, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Colts', 21, 'Redskins', 9, 2, 23, 26, 90, 0, 0, 1, 2)
#     teams = Predictor.update_teams(teams, 'Texans', 17, 'Titans', 20, 21, 8, 88, 70, 0, 0, 2, 2)
#     teams = Predictor.update_teams(teams, 'Eagles', 21, 'Buccaneers', 27, 13, 9, 55, 44, 3, 5, 0, 0)
#     teams = Predictor.update_teams(teams, 'Chiefs', 42, 'Steelers', 37, 4, 10, 76, 90, 0, 0, 2, 2)
#     teams = Predictor.update_teams(teams, 'Dolphins', 20, 'Jets', 12, 46, 14, 10, 50, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Chargers', 31, 'Bills', 20, 16, 36, 45, 31, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Vikings', 29, 'Packers', 29, 13, 28, 70, 54, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Browns', 18, 'Saints', 21, 12, 30, 43, 53, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Cardinals', 0, 'Rams', 34, 7, 12, 47, 49, 1, 1, 1, 3)
#     teams = Predictor.update_teams(teams, 'Lions', 27, '49ers', 30, 18, 50, 105, 86, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Patriots', 20, 'Jaguars', 31, 14, 0, 25, 71, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Raiders', 19, 'Broncos', 20, 7, 5, 30, 35, 0, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Giants', 13, 'Cowboys', 20, 59, 0, 24, 47, 2, 2, 1, 1)
#     teams = Predictor.update_teams(teams, 'Seahawks', 17, 'Bears', 24, 24, 15, 37, 55, 0, 0, 1, 1)
#     # print()
#     return teams
#
#
# def week_3(teams):
#     # probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jets', 'Browns', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Saints', 'Falcons', 1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Packers', 'Redskins', -2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Eagles', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bills', 'Vikings', 17))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Raiders', 'Dolphins', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Broncos', 'Ravens', 5.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bengals', 'Panthers', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Giants', 'Texans', 6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Titans', 'Jaguars', 10))
#     # probabilities.append(Predictor.predict_game_outcome(teams, '49ers', 'Chiefs', 6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Rams', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cowboys', 'Seahawks', 1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bears', 'Cardinals', -6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Lions', -7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Steelers', 'Buccaneers', -1))
#     #
#     # probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     # for game in probabilities:
#     #     print(game[1])
#     # print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Jets', away_score=17, home_name='Browns', home_score=21,
#                                    away_sacked_yards=8, home_sacked_yards=30, away_penalty_yards=55,
#                                    home_penalty_yards=41, away_convs=1, away_conv_attempts=1, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Saints', 43, 'Falcons', 37, 5, 15, 120, 54, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Packers', 17, 'Redskins', 31, 25, 0, 115, 66, 0, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Colts', 16, 'Eagles', 20, 23, 28, 77, 110, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Bills', 27, 'Vikings', 6, 32, 18, 84, 59, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Raiders', 20, 'Dolphins', 28, 20, 9, 38, 74, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Broncos', 14, 'Ravens', 27, 19, 12, 120, 52, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Bengals', 21, 'Panthers', 31, 22, 3, 36, 17, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, 'Giants', 27, 'Texans', 22, 32, 17, 20, 50, 0, 0, 2, 2)
#     teams = Predictor.update_teams(teams, 'Titans', 9, 'Jaguars', 6, 25, 10, 30, 75, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, '49ers', 27, 'Chiefs', 38, 23, 7, 147, 48, 2, 2, 1, 1)
#     teams = Predictor.update_teams(teams, 'Bears', 16, 'Cardinals', 14, 26, 25, 45, 43, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Chargers', 23, 'Rams', 35, 11, 4, 30, 64, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Cowboys', 13, 'Seahawks', 24, 31, 10, 55, 67, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Patriots', 10, 'Lions', 26, 13, 7, 38, 70, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Steelers', 30, 'Buccaneers', 27, 18, 19, 155, 80, 0, 0, 1, 1)
#
#     return teams
#
#
# def week_4(teams):
#     # probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Vikings', 'Rams', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jets', 'Jaguars', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Dolphins', 'Patriots', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Titans', -4))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Texans', 'Colts', 1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bills', 'Packers', 9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Lions', 'Cowboys', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Buccaneers', 'Bears', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bengals', 'Falcons', 5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Seahawks', 'Cardinals', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Browns', 'Raiders', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Saints', 'Giants', -3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, '49ers', 'Chargers', 10.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Ravens', 'Steelers', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chiefs', 'Broncos', -4.5))
#     #
#     # probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     # for game in probabilities:
#     #     print(game[1])
#     # print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Vikings', away_score=31, home_name='Rams', home_score=38,
#                                    away_sacked_yards=30, home_sacked_yards=9, away_penalty_yards=35,
#                                    home_penalty_yards=15, away_convs=2, away_conv_attempts=2, home_convs=0,
#                                    home_conv_attempts=1)
#     teams = Predictor.update_teams(teams, 'Jets', 12, 'Jaguars', 31, 23, 11, 43, 89, 0, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Dolphins', 7, 'Patriots', 38, 19, 0, 89, 57, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Eagles', 23, 'Titans', 26, 33, 17, 77, 30, 0, 0, 3, 3)
#     teams = Predictor.update_teams(teams, 'Texans', 37, 'Colts', 34, 28, 27, 54, 85, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Bills', 0, 'Packers', 22, 64, 16, 20, 60, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, 'Lions', 24, 'Cowboys', 26, 21, 24, 58, 20, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Buccaneers', 10, 'Bears', 48, 20, 10, 99, 38, 2, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Bengals', 37, 'Falcons', 36, 29, 16, 55, 95, 2, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Seahawks', 20, 'Cardinals', 17, 12, 9, 57, 38, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Browns', 42, 'Raiders', 45, 16, 11, 55, 65, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Saints', 33, 'Giants', 18, 8, 21, 51, 67, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, '49ers', 27, 'Chargers', 29, 10, 8, 29, 49, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Ravens', 26, 'Steelers', 14, 8, 9, 50, 40, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Chiefs', 27, 'Broncos', 23, 0, 19, 93, 25, 2, 2, 2, 3)
#
#     return teams
#
#
# def week_5(teams):
#     # This week is missing in NFL Elo game, should yield 83.0 points
#     # [22.7, -16.7, 8.3, 1.7, 6.8, 3.1, -4.8, 22.3, 15.2, 11.3, -8.4, -17.0, 18.1, 8.8, 11.6]
#
#     # probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Patriots', 10.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Titans', 'Bills', -6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Falcons', 'Steelers', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Broncos', 'Jets', 1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jaguars', 'Chiefs', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Packers', 'Lions', 1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Ravens', 'Browns', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Giants', 'Panthers', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Dolphins', 'Bengals', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Raiders', 'Chargers', 5.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cardinals', '49ers', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Vikings', 'Eagles', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', 'Seahawks', -7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cowboys', 'Texans', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Redskins', 'Saints', 5.5))
#     #
#     # probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     # for game in probabilities:
#     #     print(game[1])
#     # print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Colts', away_score=24, home_name='Patriots', home_score=38,
#                                    away_sacked_yards=10, home_sacked_yards=0, away_penalty_yards=35,
#                                    home_penalty_yards=50, away_convs=2, away_conv_attempts=3, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Titans', 12, 'Bills', 13, 8, 3, 40, 30, 0, 0, 2, 3)
#     teams = Predictor.update_teams(teams, 'Falcons', 17, 'Steelers', 41, 43, 0, 75, 58, 1, 3, 0, 0)
#     teams = Predictor.update_teams(teams, 'Broncos', 16, 'Jets', 34, 33, 9, 45, 33, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Jaguars', 14, 'Chiefs', 30, 29, 15, 45, 105, 3, 5, 0, 0)
#     teams = Predictor.update_teams(teams, 'Packers', 23, 'Lions', 31, 19, 13, 112, 71, 2, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Ravens', 9, 'Browns', 12, 4, 38, 41, 66, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Giants', 31, 'Panthers', 33, 1, 5, 62, 22, 1, 2, 1, 1)
#     teams = Predictor.update_teams(teams, 'Dolphins', 17, 'Bengals', 27, 16, 19, 85, 47, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Raiders', 10, 'Chargers', 26, 20, 6, 55, 82, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Cardinals', 28, '49ers', 18, 6, 49, 46, 65, 0, 0, 1, 2)
#     teams = Predictor.update_teams(teams, 'Vikings', 23, 'Eagles', 21, 3, 28, 23, 52, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Rams', 33, 'Seahawks', 31, 8, 15, 45, 50, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Cowboys', 16, 'Texans', 19, 14, 1, 72, 25, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Redskins', 19, 'Saints', 43, 31, 14, 38, 45, 2, 4, 0, 0)
#
#     return teams
#
#
# def week_6(teams):
#     # probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Giants', -2))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Buccaneers', 'Falcons', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Panthers', 'Redskins', -1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Seahawks', 'Raiders', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Jets', 2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cardinals', 'Vikings', 10))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Steelers', 'Bengals', 1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Browns', 1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bills', 'Texans', 10))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bears', 'Dolphins', -5.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', 'Broncos', -7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Ravens', 'Titans', -2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jaguars', 'Cowboys', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chiefs', 'Patriots', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, '49ers', 'Packers', 9.5))
#     #
#     # probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     # for game in probabilities:
#     #     print(game[1])
#     # print()
#
#     # This game is missing in NFL Elo game, should yield 5.3 points
#     teams = Predictor.update_teams(teams=teams, away_name='Eagles', away_score=34, home_name='Giants', home_score=13,
#                                    away_sacked_yards=7, home_sacked_yards=27, away_penalty_yards=25,
#                                    home_penalty_yards=61, away_convs=0, away_conv_attempts=0, home_convs=0,
#                                    home_conv_attempts=1)
#     teams = Predictor.update_teams(teams, 'Buccaneers', 29, 'Falcons', 34, 6, 8, 20, 30, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Panthers', 17, 'Redskins', 23, 6, 7, 55, 43, 0, 1, 1, 2)
#     teams = Predictor.update_teams(teams, 'Seahawks', 27, 'Raiders', 3, 8, 36, 64, 38, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, 'Colts', 34, 'Jets', 42, 0, 13, 66, 49, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Cardinals', 17, 'Vikings', 27, 32, 17, 30, 52, 0, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Steelers', 28, 'Bengals', 21, 0, 16, 69, 30, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Chargers', 38, 'Browns', 14, 12, 24, 72, 53, 0, 0, 1, 3)
#     teams = Predictor.update_teams(teams, 'Bills', 13, 'Texans', 20, 16, 35, 104, 50, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Bears', 28, 'Dolphins', 31, 13, 0, 58, 67, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Rams', 23, 'Broncos', 20, 27, 25, 81, 61, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Ravens', 21, 'Titans', 0, 0, 66, 70, 35, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Jaguars', 7, 'Cowboys', 40, 10, 11, 30, 25, 0, 1, 2, 2)
#     teams = Predictor.update_teams(teams, 'Chiefs', 40, 'Patriots', 43, 0, 13, 58, 0, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, '49ers', 30, 'Packers', 33, 18, 20, 10, 54, 0, 0, 1, 2)
#
#     return teams
#
#
# def week_7(teams):
#     # probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Broncos', 'Cardinals', -1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Titans', 'Chargers', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Texans', 'Jaguars', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Panthers', 'Eagles', 5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Vikings', 'Jets', -3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Bears', -1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bills', 'Colts', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Browns', 'Buccaneers', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Lions', 'Dolphins', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Saints', 'Ravens', 2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cowboys', 'Redskins', -1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', '49ers', -9))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bengals', 'Chiefs', 6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Giants', 'Falcons', 4))
#     #
#     # probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     # for game in probabilities:
#     #     print(game[1])
#     # print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Broncos', away_score=45, home_name='Cardinals',
#                                    home_score=10, away_sacked_yards=11, home_sacked_yards=40, away_penalty_yards=50,
#                                    home_penalty_yards=45, away_convs=0, away_conv_attempts=0, home_convs=0,
#                                    home_conv_attempts=1)
#     teams = Predictor.update_teams(teams, 'Titans', 19, 'Chargers', 20, 11, 9, 32, 31, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Texans', 20, 'Jaguars', 7, 8, 28, 35, 15, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, 'Panthers', 21, 'Eagles', 17, 19, 26, 78, 35, 1, 1, 3, 4)
#     teams = Predictor.update_teams(teams, 'Vikings', 37, 'Jets', 17, 13, 14, 55, 71, 1, 1, 1, 2)
#     teams = Predictor.update_teams(teams, 'Patriots', 38, 'Bears', 31, 4, 14, 64, 40, 1, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Bills', 5, 'Colts', 37, 7, 0, 59, 35, 0, 1, 1, 2)
#     teams = Predictor.update_teams(teams, 'Browns', 23, 'Buccaneers', 26, 29, 23, 114, 65, 0, 2, 1, 1)
#     teams = Predictor.update_teams(teams, 'Lions', 32, 'Dolphins', 21, 8, 24, 43, 50, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Saints', 24, 'Ravens', 23, 7, 10, 52, 54, 4, 5, 1, 2)
#     teams = Predictor.update_teams(teams, 'Cowboys', 17, 'Redskins', 20, 23, 3, 65, 35, 1, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Rams', 39, '49ers', 10, 17, 49, 10, 10, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, 'Bengals', 10, 'Chiefs', 45, 13, 5, 58, 50, 0, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Giants', 20, 'Falcons', 23, 27, 23, 58, 29, 0, 1, 0, 0)
#
#     return teams
#
#
# def week_8(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Dolphins', 'Texans', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Jaguars', -4))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Broncos', 'Chiefs', 9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Browns', 'Steelers', 8.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Redskins', 'Giants', -1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Seahawks', 'Lions', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Buccaneers', 'Bengals', 4))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jets', 'Bears', 9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Ravens', 'Panthers', -2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Raiders', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, '49ers', 'Cardinals', -1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Packers', 'Rams', 8))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Saints', 'Vikings', -1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Bills', -14))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Dolphins', away_score=23, home_name='Texans',
#                                    home_score=42, away_sacked_yards=15, home_sacked_yards=0, away_penalty_yards=63,
#                                    home_penalty_yards=57, away_convs=0, away_conv_attempts=1, home_convs=1,
#                                    home_conv_attempts=1)
#     teams = Predictor.update_teams(teams, 'Eagles', 24, 'Jaguars', 18, 24, 21, 36, 45, 0, 0, 1, 2)
#     teams = Predictor.update_teams(teams, 'Broncos', 23, 'Chiefs', 30, 40, 12, 83, 50, 1, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Browns', 18, 'Steelers', 33, 17, 4, 52, 60, 1, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Redskins', 20, 'Giants', 13, 0, 50, 90, 102, 0, 0, 3, 4)
#     teams = Predictor.update_teams(teams, 'Seahawks', 28, 'Lions', 14, 11, 13, 111, 32, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Buccaneers', 34, 'Bengals', 37, 20, 16, 75, 46, 2, 2, 1, 2)
#     teams = Predictor.update_teams(teams, 'Jets', 10, 'Bears', 24, 3, 4, 45, 25, 1, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Ravens', 21, 'Panthers', 36, 14, 0, 68, 30, 1, 2, 1, 1)
#     teams = Predictor.update_teams(teams, 'Colts', 42, 'Raiders', 28, 0, 0, 77, 79, 0, 0, 1, 2)
#     teams = Predictor.update_teams(teams, '49ers', 15, 'Cardinals', 18, 30, 19, 59, 62, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Packers', 27, 'Rams', 29, 33, 26, 10, 15, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Saints', 30, 'Vikings', 20, 0, 21, 64, 54, 0, 0, 2, 3)
#     teams = Predictor.update_teams(teams, 'Patriots', 25, 'Bills', 6, 13, 26, 33, 51, 0, 0, 0, 2)
#
#     return teams
#
#
# def week_9(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Raiders', '49ers', -1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bears', 'Bills', -10))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Buccaneers', 'Panthers', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chiefs', 'Browns', -8))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jets', 'Dolphins', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Steelers', 'Ravens', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Lions', 'Vikings', 5.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Falcons', 'Redskins', 2))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Texans', 'Broncos', 1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Seahawks', 0))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', 'Saints', -2))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Packers', 'Patriots', 5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Titans', 'Cowboys', 5))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Raiders', away_score=3, home_name='49ers', home_score=34,
#                                    away_sacked_yards=39, home_sacked_yards=0, away_penalty_yards=40,
#                                    home_penalty_yards=23, away_convs=0, away_conv_attempts=1, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Bears', 41, 'Bills', 9, 9, 22, 129, 163, 0, 0, 1, 3)
#     teams = Predictor.update_teams(teams, 'Buccaneers', 28, 'Panthers', 42, 24, 19, 49, 84, 1, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Chiefs', 37, 'Browns', 21, 15, 22, 86, 20, 0, 0, 2, 3)
#     teams = Predictor.update_teams(teams, 'Jets', 6, 'Dolphins', 13, 27, 35, 45, 55, 0, 2, 2, 2)
#     teams = Predictor.update_teams(teams, 'Steelers', 23, 'Ravens', 16, 10, 14, 103, 25, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Lions', 9, 'Vikings', 24, 56, 9, 86, 15, 3, 3, 1, 1)
#     teams = Predictor.update_teams(teams, 'Falcons', 38, 'Redskins', 14, 13, 19, 50, 147, 0, 0, 2, 2)
#     teams = Predictor.update_teams(teams, 'Texans', 19, 'Broncos', 17, 21, 17, 60, 50, 0, 1, 2, 2)
#     teams = Predictor.update_teams(teams, 'Chargers', 25, 'Seahawks', 17, 13, 33, 105, 83, 0, 0, 3, 3)
#     teams = Predictor.update_teams(teams, 'Rams', 35, 'Saints', 45, 0, 0, 32, 20, 0, 2, 2, 2)
#     teams = Predictor.update_teams(teams, 'Packers', 17, 'Patriots', 31, 9, 21, 63, 30, 0, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Titans', 28, 'Cowboys', 14, 25, 18, 20, 52, 0, 0, 0, 1)
#
#     return teams
#
#
# def week_10(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Panthers', 'Steelers', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Saints', 'Bengals', -6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Falcons', 'Browns', -5.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Lions', 'Bears', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cardinals', 'Chiefs', 15.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Titans', -6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Redskins', 'Buccaneers', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bills', 'Jets', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jaguars', 'Colts', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Raiders', -10))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Seahawks', 'Rams', 9))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Dolphins', 'Packers', 10.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cowboys', 'Eagles', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Giants', '49ers', 3))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Panthers', away_score=21, home_name='Steelers',
#                                    home_score=52, away_sacked_yards=46, home_sacked_yards=6, away_penalty_yards=42,
#                                    home_penalty_yards=24, away_convs=1, away_conv_attempts=1, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Saints', 51, 'Bengals', 14, 0, 24, 5, 26, 1, 1, 1, 2)
#     teams = Predictor.update_teams(teams, 'Falcons', 16, 'Browns', 28, 19, 0, 5, 38, 2, 4, 0, 0)
#     teams = Predictor.update_teams(teams, 'Lions', 22, 'Bears', 34, 45, 7, 41, 46, 4, 5, 0, 0)
#     teams = Predictor.update_teams(teams, 'Cardinals', 14, 'Chiefs', 26, 42, 37, 30, 63, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Patriots', 10, 'Titans', 34, 23, 14, 31, 35, 2, 3, 0, 0)
#     teams = Predictor.update_teams(teams, 'Redskins', 16, 'Buccaneers', 3, 8, 8, 52, 50, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Bills', 41, 'Jets', 10, 8, 19, 30, 10, 1, 1, 2, 3)
#     teams = Predictor.update_teams(teams, 'Jaguars', 26, 'Colts', 29, 0, 0, 61, 45, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Chargers', 20, 'Raiders', 6, 1, 40, 70, 55, 1, 1, 1, 3)
#     teams = Predictor.update_teams(teams, 'Seahawks', 31, 'Rams', 36, 35, 11, 56, 102, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Dolphins', 12, 'Packers', 31, 50, 17, 10, 45, 3, 4, 1, 2)
#     teams = Predictor.update_teams(teams, 'Cowboys', 27, 'Eagles', 20, 31, 10, 37, 0, 1, 1, 0, 2)
#     teams = Predictor.update_teams(teams, 'Giants', 27, '49ers', 23, 8, 0, 78, 97, 0, 0, 0, 0)
#     return teams
#
#
# def week_11(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Packers', 'Seahawks', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bengals', 'Ravens', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cowboys', 'Falcons', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Buccaneers', 'Giants', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Steelers', 'Jaguars', -4.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Texans', 'Redskins', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Titans', 'Colts', 1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Panthers', 'Lions', -4))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Broncos', 'Chargers', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Raiders', 'Cardinals', 5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Saints', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Vikings', 'Bears', 2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chiefs', 'Rams', 3.5))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Packers', away_score=24, home_name='Seahawks',
#                                    home_score=27, away_sacked_yards=21, home_sacked_yards=20, away_penalty_yards=80,
#                                    home_penalty_yards=30, away_convs=0, away_conv_attempts=0, home_convs=2,
#                                    home_conv_attempts=2)
#     teams = Predictor.update_teams(teams, 'Bengals', 21, 'Ravens', 24, 4, 12, 58, 31, 0, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Cowboys', 22, 'Falcons', 19, 17, 17, 20, 15, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Buccaneers', 35, 'Giants', 38, 7, 35, 40, 80, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Steelers', 20, 'Jaguars', 16, 17, 40, 15, 111, 0, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Texans', 23, 'Redskins', 21, 27, 35, 40, 43, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Titans', 10, 'Colts', 38, 27, 0, 112, 60, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Panthers', 19, 'Lions', 20, 26, 5, 15, 5, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Broncos', 23, 'Chargers', 22, 0, 17, 43, 120, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Raiders', 23, 'Cardinals', 21, 19, 8, 20, 62, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Eagles', 7, 'Saints', 48, 18, 0, 49, 69, 0, 2, 1, 1)
#     teams = Predictor.update_teams(teams, 'Vikings', 20, 'Bears', 25, 16, 5, 50, 46, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Chiefs', 51, 'Rams', 54, 30, 34, 135, 60, 1, 1, 1, 1)
#
#     return teams
#
#
# def week_12(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bears', 'Lions', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Redskins', 'Cowboys', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Falcons', 'Saints', 13))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Browns', 'Bengals', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, '49ers', 'Buccaneers', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jaguars', 'Bills', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Raiders', 'Ravens', 10.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Seahawks', 'Panthers', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Jets', -9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Giants', 'Eagles', 6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cardinals', 'Chargers', 12))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Dolphins', 'Colts', 9))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Steelers', 'Broncos', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Packers', 'Vikings', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Titans', 'Texans', 3.5))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Bears', away_score=23, home_name='Lions',
#                                    home_score=16, away_sacked_yards=12, home_sacked_yards=14, away_penalty_yards=20,
#                                    home_penalty_yards=54, away_convs=0, away_conv_attempts=0, home_convs=1,
#                                    home_conv_attempts=2)
#     teams = Predictor.update_teams(teams, 'Redskins', 23, 'Cowboys', 31, 17, 31, 25, 30, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, 'Falcons', 17, 'Saints', 31, 37, 9, 26, 19, 4, 6, 0, 0)
#     teams = Predictor.update_teams(teams, 'Browns', 35, 'Bengals', 20, 0, 12, 61, 96, 1, 1, 2, 4)
#     teams = Predictor.update_teams(teams, '49ers', 9, 'Buccaneers', 27, 27, 8, 68, 70, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Jaguars', 21, 'Bills', 24, 20, 0, 90, 80, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Raiders', 17, 'Ravens', 34, 12, 4, 24, 45, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Seahawks', 30, 'Panthers', 27, 17, 0, 36, 30, 2, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Patriots', 27, 'Jets', 13, 0, 12, 105, 47, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Giants', 22, 'Eagles', 25, 21, 22, 91, 46, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Cardinals', 10, 'Chargers', 45, 18, 23, 35, 30, 0, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Dolphins', 24, 'Colts', 27, 3, 10, 75, 52, 0, 0, 2, 2)
#     teams = Predictor.update_teams(teams, 'Steelers', 17, 'Broncos', 24, 12, 13, 34, 50, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Packers', 17, 'Vikings', 24, 26, 17, 20, 55, 0, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Titans', 17, 'Texans', 34, 43, 29, 50, 53, 0, 1, 1, 1)
#
#     return teams
#
#
# def week_13(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Saints', 'Cowboys', -7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Ravens', 'Falcons', 2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Panthers', 'Buccaneers', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bears', 'Giants', -4))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bills', 'Dolphins', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Jaguars', -4))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Browns', 'Texans', 5.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Broncos', 'Bengals', -4.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', 'Lions', -10))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cardinals', 'Packers', 13.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chiefs', 'Raiders', -14))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jets', 'Titans', 9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, '49ers', 'Seahawks', 9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Vikings', 'Patriots', 5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Steelers', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Redskins', 'Eagles', 5.5))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Saints', away_score=10, home_name='Cowboys',
#                                    home_score=13, away_sacked_yards=16, home_sacked_yards=40, away_penalty_yards=48,
#                                    home_penalty_yards=80, away_convs=0, away_conv_attempts=1, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Ravens', 26, 'Falcons', 16, 8, 34, 88, 82, 2, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Panthers', 17, 'Buccaneers', 24, 24, 29, 104, 69, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Bears', 27, 'Giants', 30, 28, 22, 58, 107, 4, 5, 2, 2)
#     teams = Predictor.update_teams(teams, 'Bills', 17, 'Dolphins', 21, 14, 22, 120, 89, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Colts', 0, 'Jaguars', 6, 24, 18, 65, 74, 0, 3, 0, 0)
#     teams = Predictor.update_teams(teams, 'Browns', 13, 'Texans', 29, 0, 27, 45, 77, 0, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Broncos', 24, 'Bengals', 10, 8, 36, 60, 100, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Rams', 30, 'Lions', 16, 12, 37, 105, 54, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Cardinals', 20, 'Packers', 17, 16, 6, 49, 43, 0, 0, 2, 2)
#     teams = Predictor.update_teams(teams, 'Chiefs', 40, 'Raiders', 33, 0, 14, 94, 74, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Jets', 22, 'Titans', 26, 4, 9, 96, 75, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, '49ers', 16, 'Seahawks', 43, 28, 22, 128, 100, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Vikings', 10, 'Patriots', 24, 18, 0, 55, 60, 1, 2, 1, 1)
#     teams = Predictor.update_teams(teams, 'Chargers', 33, 'Steelers', 30, 13, 10, 80, 59, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Redskins', 13, 'Eagles', 28, 19, 0, 69, 38, 0, 0, 0, 1)
#
#     return teams
#
#
# def week_14(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jaguars', 'Titans', 4.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jets', 'Bills', 4.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Giants', 'Redskins', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Saints', 'Buccaneers', -9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Dolphins', -9))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Ravens', 'Chiefs', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Texans', 4))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Falcons', 'Packers', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Panthers', 'Browns', 0))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Broncos', '49ers', -3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bengals', 'Chargers', 15.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Lions', 'Cardinals', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Steelers', 'Raiders', -10))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Cowboys', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', 'Bears', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Vikings', 'Seahawks', 3))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     # This game is missing in NFL Elo game, should yield 17.7 points
#     teams = Predictor.update_teams(teams=teams, away_name='Jaguars', away_score=9, home_name='Titans',
#                                    home_score=30, away_sacked_yards=45, home_sacked_yards=0, away_penalty_yards=67,
#                                    home_penalty_yards=0, away_convs=2, away_conv_attempts=5, home_convs=0,
#                                    home_conv_attempts=1)
#     teams = Predictor.update_teams(teams, 'Jets', 27, 'Bills', 23, 0, 14, 93, 47, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Giants', 40, 'Redskins', 16, 22, 29, 18, 135, 0, 0, 0, 1)
#     teams = Predictor.update_teams(teams, 'Saints', 28, 'Buccaneers', 14, 3, 39, 51, 84, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Patriots', 33, 'Dolphins', 34, 14, 42, 30, 81, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Ravens', 24, 'Chiefs', 27, 24, 30, 112, 53, 1, 3, 3, 3)
#     teams = Predictor.update_teams(teams, 'Colts', 24, 'Texans', 21, 13, 41, 59, 35, 0, 0, 2, 2)
#     teams = Predictor.update_teams(teams, 'Falcons', 20, 'Packers', 34, 25, 34, 101, 37, 3, 3, 0, 0)
#     teams = Predictor.update_teams(teams, 'Panthers', 20, 'Browns', 26, 1, 6, 68, 52, 2, 3, 0, 0)
#     teams = Predictor.update_teams(teams, 'Broncos', 14, '49ers', 20, 15, 27, 62, 87, 5, 7, 0, 1)
#     teams = Predictor.update_teams(teams, 'Bengals', 21, 'Chargers', 26, 19, 17, 34, 38, 0, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Lions', 17, 'Cardinals', 3, 5, 22, 0, 0, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Steelers', 21, 'Raiders', 24, 6, 23, 14, 130, 0, 1, 2, 2)
#     teams = Predictor.update_teams(teams, 'Eagles', 23, 'Cowboys', 29, 6, 21, 49, 111, 1, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Rams', 6, 'Bears', 15, 25, 10, 57, 45, 1, 3, 0, 0)
#     teams = Predictor.update_teams(teams, 'Vikings', 7, 'Seahawks', 21, 9, 12, 51, 45, 0, 2, 1, 1)
#
#     return teams
#
#
# def week_15(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Chiefs', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Texans', 'Jets', -7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Browns', 'Broncos', 1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Packers', 'Bears', 5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Lions', 'Bills', 2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Buccaneers', 'Ravens', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cardinals', 'Falcons', 9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Raiders', 'Bengals', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Titans', 'Giants', -1))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Dolphins', 'Vikings', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Redskins', 'Jaguars', 7.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cowboys', 'Colts', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Seahawks', '49ers', -3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Steelers', -2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Rams', 12.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Saints', 'Panthers', -6))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Chargers', away_score=29, home_name='Chiefs',
#                                    home_score=28, away_sacked_yards=25, home_sacked_yards=9, away_penalty_yards=50,
#                                    home_penalty_yards=66, away_convs=1, away_conv_attempts=1, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Texans', 29, 'Jets', 22, 55, 25, 43, 60, 0, 0, 1, 3)
#     teams = Predictor.update_teams(teams, 'Browns', 17, 'Broncos', 16, 13, 19, 75, 51, 1, 2, 1, 2)
#     teams = Predictor.update_teams(teams, 'Packers', 17, 'Bears', 24, 39, 3, 48, 24, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Lions', 13, 'Bills', 14, 0, 9, 81, 20, 0, 0, 1, 2)
#     teams = Predictor.update_teams(teams, 'Buccaneers', 12, 'Ravens', 20, 1, 3, 58, 70, 0, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Cardinals', 14, 'Falcons', 40, 50, 11, 75, 50, 2, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Raiders', 16, 'Bengals', 30, 34, 17, 90, 85, 0, 0, 2, 2)
#     teams = Predictor.update_teams(teams, 'Titans', 17, 'Giants', 0, 8, 16, 35, 58, 1, 1, 2, 3)
#     teams = Predictor.update_teams(teams, 'Dolphins', 17, 'Vikings', 41, 71, 17, 49, 45, 0, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Redskins', 16, 'Jaguars', 13, 15, 37, 48, 55, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Cowboys', 0, 'Colts', 23, 26, 0, 74, 55, 1, 5, 0, 0)
#     teams = Predictor.update_teams(teams, 'Seahawks', 23, '49ers', 26, 20, 18, 148, 66, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Patriots', 10, 'Steelers', 17, 7, 17, 106, 40, 0, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Eagles', 30, 'Rams', 23, 0, 8, 49, 50, 0, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Saints', 12, 'Panthers', 9, 12, 32, 80, 48, 1, 1, 1, 2)
#
#     return teams
#
#
# def week_16(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Redskins', 'Titans', 10.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Ravens', 'Chargers', 4))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bengals', 'Browns', 10))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Buccaneers', 'Cowboys', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Vikings', 'Lions', -6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bills', 'Patriots', 13.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Packers', 'Jets', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Texans', 'Eagles', 1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Falcons', 'Panthers', -3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Giants', 'Colts', 9.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jaguars', 'Dolphins', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', 'Cardinals', -14))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bears', '49ers', -4))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Steelers', 'Saints', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chiefs', 'Seahawks', -2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Broncos', 'Raiders', -3))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Redskins', away_score=16, home_name='Titans',
#                                    home_score=25, away_sacked_yards=22, home_sacked_yards=19, away_penalty_yards=49,
#                                    home_penalty_yards=10, away_convs=0, away_conv_attempts=0, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Ravens', 22, 'Chargers', 10, 2, 34, 40, 69, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Bengals', 18, 'Browns', 26, 26, 0, 15, 57, 1, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Buccaneers', 20, 'Cowboys', 27, 16, 9, 72, 67, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Vikings', 27, 'Lions', 9, 13, 24, 78, 55, 0, 1, 1, 3)
#     teams = Predictor.update_teams(teams, 'Bills', 12, 'Patriots', 24, 0, 9, 29, 49, 0, 1, 1, 2)
#     teams = Predictor.update_teams(teams, 'Packers', 44, 'Jets', 38, 29, 18, 86, 172, 2, 2, 2, 2)
#     teams = Predictor.update_teams(teams, 'Texans', 30, 'Eagles', 32, 30, 9, 61, 105, 0, 1, 4, 4)
#     teams = Predictor.update_teams(teams, 'Falcons', 24, 'Panthers', 10, 6, 17, 41, 86, 0, 0, 0, 2)
#     teams = Predictor.update_teams(teams, 'Giants', 27, 'Colts', 28, 0, 4, 24, 29, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Jaguars', 17, 'Dolphins', 7, 27, 25, 97, 95, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Rams', 31, 'Cardinals', 9, 24, 23, 66, 54, 0, 0, 1, 2)
#     teams = Predictor.update_teams(teams, 'Bears', 14, '49ers', 9, 11, 9, 30, 45, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Steelers', 28, 'Saints', 31, 16, 13, 79, 91, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Chiefs', 31, 'Seahawks', 38, 8, 17, 76, 20, 1, 1, 0, 0)
#     teams = Predictor.update_teams(teams, 'Broncos', 14, 'Raiders', 27, 2, 8, 91, 50, 1, 2, 0, 0)
#
#     return teams
#
#
# def week_17(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Dolphins', 'Bills', 5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Falcons', 'Buccaneers', -2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Panthers', 'Saints', 8))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cowboys', 'Giants', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Lions', 'Packers', 8.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jaguars', 'Texans', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Jets', 'Patriots', 14))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Raiders', 'Chiefs', 14))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Browns', 'Ravens', 6.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Redskins', -6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bears', 'Vikings', 6))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Bengals', 'Steelers', 14))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Broncos', -7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, '49ers', 'Rams', 10.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cardinals', 'Seahawks', 14))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Titans', -4.5))
#
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Dolphins', away_score=17, home_name='Bills',
#                                    home_score=42, away_sacked_yards=24, home_sacked_yards=9, away_penalty_yards=35,
#                                    home_penalty_yards=24, away_convs=1, away_conv_attempts=1, home_convs=1,
#                                    home_conv_attempts=2)
#     teams = Predictor.update_teams(teams, 'Falcons', 34, 'Buccaneers', 32, 3, 4, 85, 30, 0, 0, 0, 0)
#     teams = Predictor.update_teams(teams, 'Panthers', 33, 'Saints', 14, 5, 8, 10, 94, 1, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Cowboys', 36, 'Giants', 35, 19, 3, 46, 78, 2, 2, 2, 3)
#     teams = Predictor.update_teams(teams, 'Lions', 31, 'Packers', 0, 2, 29, 34, 38, 1, 1, 1, 3)
#     teams = Predictor.update_teams(teams, 'Jaguars', 3, 'Texans', 20, 18, 26, 68, 15, 0, 0, 1, 1)
#     teams = Predictor.update_teams(teams, 'Jets', 3, 'Patriots', 38, 32, 6, 30, 30, 0, 2, 0, 1)
#     teams = Predictor.update_teams(teams, 'Raiders', 3, 'Chiefs', 35, 20, 0, 22, 50, 1, 1, 1, 3)
#     teams = Predictor.update_teams(teams, 'Browns', 24, 'Ravens', 26, 0, 12, 35, 65, 0, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Eagles', 24, 'Redskins', 0, 12, 30, 25, 15, 1, 1, 2, 4)
#     teams = Predictor.update_teams(teams, 'Bears', 24, 'Vikings', 10, 0, 31, 102, 30, 0, 1, 1, 4)
#     teams = Predictor.update_teams(teams, 'Bengals', 13, 'Steelers', 16, 24, 9, 85, 70, 0, 1, 2, 2)
#     teams = Predictor.update_teams(teams, 'Chargers', 23, 'Broncos', 9, 16, 4, 51, 99, 0, 0, 1, 2)
#     teams = Predictor.update_teams(teams, '49ers', 32, 'Rams', 48, 18, 0, 41, 57, 1, 1, 0, 1)
#     teams = Predictor.update_teams(teams, 'Cardinals', 24, 'Seahawks', 27, 36, 43, 20, 10, 1, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Colts', 33, 'Titans', 17, 7, 0, 96, 75, 1, 1, 0, 1)
#
#     return teams
#
#
# def wildcard(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Texans', 1.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Seahawks', 'Cowboys', 2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Ravens', 2.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Bears', 6.5))
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Colts', away_score=21, home_name='Texans',
#                                    home_score=7, away_sacked_yards=0, home_sacked_yards=18, away_penalty_yards=10,
#                                    home_penalty_yards=67, away_convs=0, away_conv_attempts=0, home_convs=2,
#                                    home_conv_attempts=5)
#     teams = Predictor.update_teams(teams, 'Seahawks', 22, 'Cowboys', 24, 7, 10, 36, 36, 2, 2, 0, 0)
#     teams = Predictor.update_teams(teams, 'Chargers', 23, 'Ravens', 17, 6, 55, 35, 41, 1, 1, 2, 2)
#     teams = Predictor.update_teams(teams, 'Eagles', 16, 'Bears', 15, 8, 12, 25, 52, 1, 1, 0, 0)
#
#     return teams
#
#
# def divisional(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Colts', 'Chiefs', 4.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Cowboys', 'Rams', 7))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Chargers', 'Patriots', 3.5))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Eagles', 'Saints', 8.5))
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Colts', away_score=13, home_name='Chiefs',
#                                    home_score=31, away_sacked_yards=27, home_sacked_yards=25, away_penalty_yards=70,
#                                    home_penalty_yards=54, away_convs=1, away_conv_attempts=1, home_convs=3,
#                                    home_conv_attempts=4)
#     teams = Predictor.update_teams(teams, 'Cowboys', 22, 'Rams', 30, 8, 0, 26, 41, 3, 4, 2, 2)
#     teams = Predictor.update_teams(teams, 'Chargers', 28, 'Patriots', 41, 15, 0, 33, 75, 1, 1, 1, 1)
#     teams = Predictor.update_teams(teams, 'Eagles', 14, 'Saints', 20, 0, 18, 30, 84, 0, 0, 2, 2)
#
#     return teams
#
#
# def conference(teams):
#     probabilities = list()
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', 'Saints', 3))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Chiefs', 3))
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Rams', away_score=26, home_name='Saints',
#                                    home_score=23, away_sacked_yards=8, home_sacked_yards=7, away_penalty_yards=64,
#                                    home_penalty_yards=20, away_convs=1, away_conv_attempts=1, home_convs=0,
#                                    home_conv_attempts=0)
#     teams = Predictor.update_teams(teams, 'Patriots', 37, 'Chiefs', 31, 0, 46, 61, 28, 1, 2, 0, 0)
#
#     return teams
#
#
# def superbowl(teams):
#     probabilities = list()
#     spread = -2.5
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Patriots', 'Rams', spread))
#     # probabilities.append(Predictor.predict_game_outcome(teams, 'Rams', 'Patriots', -1 * spread))
#     probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
#     for game in probabilities:
#         print(game[1])
#     print()
#
#     teams = Predictor.update_teams(teams=teams, away_name='Patriots', away_score=13, home_name='Rams',
#                                    home_score=3, away_sacked_yards=9, home_sacked_yards=31, away_penalty_yards=20,
#                                    home_penalty_yards=65, away_convs=0, away_conv_attempts=1, home_convs=0,
#                                    home_conv_attempts=0)
#
#     return teams


def print_elo_rankings(teams, eliminated_teams):
    sorted_by_losses = sorted(teams, key=lambda tup: tup[2])
    sorted_by_wins = sorted(sorted_by_losses, reverse=True, key=lambda tup: tup[1])
    sorted_by_elo = sorted(sorted_by_wins, reverse=True, key=lambda tup: tup[4])
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Tier'])
    for rank, team in enumerate(sorted_by_elo):
        row = list()
        row.append(rank + 1)
        team_info = list()
        team_info.append(team[0])
        team_info.append(team[1])
        team_info.append(team[2])
        team_info.append(team[3])
        team_info.append(team[4])
        team_info.append(get_tier(teams, team))
        row = row + team_info
        row = [get_tier_color(row[-1], val) for val in row]
        if team[0] not in eliminated_teams:
            table.add_row(row)
    print('Elo Rankings')
    print(table)
    print()


def print_standings(teams, eliminated_teams):
    sorted_by_yards = sorted(teams, reverse=True, key=lambda tup: tup[13])
    sorted_by_point_diff = sorted(sorted_by_yards, reverse=True, key=lambda tup: tup[5] - tup[6])
    sorted_by_losses = sorted(sorted_by_point_diff, key=lambda tup: tup[2])
    sorted_by_wins = sorted(sorted_by_losses, reverse=True, key=lambda tup: tup[1])
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo'])
    for rank, team in enumerate(sorted_by_wins):
        row = list()
        row.append(rank + 1)
        team_info = list()
        team_info.append(team[0])
        team_info.append(team[1])
        team_info.append(team[2])
        team_info.append(team[3])
        team_info.append(team[4])
        row = row + team_info
        if team[0] not in eliminated_teams:
            table.add_row(row)
    print('Standings')
    print(table)
    print()


def print_full_standings(teams, eliminated_teams):
    sorted_by_yards = sorted(teams, reverse=True, key=lambda tup: tup[13])
    sorted_by_point_diff = sorted(sorted_by_yards, reverse=True, key=lambda tup: tup[5] - tup[6])
    sorted_by_losses = sorted(sorted_by_point_diff, key=lambda tup: tup[2])
    sorted_by_wins = sorted(sorted_by_losses, reverse=True, key=lambda tup: tup[1])
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Points', 'Points Against', 'Point Diff.',
                         'Touchdowns', 'Passer Rating', 'Total Yards', 'First Downs', '3rd Down %'])
    for rank, team in enumerate(sorted_by_wins):
        row = list()
        row.append(rank + 1)
        team_info = list()
        team_info.append(team[0])
        team_info.append(team[1])
        team_info.append(team[2])
        team_info.append(team[3])
        team_info.append(team[4])
        team_info.append(round(team[5], 1))
        team_info.append(round(team[6], 1))
        team_info.append(round(team[5] - team[6], 1))
        team_info.append(round(team[7], 1))
        average_completion_pct = team[9] / team[10] if team[10] > 0 else 0
        a = (average_completion_pct - .3) * 5
        average_yards_per_pass_attempt = team[8] / team[10] if team[10] > 0 else 0
        b = (average_yards_per_pass_attempt - 3) * .25
        average_passing_touchdowns_per_attempt = team[11] / team[10] if team[10] > 0 else 0
        c = average_passing_touchdowns_per_attempt * 20
        average_ints_per_attempt = team[12] / team[10] if team[10] > 0 else 0
        d = 2.375 - (average_ints_per_attempt * 25)
        passer_rating = ((a + b + c + d) / 6) * 100
        team_info.append(round(passer_rating, 2))
        team_info.append(round(team[13], 1))
        team_info.append(round(team[14], 1))
        if team[16] == 0:
            team_info.append('--')
        else:
            team_info.append(round(100 * (team[15] / team[16]), 1))
        row = row + team_info
        if team[0] not in eliminated_teams:
            table.add_row(row)
    print('Standings')
    print(table)
    print()


def print_league_details(teams, eliminated_teams, full_standings=False):
    if full_standings:
        print_full_standings(teams, eliminated_teams)
    else:
        print_standings(teams, eliminated_teams)
    print_elo_rankings(teams, eliminated_teams)


def get_team(teams, team_name):
    for team in teams:
        if team[0] == team_name:
            return team


def get_tier(teams, team):
    team_elo = team[4]
    elos = [team_info[4] for team_info in teams]
    avg_elo = statistics.mean(elos)
    elo_dev = statistics.stdev(elos)
    elo_dev_third = elo_dev / 3

    if team_elo > avg_elo + elo_dev_third * 8:
        tier = 'S+'
    elif team_elo > avg_elo + elo_dev_third * 7:
        tier = 'S'
    elif team_elo > avg_elo + elo_dev_third * 6:
        tier = 'S-'
    elif team_elo > avg_elo + elo_dev_third * 5:
        tier = 'A+'
    elif team_elo > avg_elo + elo_dev_third * 4:
        tier = 'A'
    elif team_elo > avg_elo + elo_dev_third * 3:
        tier = 'A-'
    elif team_elo > avg_elo + elo_dev_third * 2:
        tier = 'B+'
    elif team_elo > avg_elo + elo_dev_third * 1:
        tier = 'B'
    elif team_elo >= avg_elo:
        tier = 'B-'
    elif team_elo > avg_elo - elo_dev_third * 1:
        tier = 'C+'
    elif team_elo > avg_elo - elo_dev_third * 2:
        tier = 'C'
    elif team_elo > avg_elo - elo_dev_third * 3:
        tier = 'C-'
    elif team_elo > avg_elo - elo_dev_third * 4:
        tier = 'D+'
    elif team_elo > avg_elo - elo_dev_third * 5:
        tier = 'D'
    elif team_elo > avg_elo - elo_dev_third * 6:
        tier = 'D-'
    elif team_elo > avg_elo - elo_dev_third * 7:
        tier = 'F+'
    elif team_elo > avg_elo - elo_dev_third * 8:
        tier = 'F'
    else:
        tier = 'F-'
    return tier


def get_tier_color(tier, string):
    if tier == 'S+':
        string = Colors.UNDERLINE + str(string) + Colors.ENDC
    elif tier == 'S':
        string = Colors.UNDERLINE + str(string) + Colors.ENDC
    elif tier == 'S-':
        string = Colors.WHITE + str(string) + Colors.ENDC
    elif tier == 'A+':
        string = Colors.BRIGHT_PURPLE + str(string) + Colors.ENDC
    elif tier == 'A':
        string = Colors.PURPLE + str(string) + Colors.ENDC
    elif tier == 'A-':
        string = Colors.BRIGHT_BLUE + str(string) + Colors.ENDC
    elif tier == 'B+':
        string = Colors.BLUE + str(string) + Colors.ENDC
    elif tier == 'B':
        string = Colors.BRIGHT_CYAN + str(string) + Colors.ENDC
    elif tier == 'B-':
        string = Colors.CYAN + str(string) + Colors.ENDC
    elif tier == 'C+':
        string = Colors.GREEN + str(string) + Colors.ENDC
    elif tier == 'C':
        string = Colors.BRIGHT_YELLOW + str(string) + Colors.ENDC
    elif tier == 'C-':
        string = Colors.YELLOW + str(string) + Colors.ENDC
    elif tier == 'D+':
        string = Colors.BRIGHT_RED + str(string) + Colors.ENDC
    elif tier == 'D':
        string = Colors.RED + str(string) + Colors.ENDC
    elif tier == 'D-':
        string = Colors.BRIGHT_GRAY + str(string) + Colors.ENDC
    elif tier == 'F+':
        string = Colors.GRAY + str(string) + Colors.ENDC
    elif tier == 'F':
        string = Colors.BLACK + str(string) + Colors.ENDC
    elif tier == 'F-':
        string = Colors.BLACK + str(string) + Colors.ENDC
    return string


class Colors:
    BLACK = '\033[97m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_PURPLE = '\033[95m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GRAY = '\033[37m'
    GRAY = '\033[90m'
    WHITE = '\033[30m'
    UNDERLINE = '\033[4m'
    CYAN = '\033[36m'
    PURPLE = '\033[35m'
    BLUE = '\033[34m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    ENDC = '\033[0m'
