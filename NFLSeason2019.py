import maya

import Projects.nfl.NFL_Prediction.NFLPredictor as Predictor
import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
import Projects.nfl.NFL_Prediction.PlottingHelper as Plotter
import Projects.nfl.NFL_Prediction.StandingsHelper as Standings
import Projects.nfl.NFL_Prediction.NFLGraph as Graph

nfl_teams = list()


def season():
    # Setup
    global nfl_teams
    nfl_teams = set_up_teams()
    Graph.create_games_graph()
    eliminated_teams = list()

    # Preseason Info
    print('Preseason Info')
    Standings.print_league_details(nfl_teams, eliminated_teams)
    Standings.print_schedule_difficulty(nfl_teams)
    print('*' * 120, '\n')

    # Plot the regular seasons initial ratings
    if maya.now() < maya.when('3 September 2019', timezone='US/Central'):
        Plotter.plot_elo_function(nfl_teams, '', 'Initial Ratings')

    # Regular Season
    print('Regular Season')
    nfl_teams = handle_week(nfl_teams, 'Week 1', week_1, eliminated_teams, '3 September 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 2', week_2, eliminated_teams, '10 September 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 3', week_3, eliminated_teams, '17 September 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 4', week_4, eliminated_teams, '24 September 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 5', week_5, eliminated_teams, '1 October 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 6', week_6, eliminated_teams, '8 October 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 7', week_7, eliminated_teams, '15 October 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 8', week_8, eliminated_teams, '22 October 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 9', week_9, eliminated_teams, '29 October 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 10', week_10, eliminated_teams, '5 November 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 11', week_11, eliminated_teams, '12 November 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 12', week_12, eliminated_teams, '19 November 2019')
    eliminated_teams.extend([])
    nfl_teams = handle_week(nfl_teams, 'Week 13', week_13, eliminated_teams, '26 November 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 14', week_14, eliminated_teams, '3 December 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 15', week_15, eliminated_teams, '10 December 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 16', week_16, eliminated_teams, '17 December 2019')
    nfl_teams = handle_week(nfl_teams, 'Week 17', week_17, eliminated_teams, '24 December 2019')
    eliminated_teams.extend([])

    # Regular Season remaining schedule difficulty
    Standings.print_schedule_difficulty(nfl_teams,
                                        remaining=True,
                                        completed=maya.now() >= maya.when('31 December 2019', timezone='US/Central'))

    # Get the pagerank of the teams
    Standings.print_team_pagerank(nfl_teams)

    # Show the parity clock
    Graph.parity_clock()

    # Show the graph
    Plotter.show_graph(Graph.nfl)

    # Playoff Info
    print('Current Playoff Picture:')
    afc_playoff_teams, nfc_playoff_teams = Playoffs.get_playoff_picture(nfl_teams, verbose=True)

    print('-' * 15 + 'AFC' + '-' * 15)
    Playoffs.create_playoff_bracket(afc_playoff_teams)
    print()

    print('-' * 15 + 'NFC' + '-' * 15)
    Playoffs.create_playoff_bracket(nfc_playoff_teams)
    print()

    print('*' * 120, '\n')

    # Playoffs
    print('Playoffs')
    if maya.now() < maya.when('31 December 2019', timezone='US/Central'):
        Playoffs.monte_carlo(nfl_teams)

    nfl_teams = handle_week(nfl_teams, 'Wildcard Weekend', wildcard, eliminated_teams, '31 December 2019')
    eliminated_teams.extend([])
    nfl_teams = handle_week(nfl_teams, 'Divisional Round', divisional, eliminated_teams, '7 January 2020')
    eliminated_teams.extend([])
    nfl_teams = handle_week(nfl_teams, 'Conference Finals', conference, eliminated_teams, '14 January 2020')
    eliminated_teams.extend([])
    nfl_teams = handle_week(nfl_teams, 'Superbowl', superbowl, eliminated_teams, '28 January 2020')

    # Save the standings csv
    Standings.get_full_standings_csv(nfl_teams)

    # Final Outcome
    league = Playoffs.get_league_structure()
    for conf_name, conf in league.items():
        for div_name, division in conf.items():
            Plotter.plot_team_elo_over_season(div_name, division)

    if maya.now() > maya.when('28 January 2020', timezone='US/Central'):
        Playoffs.completed_games.to_csv('..\\Projects\\nfl\\NFL_Prediction\\Game Data\\2019.csv', index=False)


def set_up_teams():
    def create_base_team(team_name, elo):
        return team_name, 0, 0, 0, elo, 0, 0, 0, 0, 0, 0, 0, 0, 0

    teams = list()
    teams.append(create_base_team('49ers', 1469.443))
    teams.append(create_base_team('Bears', 1557.476))
    teams.append(create_base_team('Bengals', 1459.57))
    teams.append(create_base_team('Bills', 1462.998))
    teams.append(create_base_team('Broncos', 1506.329))
    teams.append(create_base_team('Browns', 1514.419))
    teams.append(create_base_team('Buccaneers', 1474.242))
    teams.append(create_base_team('Cardinals', 1397.315))
    teams.append(create_base_team('Chargers', 1570.778))
    teams.append(create_base_team('Chiefs', 1577.908))
    teams.append(create_base_team('Colts', 1470.814))
    teams.append(create_base_team('Cowboys', 1528.68))
    teams.append(create_base_team('Dolphins', 1380.449))
    teams.append(create_base_team('Eagles', 1545.958))
    teams.append(create_base_team('Falcons', 1474.105))
    teams.append(create_base_team('Giants', 1460.393))
    teams.append(create_base_team('Jaguars', 1471.362))
    teams.append(create_base_team('Jets', 1421.038))
    teams.append(create_base_team('Lions', 1459.433))
    teams.append(create_base_team('Packers', 1532.109))
    teams.append(create_base_team('Panthers', 1509.894))
    teams.append(create_base_team('Patriots', 1593.129))
    teams.append(create_base_team('Raiders', 1410.342))
    teams.append(create_base_team('Rams', 1571.6))
    teams.append(create_base_team('Ravens', 1534.302))
    teams.append(create_base_team('Redskins', 1430.362))
    teams.append(create_base_team('Saints', 1587.095))
    teams.append(create_base_team('Seahawks', 1535.948))
    teams.append(create_base_team('Steelers', 1542.941))
    teams.append(create_base_team('Texans', 1528.406))
    teams.append(create_base_team('Titans', 1489.737))
    teams.append(create_base_team('Vikings', 1531.423))

    return teams


def handle_week(teams,
                week_name,
                week,
                eliminated_teams,
                week_start_date,
                full_standings=True,
                divisional_view=False,
                model_rankings=False):
    # Get the starting time of the week
    week_start = maya.when(week_start_date, timezone='US/Central')

    # If the week has started
    if maya.now() >= week_start:
        # Get the ending time of the week
        week_end_date = week_start.add(weeks=1)

        # Print the week name
        print(week_name)

        # Handle the week
        teams = week(teams, week_end_date)

        # Print the tables
        Standings.print_league_details(teams,
                                       eliminated_teams,
                                       full_standings=full_standings,
                                       divisional_view=divisional_view,
                                       model_rankings=model_rankings)

        # Print a separator
        print('*' * 120, '\n')

        # If the week has not ended
        if maya.now() < week_end_date:
            # Plot the team's percentiles based on their elo
            Plotter.plot_elo_function(teams, '', week_name)

            # Absolute Rankings
            Plotter.plot_elo_function(teams, '', week_name, absolute=True, show_plot=False)

            # Conference and division Rankings
            league = Playoffs.get_league_structure()
            for conf_name, conf in league.items():
                for div_name, division in conf.items():
                    Plotter.plot_division_elo_function(teams, div_name, week_name)
                Plotter.plot_conference_elo_function(teams, conf_name, week_name)

    # Return the updated teams
    return teams


def week_1(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week1_schedule(week_end_date))

    # Results
    teams = set_game_outcome(teams,
                             spread=-3,
                             home_name='Bears',
                             home_score=3,
                             home_touchdowns=0,
                             home_net_pass_yards=208,
                             home_pass_completions=26,
                             home_pass_attempts=45,
                             home_pass_tds=0,
                             home_interceptions_thrown=1,
                             home_total_yards=254,
                             away_name='Packers',
                             away_score=10,
                             away_touchdowns=1,
                             away_net_pass_yards=166,
                             away_pass_completions=18,
                             away_pass_attempts=30,
                             away_pass_tds=1,
                             away_interceptions_thrown=0,
                             away_total_yards=213)

    teams = set_game_outcome(teams, 1.5,
                             'Panthers', 27, 3, 216, 25, 38, 0, 1, 343,
                             'Rams', 30, 3, 183, 23, 39, 1, 1, 349)

    teams = set_game_outcome(teams, -10,
                             'Eagles', 32, 4, 313, 28, 39, 3, 0, 436,
                             'Redskins', 27, 3, 370, 30, 45, 3, 0, 398)

    teams = set_game_outcome(teams, -2.5,
                             'Jets', 16, 1, 155, 28, 41, 1, 0, 223,
                             'Bills', 17, 2, 242, 24, 37, 1, 2, 370)

    teams = set_game_outcome(teams, -3.5,
                             'Vikings', 28, 4, 97, 8, 10, 1, 0, 269,
                             'Falcons', 12, 2, 272, 33, 46, 2, 2, 345)

    teams = set_game_outcome(teams, 7,
                             'Dolphins', 10, 1, 179, 15, 32, 1, 2, 200,
                             'Ravens', 59, 8, 378, 23, 26, 6, 0, 643)

    teams = set_game_outcome(teams, 3.5,
                             'Jaguars', 26, 3, 347, 27, 33, 3, 1, 428,
                             'Chiefs', 40, 4, 378, 25, 34, 3, 0, 491)

    teams = set_game_outcome(teams, -5.5,
                             'Browns', 13, 2, 244, 25, 39, 1, 3, 346,
                             'Titans', 43, 4, 216, 14, 24, 3, 0, 339)

    teams = set_game_outcome(teams, -6,
                             'Chargers', 30, 4, 310, 25, 34, 3, 1, 435,
                             'Colts', 24, 3, 173, 21, 27, 2, 0, 376)

    teams = set_game_outcome(teams, -9,
                             'Seahawks', 21, 3, 161, 14, 20, 2, 0, 233,
                             'Bengals', 20, 2, 395, 35, 51, 2, 0, 429)

    teams = set_game_outcome(teams, -1.5,
                             'Buccaneers', 17, 1, 174, 20, 36, 1, 3, 295,
                             '49ers', 31, 1, 158, 18, 27, 1, 1, 256)

    teams = set_game_outcome(teams, -7,
                             'Cowboys', 35, 5, 405, 25, 32, 4, 0, 494,
                             'Giants', 17, 2, 319, 33, 48, 1, 0, 470)

    teams = set_game_outcome(teams, 3,
                             'Cardinals', 27, 2, 275, 29, 54, 2, 1, 387,
                             'Lions', 27, 3, 361, 27, 45, 3, 0, 477)

    teams = set_game_outcome(teams, -5.5,
                             'Patriots', 33, 3, 366, 25, 37, 3, 0, 465,
                             'Steelers', 3, 0, 276, 27, 47, 0, 1, 308)

    teams = set_game_outcome(teams, -7,
                             'Saints', 30, 3, 362, 32, 43, 2, 1, 510,
                             'Texans', 28, 4, 234, 20, 30, 3, 1, 414)

    teams = set_game_outcome(teams, 2.5,
                             'Raiders', 24, 3, 259, 22, 26, 1, 0, 357,
                             'Broncos', 16, 1, 249, 21, 31, 1, 0, 344)

    return teams


def week_2(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week2_schedule(week_end_date))

    # Results
    teams = set_game_outcome(teams,
                             spread=-6.5,
                             home_name='Panthers',
                             home_score=14,
                             home_touchdowns=0,
                             home_net_pass_yards=304,
                             home_pass_completions=24,
                             home_pass_attempts=50,
                             home_pass_tds=0,
                             home_interceptions_thrown=0,
                             home_total_yards=343,
                             away_name='Buccaneers',
                             away_score=20,
                             away_touchdowns=2,
                             away_net_pass_yards=189,
                             away_pass_completions=16,
                             away_pass_attempts=25,
                             away_pass_tds=1,
                             away_interceptions_thrown=0,
                             away_total_yards=289)

    teams = set_game_outcome(teams, -13,
                             'Ravens', 23, 2, 258, 24, 37, 2, 0, 440,
                             'Cardinals', 17, 1, 329, 25, 40, 0, 0, 349)

    teams = set_game_outcome(teams, 6.5,
                             'Redskins', 21, 3, 208, 26, 37, 2, 0, 255,
                             'Cowboys', 31, 4, 261, 26, 30, 3, 1, 474)

    teams = set_game_outcome(teams, -3,
                             'Titans', 17, 2, 119, 19, 28, 1, 0, 242,
                             'Colts', 19, 3, 121, 17, 28, 3, 1, 288)

    teams = set_game_outcome(teams, -4,
                             'Steelers', 26, 3, 180, 20, 34, 2, 1, 261,
                             'Seahawks', 28, 4, 274, 29, 35, 3, 0, 426)

    teams = set_game_outcome(teams, 1,
                             'Giants', 14, 2, 241, 26, 45, 1, 2, 370,
                             'Bills', 28, 4, 237, 19, 30, 1, 0, 388)

    teams = set_game_outcome(teams, -1,
                             'Bengals', 17, 2, 291, 26, 42, 2, 1, 316,
                             '49ers', 41, 5, 313, 18, 26, 3, 1, 572)

    teams = set_game_outcome(teams, 0,
                             'Lions', 13, 2, 245, 22, 30, 2, 2, 339,
                             'Chargers', 10, 1, 287, 21, 36, 0, 1, 424)

    teams = set_game_outcome(teams, -3,
                             'Packers', 21, 3, 191, 22, 34, 2, 0, 335,
                             'Vikings', 16, 2, 223, 14, 32, 1, 2, 421)

    teams = set_game_outcome(teams, -7,
                             'Texans', 13, 1, 137, 16, 29, 0, 0, 263,
                             'Jaguars', 12, 1, 178, 23, 33, 1, 0, 281)

    teams = set_game_outcome(teams, 18,
                             'Dolphins', 0, 0, 142, 18, 39, 0, 4, 184,
                             'Patriots', 43, 4, 255, 20, 28, 2, 0, 381)

    teams = set_game_outcome(teams, 7,
                             'Raiders', 10, 1, 178, 23, 38, 1, 2, 307,
                             'Chiefs', 28, 4, 433, 30, 44, 4, 0, 464)

    teams = set_game_outcome(teams, -1.5,
                             'Rams', 27, 3, 265, 19, 28, 1, 0, 380,
                             'Saints', 9, 0, 187, 20, 35, 0, 1, 244)

    teams = set_game_outcome(teams, 3,
                             'Broncos', 14, 1, 282, 35, 50, 1, 1, 372,
                             'Bears', 16, 1, 120, 16, 27, 0, 0, 273)

    teams = set_game_outcome(teams, 1,
                             'Falcons', 24, 3, 310, 27, 43, 3, 3, 367,
                             'Eagles', 20, 2, 237, 28, 48, 1, 2, 286)

    teams = set_game_outcome(teams, 6.5,
                             'Jets', 3, 0, 169, 23, 31, 0, 0, 262,
                             'Browns', 23, 2, 305, 19, 35, 1, 1, 375)

    return teams


def week_3(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week3_schedule(week_end_date))

    # Results
    teams = set_game_outcome(teams,
                             spread=2,
                             home_name='Jaguars',
                             home_score=20,
                             home_touchdowns=2,
                             home_net_pass_yards=204,
                             home_pass_completions=20,
                             home_pass_attempts=30,
                             home_pass_tds=2,
                             home_interceptions_thrown=0,
                             home_total_yards=292,
                             away_name='Titans',
                             away_score=7,
                             away_touchdowns=1,
                             away_net_pass_yards=249,
                             away_pass_completions=23,
                             away_pass_attempts=40,
                             away_pass_tds=0,
                             away_interceptions_thrown=0,
                             away_total_yards=340)

    teams = set_game_outcome(teams, -6,
                             'Bills', 21, 2, 241, 23, 36, 1, 1, 416,
                             'Bengals', 17, 2, 239, 20, 36, 1, 2, 306)

    teams = set_game_outcome(teams, -4,
                             'Eagles', 24, 3, 246, 19, 36, 2, 0, 373,
                             'Lions', 27, 2, 201, 18, 32, 1, 0, 288)

    teams = set_game_outcome(teams, -21,
                             'Patriots', 30, 4, 313, 30, 45, 2, 1, 381,
                             'Jets', 14, 0, 69, 12, 22, 0, 1, 105)

    teams = set_game_outcome(teams, -9,
                             'Vikings', 34, 4, 174, 15, 21, 1, 0, 385,
                             'Raiders', 14, 2, 214, 27, 34, 2, 1, 302)

    teams = set_game_outcome(teams, -4.5,
                             'Chiefs', 33, 4, 363, 27, 37, 3, 0, 503,
                             'Ravens', 28, 4, 249, 22, 43, 0, 0, 452)

    teams = set_game_outcome(teams, -1,
                             'Colts', 27, 3, 300, 28, 37, 2, 0, 379,
                             'Falcons', 24, 3, 304, 29, 34, 3, 1, 397)

    teams = set_game_outcome(teams, -7,
                             'Packers', 27, 3, 235, 17, 29, 1, 0, 312,
                             'Broncos', 16, 2, 161, 20, 29, 0, 1, 310)

    teams = set_game_outcome(teams, -22.5,
                             'Cowboys', 31, 4, 241, 19, 32, 2, 1, 476,
                             'Dolphins', 6, 0, 211, 20, 41, 0, 0, 283)

    teams = set_game_outcome(teams, -4.5,
                             'Buccaneers', 31, 3, 355, 23, 37, 3, 1, 499,
                             'Giants', 32, 4, 312, 23, 36, 2, 0, 384)

    teams = set_game_outcome(teams, -2,
                             'Cardinals', 20, 2, 127, 30, 43, 2, 2, 248,
                             'Panthers', 38, 5, 240, 19, 26, 4, 0, 413)

    teams = set_game_outcome(teams, -6.5,
                             '49ers', 24, 3, 268, 23, 32, 1, 2, 436,
                             'Steelers', 20, 2, 160, 14, 27, 2, 1, 241)

    teams = set_game_outcome(teams, -5.5,
                             'Seahawks', 27, 4, 406, 32, 50, 2, 0, 515,
                             'Saints', 33, 3, 177, 19, 27, 2, 0, 365)

    teams = set_game_outcome(teams, -3,
                             'Chargers', 20, 2, 293, 31, 46, 2, 0, 366,
                             'Texans', 27, 4, 337, 25, 34, 3, 0, 376)

    teams = set_game_outcome(teams, 4,
                             'Browns', 13, 1, 175, 18, 36, 1, 1, 270,
                             'Rams', 20, 2, 255, 24, 38, 2, 2, 345)

    teams = set_game_outcome(teams, 5,
                             'Redskins', 15, 2, 287, 30, 43, 2, 3, 356,
                             'Bears', 31, 3, 208, 25, 31, 3, 1, 298)

    return teams


def week_4(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week4_schedule(week_end_date))

    # Results
    teams = set_game_outcome(teams,
                             spread=-3.5,
                             home_name='Packers',
                             home_score=27,
                             home_touchdowns=3,
                             home_net_pass_yards=414,
                             home_pass_completions=34,
                             home_pass_attempts=53,
                             home_pass_tds=2,
                             home_interceptions_thrown=1,
                             home_total_yards=491,
                             away_name='Eagles',
                             away_score=34,
                             away_touchdowns=5,
                             away_net_pass_yards=160,
                             away_pass_completions=16,
                             away_pass_attempts=27,
                             away_pass_tds=3,
                             away_interceptions_thrown=0,
                             away_total_yards=336)

    teams = set_game_outcome(teams, -3.5,
                             'Falcons', 10, 1, 364, 35, 53, 0, 0, 422,
                             'Titans', 24, 3, 227, 18, 27, 3, 0, 365)

    teams = set_game_outcome(teams, -3,
                             'Giants', 24, 2, 225, 23, 31, 1, 2, 389,
                             'Redskins', 3, 0, 121, 15, 28, 0, 4, 176)

    teams = set_game_outcome(teams, 14.5,
                             'Dolphins', 10, 1, 161, 17, 24, 1, 1, 233,
                             'Chargers', 30, 3, 311, 25, 31, 2, 0, 390)

    teams = set_game_outcome(teams, -5,
                             'Colts', 24, 3, 265, 24, 46, 3, 1, 346,
                             'Raiders', 31, 3, 189, 21, 31, 2, 0, 377)

    teams = set_game_outcome(teams, -5.5,
                             'Texans', 10, 1, 128, 21, 34, 0, 1, 264,
                             'Panthers', 16, 1, 203, 24, 34, 0, 0, 297)

    teams = set_game_outcome(teams, 7,
                             'Lions', 30, 3, 261, 21, 34, 3, 0, 447,
                             'Chiefs', 34, 3, 315, 24, 42, 0, 0, 438)

    teams = set_game_outcome(teams, -7,
                             'Ravens', 25, 3, 222, 24, 34, 3, 2, 395,
                             'Browns', 40, 5, 337, 20, 31, 1, 1, 530)

    teams = set_game_outcome(teams, 7,
                             'Bills', 10, 1, 240, 22, 44, 0, 4, 375,
                             'Patriots', 16, 1, 150, 18, 39, 0, 1, 224)

    teams = set_game_outcome(teams, -9.5,
                             'Rams', 40, 4, 490, 45, 68, 2, 3, 518,
                             'Buccaneers', 55, 6, 376, 28, 41, 4, 1, 464)

    teams = set_game_outcome(teams, 5.5,
                             'Cardinals', 10, 1, 206, 22, 32, 0, 1, 321,
                             'Seahawks', 27, 2, 225, 22, 28, 1, 0, 340)

    teams = set_game_outcome(teams, -1,
                             'Bears', 16, 1, 194, 24, 33, 1, 0, 269,
                             'Vikings', 6, 1, 182, 27, 36, 0, 0, 222)

    teams = set_game_outcome(teams, -2.5,
                             'Broncos', 24, 3, 303, 22, 38, 3, 1, 371,
                             'Jaguars', 26, 2, 186, 19, 33, 2, 0, 455)

    teams = set_game_outcome(teams, 2.5,
                             'Saints', 12, 0, 149, 23, 30, 0, 1, 266,
                             'Cowboys', 10, 1, 212, 22, 33, 0, 1, 257)

    teams = set_game_outcome(teams, -3.5,
                             'Steelers', 27, 3, 260, 27, 31, 2, 0, 326,
                             'Bengals', 3, 0, 102, 21, 37, 0, 2, 175)

    return teams


def week_5(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week5_schedule(week_end_date))

    # Results
    teams = set_game_outcome(teams,
                             spread=-1.5,
                             home_name='Seahawks',
                             home_score=30,
                             home_touchdowns=4,
                             home_net_pass_yards=262,
                             home_pass_completions=17,
                             home_pass_attempts=23,
                             home_pass_tds=4,
                             home_interceptions_thrown=0,
                             home_total_yards=429,
                             away_name='Rams',
                             away_score=29,
                             away_touchdowns=3,
                             away_net_pass_yards=395,
                             away_pass_completions=29,
                             away_pass_attempts=49,
                             away_pass_tds=1,
                             away_interceptions_thrown=1,
                             away_total_yards=477)

    teams = set_game_outcome(teams, -3.5,
                             'Panthers', 34, 4, 160, 17, 31, 1, 0, 445,
                             'Jaguars', 27, 3, 358, 26, 45, 2, 1, 507)

    teams = set_game_outcome(teams, 16.5,
                             'Redskins', 7, 1, 75, 18, 27, 0, 1, 220,
                             'Patriots', 33, 4, 312, 28, 42, 3, 1, 432)

    teams = set_game_outcome(teams, -3,
                             'Titans', 7, 1, 150, 13, 22, 0, 0, 252,
                             'Bills', 14, 2, 204, 23, 32, 2, 1, 313)

    teams = set_game_outcome(teams, 3,
                             'Steelers', 23, 2, 192, 21, 31, 1, 1, 269,
                             'Ravens', 26, 2, 139, 19, 28, 1, 3, 277)

    teams = set_game_outcome(teams, -3,
                             'Bengals', 23, 2, 262, 27, 38, 2, 0, 370,
                             'Cardinals', 26, 2, 248, 20, 32, 0, 0, 514)

    teams = set_game_outcome(teams, -4,
                             'Texans', 53, 6, 426, 28, 33, 5, 0, 592,
                             'Falcons', 32, 4, 316, 32, 46, 3, 1, 373)

    teams = set_game_outcome(teams, -3,
                             'Saints', 31, 4, 345, 28, 36, 4, 1, 457,
                             'Buccaneers', 24, 3, 158, 15, 27, 2, 0, 252)

    teams = set_game_outcome(teams, 5.5,
                             'Giants', 10, 1, 147, 21, 38, 1, 1, 211,
                             'Vikings', 28, 2, 279, 22, 27, 2, 0, 490)

    teams = set_game_outcome(teams, 6.5,
                             'Raiders', 24, 3, 229, 25, 32, 0, 0, 398,
                             'Bears', 21, 3, 194, 22, 30, 2, 2, 236)

    teams = set_game_outcome(teams, -14,
                             'Eagles', 31, 2, 181, 17, 29, 1, 0, 265,
                             'Jets', 6, 1, 61, 15, 26, 0, 2, 128)

    teams = set_game_outcome(teams, -4.5,
                             'Chargers', 13, 0, 211, 32, 48, 0, 2, 246,
                             'Broncos', 20, 2, 159, 14, 20, 1, 1, 350)

    teams = set_game_outcome(teams, -3.5,
                             'Cowboys', 24, 3, 441, 27, 44, 2, 3, 563,
                             'Packers', 34, 4, 215, 22, 34, 0, 0, 335)

    teams = set_game_outcome(teams, -11,
                             'Chiefs', 13, 1, 288, 22, 39, 1, 0, 324,
                             'Colts', 19, 1, 151, 18, 29, 0, 1, 331)

    teams = set_game_outcome(teams, -5,
                             '49ers', 31, 4, 171, 20, 29, 2, 0, 446,
                             'Browns', 3, 0, 78, 9, 24, 0, 2, 180)

    return teams


def week_6(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week6_schedule(week_end_date))

    # Results
    teams = set_game_outcome(teams,
                             spread=-17,
                             home_name='Patriots',
                             home_score=35,
                             home_touchdowns=4,
                             home_net_pass_yards=313,
                             home_pass_completions=31,
                             home_pass_attempts=41,
                             home_pass_tds=0,
                             home_interceptions_thrown=1,
                             home_total_yards=427,
                             away_name='Giants',
                             away_score=14,
                             away_touchdowns=1,
                             away_net_pass_yards=161,
                             away_pass_completions=15,
                             away_pass_attempts=31,
                             away_pass_tds=1,
                             away_interceptions_thrown=3,
                             away_total_yards=213)

    teams = set_game_outcome(teams, 2,
                             'Buccaneers', 26, 3, 365, 30, 54, 1, 5, 407,
                             'Panthers', 37, 4, 209, 20, 32, 2, 0, 268)

    teams = set_game_outcome(teams, 6,
                             'Dolphins', 16, 2, 187, 27, 43, 1, 2, 271,
                             'Redskins', 17, 2, 166, 13, 25, 2, 0, 311)

    teams = set_game_outcome(teams, -3.5,
                             'Vikings', 38, 5, 325, 22, 29, 4, 1, 447,
                             'Eagles', 20, 2, 292, 26, 41, 2, 2, 400)

    teams = set_game_outcome(teams, -3.5,
                             'Chiefs', 24, 3, 256, 19, 35, 3, 1, 309,
                             'Texans', 31, 4, 280, 30, 42, 1, 2, 472)

    teams = set_game_outcome(teams, -2.5,
                             'Jaguars', 6, 0, 151, 14, 29, 0, 1, 226,
                             'Saints', 13, 1, 222, 24, 36, 1, 0, 326)

    teams = set_game_outcome(teams, -1,
                             'Browns', 28, 4, 249, 22, 37, 1, 3, 406,
                             'Seahawks', 32, 2, 284, 23, 33, 2, 0, 454)

    teams = set_game_outcome(teams, -10.5,
                             'Ravens', 23, 2, 228, 21, 33, 0, 0, 497,
                             'Bengals', 17, 1, 217, 21, 39, 0, 1, 250)

    teams = set_game_outcome(teams, -3,
                             'Rams', 7, 1, 56, 13, 24, 0, 0, 165,
                             '49ers', 20, 2, 232, 24, 33, 0, 1, 321)

    teams = set_game_outcome(teams, 3,
                             'Cardinals', 34, 4, 340, 27, 37, 3, 0, 442,
                             'Falcons', 33, 4, 341, 30, 36, 4, 0, 444)

    teams = set_game_outcome(teams, 7,
                             'Jets', 24, 3, 326, 23, 32, 2, 1, 372,
                             'Cowboys', 22, 2, 269, 28, 40, 0, 0, 398)

    teams = set_game_outcome(teams, -1,
                             'Broncos', 16, 1, 167, 18, 28, 0, 1, 270,
                             'Titans', 0, 0, 165, 20, 34, 0, 3, 204)

    teams = set_game_outcome(teams, -6,
                             'Chargers', 17, 2, 316, 26, 44, 2, 2, 348,
                             'Steelers', 24, 2, 132, 15, 20, 1, 1, 256)

    teams = set_game_outcome(teams, -3.5,
                             'Packers', 23, 2, 277, 24, 39, 2, 1, 437,
                             'Lions', 22, 1, 243, 18, 32, 0, 0, 299)
    return teams


def week_7(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week7_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Broncos',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Chiefs',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_8(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week8_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Vikings',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Redskins',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_9(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week9_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Cardinals',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='49ers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_10(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week10_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Raiders',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Chargers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_11(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week11_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Browns',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Steelers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_12(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week12_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Texans',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Colts',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bucanneerrs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_13(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week13_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Lions',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Bears',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_14(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week14_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Cowboys',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_15(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week15_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Ravens',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Jets',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_16(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week16_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Broncos',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Lions',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_17(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week17_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Ravens',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Steelers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def wildcard(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_wildcard_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Packers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def divisional(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_divisional_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Packers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def conference(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_conference_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Packers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def superbowl(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_superbowl_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Packers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)

    return teams


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


def set_game_outcome(teams, spread,
                     home_name, home_score, home_touchdowns, home_net_pass_yards, home_pass_completions,
                     home_pass_attempts, home_pass_tds, home_interceptions_thrown, home_total_yards,
                     away_name, away_score, away_touchdowns, away_net_pass_yards, away_pass_completions,
                     away_pass_attempts, away_pass_tds, away_interceptions_thrown, away_total_yards):
    import pandas as pd

    home_team = get_team(teams, home_name)
    away_team = get_team(teams, away_name)

    game_df = pd.DataFrame({'home_team': home_name,
                            'away_team': away_name,
                            'home_score': home_score,
                            'away_score': away_score,
                            'home_spread': spread,
                            'home_pass_completions': home_pass_completions,
                            'home_pass_attempts': home_pass_attempts,
                            'home_passing_touchdowns': home_pass_tds,
                            'home_interceptions_thrown': home_interceptions_thrown,
                            'home_net_passing_yards': home_net_pass_yards,
                            'home_total_yards': home_total_yards,
                            'home_elo': home_team[4],
                            'away_pass_completions': away_pass_completions,
                            'away_pass_attempts': away_pass_attempts,
                            'away_passing_touchdowns': away_pass_tds,
                            'away_interceptions_thrown': away_interceptions_thrown,
                            'away_net_passing_yards': away_net_pass_yards,
                            'away_total_yards': away_total_yards,
                            'away_elo': away_team[4]},
                           index=[len(Playoffs.completed_games)])
    Playoffs.completed_games = Playoffs.completed_games.append(game_df)

    teams = Predictor.update_teams(teams,
                                   home_name, home_score, home_touchdowns, home_net_pass_yards, home_pass_completions,
                                   home_pass_attempts, home_pass_tds, home_interceptions_thrown, home_total_yards,
                                   away_name, away_score, away_touchdowns, away_net_pass_yards, away_pass_completions,
                                   away_pass_attempts, away_pass_tds, away_interceptions_thrown, away_total_yards)

    Graph.set_game_outcome(teams, home_name, home_score, away_name, away_score)

    return teams
