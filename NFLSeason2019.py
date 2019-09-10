import maya

import Projects.nfl.NFL_Prediction.NFLPredictor as Predictor
import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
import Projects.nfl.NFL_Prediction.PlottingHelper as Plotter
import Projects.nfl.NFL_Prediction.StandingsHelper as Standings

nfl_teams = list()


def season():
    # Setup
    global nfl_teams
    nfl_teams = set_up_teams()
    eliminated_teams = list()

    # Preseason Info
    print('Preseason Info')
    Standings.print_league_details(nfl_teams, eliminated_teams)
    Standings.print_schedule_difficulty(nfl_teams)
    print('*' * 120, '\n')

    # Plot the regular seasons initial ratings
    Plotter.plot_elo_function(nfl_teams, '', 'Initial Ratings', show_plot=False)

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
    Playoffs.monte_carlo(nfl_teams)
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
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Panthers',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Buccaneers',
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
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_3(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week3_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Jaguars',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Titans',
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
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_4(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week4_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Packers',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Eagles',
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
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_5(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week5_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Seahawks',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Rams',
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
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_6(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week6_schedule(week_end_date))

    # Results
    # teams = set_game_outcome(teams,
    #                          spread=0,
    #                          home_name='Patriots',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Giants',
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
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
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

    return teams
