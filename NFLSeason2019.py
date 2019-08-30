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

    # Preseason
    if maya.now() < maya.when('30 August 2019', timezone='US/Central'):
        nfl_teams = handle_week(nfl_teams, 'Preseason Week 1', pre_week_1, eliminated_teams, '6 August 2019')
        nfl_teams = handle_week(nfl_teams, 'Preseason Week 2', pre_week_2, eliminated_teams, '13 August 2019')
        nfl_teams = handle_week(nfl_teams, 'Preseason Week 3', pre_week_3, eliminated_teams, '20 August 2019')
        nfl_teams = handle_week(nfl_teams, 'Preseason Week 4', pre_week_4, eliminated_teams, '27 August 2019')

    # Reset teams after preseason
    nfl_teams = set_up_teams()

    # Preseason Info
    print('Preseason Info')
    Standings.print_league_details(nfl_teams, eliminated_teams)
    Standings.print_schedule_difficulty(nfl_teams)
    print('*' * 120, '\n')

    # Plot the regular seasons initial ratings
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
    Standings.print_schedule_difficulty(nfl_teams, remaining=True)

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

    # Final Outcome
    Playoffs.monte_carlo(nfl_teams)
    league = Playoffs.get_league_structure()
    for conf_name, conf in league.items():
        for div_name, division in conf.items():
            Plotter.plot_team_elo_over_season(div_name, division)


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


def pre_week_1(teams, week_end_date):
    # If the week has not ended
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_pre_week1_schedule())

    # Results
    teams = set_game_outcome(teams,
                             spread=2.5,
                             home_name='Falcons',
                             home_score=10,
                             home_touchdowns=1,
                             home_net_pass_yards=178,
                             home_pass_completions=23,
                             home_pass_attempts=48,
                             home_pass_tds=1,
                             home_interceptions_thrown=1,
                             home_total_yards=261,
                             away_name='Broncos',
                             away_score=14,
                             away_touchdowns=2,
                             away_net_pass_yards=93,
                             away_pass_completions=17,
                             away_pass_attempts=29,
                             away_pass_tds=1,
                             away_interceptions_thrown=0,
                             away_total_yards=188)

    teams = set_game_outcome(teams, 2.5,
                             'Giants', 31, 3, 374, 29, 37, 3, 0, 414,
                             'Jets', 22, 3, 217, 28, 39, 3, 2, 286)

    teams = set_game_outcome(teams, -1.5,
                             'Eagles', 10, 1, 190, 15, 33, 1, 1, 227,
                             'Titans', 27, 4, 263, 31, 44, 4, 0, 388)

    teams = set_game_outcome(teams, -3,
                             'Bears', 13, 1, 170, 19, 30, 0, 0, 252,
                             'Panthers', 23, 2, 173, 20, 36, 1, 1, 269)

    teams = set_game_outcome(teams, -2.5,
                             'Lions', 3, 0, 21, 7, 17, 0, 1, 93,
                             'Patriots', 31, 4, 326, 26, 38, 3, 0, 459)

    teams = set_game_outcome(teams, -1.5,
                             'Packers', 28, 3, 142, 11, 21, 3, 0, 237,
                             'Texans', 26, 3, 274, 25, 40, 1, 2, 412)

    teams = set_game_outcome(teams, -3,
                             'Cardinals', 17, 2, 206, 23, 32, 1, 1, 289,
                             'Chargers', 13, 2, 178, 17, 23, 0, 1, 357)

    teams = set_game_outcome(teams, 2,
                             'Seahawks', 22, 2, 150, 14, 24, 1, 0, 301,
                             'Broncos', 14, 1, 207, 23, 39, 1, 1, 298)

    teams = set_game_outcome(teams, 1.5,
                             'Bills', 24, 3, 218, 17, 35, 1, 0, 381,
                             'Colts', 16, 1, 209, 23, 42, 0, 1, 314)

    teams = set_game_outcome(teams, -4,
                             'Dolphins', 34, 4, 265, 21, 33, 1, 1, 361,
                             'Falcons', 27, 3, 233, 20, 36, 0, 0, 337)

    teams = set_game_outcome(teams, -3.5,
                             'Ravens', 29, 1, 163, 16, 32, 1, 1, 288,
                             'Jaguars', 0, 0, 44, 10, 25, 0, 2, 112)

    teams = set_game_outcome(teams, -1.5,
                             'Browns', 30, 2, 327, 26, 43, 2, 0, 417,
                             'Redskins', 10, 1, 189, 18, 34, 1, 3, 271)

    teams = set_game_outcome(teams, -2.5,
                             'Saints', 25, 2, 196, 22, 33, 2, 1, 337,
                             'Vikings', 34, 4, 247, 19, 27, 3, 0, 460)

    teams = set_game_outcome(teams, -2.5,
                             'Steelers', 30, 3, 231, 18, 30, 3, 0, 339,
                             'Buccaneers', 28, 4, 390, 37, 57, 2, 0, 479)

    teams = set_game_outcome(teams, -3.5,
                             'Chiefs', 38, 5, 285, 23, 38, 3, 1, 400,
                             'Bengals', 17, 2, 253, 28, 46, 1, 1, 274)

    teams = set_game_outcome(teams, -5,
                             'Raiders', 14, 2, 258, 26, 37, 1, 2, 407,
                             'Rams', 3, 0, 133, 15, 28, 0, 0, 190)

    teams = set_game_outcome(teams, -4.5,
                             '49ers', 17, 2, 251, 26, 37, 2, 2, 339,
                             'Cowboys', 9, 0, 239, 29, 50, 0, 0, 294)

    return teams


def pre_week_2(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_pre_week2_schedule())

    # Results
    teams = set_game_outcome(teams,
                             spread=-2.5,
                             home_name='Giants',
                             home_score=32,
                             home_touchdowns=4,
                             home_net_pass_yards=250,
                             home_pass_completions=21,
                             home_pass_attempts=30,
                             home_pass_tds=3,
                             home_interceptions_thrown=1,
                             home_total_yards=411,
                             away_name='Bears',
                             away_score=13,
                             away_touchdowns=1,
                             away_net_pass_yards=131,
                             away_pass_completions=18,
                             away_pass_attempts=31,
                             away_pass_tds=0,
                             away_interceptions_thrown=0,
                             away_total_yards=165)

    teams = set_game_outcome(teams, -4,
                             'Redskins', 13, 1, 153, 12, 23, 1, 0, 212,
                             'Bengals', 23, 2, 243, 30, 43, 2, 1, 335)

    teams = set_game_outcome(teams, -3.5,
                             'Vikings', 25, 3, 272, 28, 35, 2, 1, 409,
                             'Seahawks', 19, 0, 145, 12, 27, 0, 1, 221)

    teams = set_game_outcome(teams, -1.5,
                             'Falcons', 10, 1, 253, 27, 43, 0, 1, 340,
                             'Jets', 22, 2, 103, 13, 20, 1, 0, 199)

    teams = set_game_outcome(teams, -1.5,
                             'Panthers', 14, 1, 155, 22, 42, 1, 1, 258,
                             'Bills', 27, 2, 279, 21, 31, 1, 1, 373)

    teams = set_game_outcome(teams, -2.5,
                             'Buccaneers', 16, 1, 234, 21, 37, 1, 0, 309,
                             'Dolphins', 14, 1, 162, 19, 38, 1, 1, 280)

    teams = set_game_outcome(teams, -3,
                             'Cardinals', 26, 3, 259, 21, 38, 3, 0, 374,
                             'Raiders', 33, 4, 250, 21, 24, 3, 0, 373)

    teams = set_game_outcome(teams, 2.5,
                             'Rams', 10, 1, 203, 26, 41, 1, 1, 270,
                             'Cowboys', 14, 2, 157, 22, 34, 1, 1, 251)

    teams = set_game_outcome(teams, -3.5,
                             'Ravens', 26, 2, 172, 16, 26, 1, 1, 343,
                             'Packers', 13, 1, 171, 18, 33, 1, 0, 226)

    teams = set_game_outcome(teams, 2.5,
                             'Steelers', 17, 2, 205, 18, 30, 1, 1, 329,
                             'Chiefs', 7, 1, 206, 22, 43, 1, 0, 315)

    teams = set_game_outcome(teams, -4.5,
                             'Texans', 30, 2, 252, 20, 33, 2, 1, 410,
                             'Lions', 23, 2, 298, 21, 37, 1, 1, 388)

    teams = set_game_outcome(teams, -3,
                             'Colts', 18, 2, 309, 30, 44, 2, 0, 382,
                             'Browns', 21, 3, 207, 20, 31, 3, 0, 271)

    teams = set_game_outcome(teams, -3,
                             'Jaguars', 10, 1, 191, 21, 38, 0, 0, 250,
                             'Eagles', 24, 3, 185, 17, 31, 1, 1, 324)

    teams = set_game_outcome(teams, 1,
                             'Titans', 17, 2, 206, 17, 33, 2, 0, 306,
                             'Patriots', 22, 3, 240, 20, 27, 1, 1, 363)

    teams = set_game_outcome(teams, -2.5,
                             'Broncos', 15, 1, 102, 19, 38, 0, 1, 215,
                             '49ers', 24, 3, 93, 8, 20, 1, 1, 278)

    teams = set_game_outcome(teams, -2.5,
                             'Chargers', 17, 1, 226, 22, 37, 1, 2, 304,
                             'Saints', 19, 2, 161, 16, 27, 2, 1, 324)

    return teams


def pre_week_3(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_pre_week3_schedule())

    # Results
    teams = set_game_outcome(teams,
                             spread=0,
                             home_name='Cowboys',
                             home_score=34,
                             home_touchdowns=3,
                             home_net_pass_yards=249,
                             home_pass_completions=24,
                             home_pass_attempts=42,
                             home_pass_tds=2,
                             home_interceptions_thrown=0,
                             home_total_yards=362,
                             away_name='Texans',
                             away_score=0,
                             away_touchdowns=0,
                             away_net_pass_yards=50,
                             away_pass_completions=10,
                             away_pass_attempts=26,
                             away_pass_tds=0,
                             away_interceptions_thrown=2,
                             away_total_yards=135)

    teams = set_game_outcome(teams, 0,
                             'Eagles', 15, 2, 215, 20, 30, 2, 0, 262,
                             'Ravens', 26, 3, 203, 19, 29, 2, 0, 243)

    teams = set_game_outcome(teams, 0,
                             'Lions', 20, 2, 212, 20, 37, 1, 0, 317,
                             'Bills', 24, 3, 203, 17, 24, 1, 0, 350)

    teams = set_game_outcome(teams, 0,
                             'Vikings', 20, 3, 178, 15, 29, 1, 0, 368,
                             'Cardinals', 9, 0, 292, 29, 46, 0, 0, 362)

    teams = set_game_outcome(teams, 0,
                             'Falcons', 7, 1, 80, 11, 20, 0, 0, 235,
                             'Redskins', 19, 1, 172, 16, 27, 0, 0, 280)

    teams = set_game_outcome(teams, 0,
                             'Buccaneers', 13, 1, 195, 24, 41, 1, 0, 265,
                             'Browns', 12, 0, 108, 16, 39, 0, 1, 141)

    teams = set_game_outcome(teams, 0,
                             'Rams', 10, 1, 226, 18, 31, 1, 0, 323,
                             'Broncos', 6, 0, 140, 22, 35, 0, 1, 213)

    teams = set_game_outcome(teams, 0,
                             'Dolphins', 22, 2, 182, 17, 25, 1, 0, 271,
                             'Jaguars', 7, 1, 176, 23, 37, 1, 1, 243)

    teams = set_game_outcome(teams, 0,
                             'Patriots', 10, 1, 187, 23, 31, 0, 0, 316,
                             'Panthers', 3, 0, 48, 11, 20, 0, 0, 99)

    teams = set_game_outcome(teams, 0,
                             'Jets', 13, 1, 205, 23, 33, 1, 0, 300,
                             'Saints', 26, 1, 325, 27, 45, 1, 0, 405)

    teams = set_game_outcome(teams, 0,
                             'Bengals', 23, 3, 325, 31, 42, 3, 0, 354,
                             'Giants', 25, 2, 267, 19, 32, 2, 0, 323)

    teams = set_game_outcome(teams, 0,
                             'Colts', 17, 2, 242, 22, 34, 1, 2, 348,
                             'Bears', 27, 1, 144, 14, 25, 1, 1, 245)

    teams = set_game_outcome(teams, 0,
                             'Titans', 6, 0, 132, 17, 27, 0, 0, 233,
                             'Steelers', 18, 2, 217, 18, 31, 2, 2, 328)

    teams = set_game_outcome(teams, 0,
                             'Chiefs', 17, 2, 185, 18, 28, 1, 0, 248,
                             '49ers', 27, 3, 257, 22, 31, 1, 0, 381)

    teams = set_game_outcome(teams, 0,
                             'Raiders', 22, 2, 248, 27, 46, 2, 0, 323,
                             'Packers', 21, 3, 193, 20, 32, 2, 1, 287)

    teams = set_game_outcome(teams, 0,
                             'Chargers', 15, 2, 181, 22, 30, 1, 0, 280,
                             'Seahawks', 23, 3, 182, 17, 27, 0, 0, 367)

    return teams


def pre_week_4(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_pre_week4_schedule())

    # Results
    teams = set_game_outcome(teams,
                             spread=-5,
                             home_name='Cowboys',
                             home_score=15,
                             home_touchdowns=1,
                             home_net_pass_yards=239,
                             home_pass_completions=25,
                             home_pass_attempts=42,
                             home_pass_tds=1,
                             home_interceptions_thrown=0,
                             home_total_yards=279,
                             away_name='Buccaneers',
                             away_score=17,
                             away_touchdowns=2,
                             away_net_pass_yards=105,
                             away_pass_completions=13,
                             away_pass_attempts=19,
                             away_pass_tds=1,
                             away_interceptions_thrown=3,
                             away_total_yards=217)

    teams = set_game_outcome(teams, 6,
                             'Redskins', 7, 1, 107, 12, 25, 1, 1, 180,
                             'Ravens', 20, 2, 203, 20, 37, 1, 0, 358)

    teams = set_game_outcome(teams, -2.5,
                             'Bears', 15, 1, 278, 27, 39, 1, 0, 379,
                             'Titans', 19, 2, 200, 17, 29, 2, 0, 354)

    teams = set_game_outcome(teams, -2.5,
                             'Packers', 27, 3, 101, 12, 22, 2, 1, 211,
                             'Chiefs', 20, 2, 198, 25, 39, 2, 1, 294)

    teams = set_game_outcome(teams, 3.5,
                             'Panthers', 25, 3, 281, 22, 34, 3, 2, 383,
                             'Steelers', 19, 1, 182, 20, 36, 1, 1, 200)

    teams = set_game_outcome(teams, -3.5,
                             'Saints', 13, 2, 181, 22, 30, 1, 0, 302,
                             'Dolphins', 16, 1, 228, 22, 29, 1, 0, 345)

    teams = set_game_outcome(teams, 4,
                             '49ers', 24, 3, 122, 13, 25, 1, 2, 285,
                             'Chargers', 27, 3, 152, 13, 22, 0, 2, 343)

    teams = set_game_outcome(teams, -2.5,
                             'Seahawks', 17, 2, 80, 5, 14, 2, 0, 214,
                             'Raiders', 15, 1, 214, 29, 40, 0, 0, 328)

    teams = set_game_outcome(teams, -3,
                             'Bills', 27, 2, 162, 22, 33, 1, 0, 282,
                             'Vikings', 23, 2, 200, 24, 34, 1, 2, 336)

    teams = set_game_outcome(teams, -2.5,
                             'Patriots', 29, 3, 202, 18, 28, 2, 1, 318,
                             'Giants', 31, 4, 332, 30, 56, 3, 2, 425)

    teams = set_game_outcome(teams, -4,
                             'Jets', 6, 0, 292, 36, 46, 0, 2, 338,
                             'Eagles', 0, 0, 64, 12, 23, 0, 1, 103)

    teams = set_game_outcome(teams, -3,
                             'Bengals', 6, 0, 227, 27, 41, 0, 0, 312,
                             'Colts', 13, 1, 218, 21, 32, 0, 1, 314)

    teams = set_game_outcome(teams, -4.5,
                             'Browns', 20, 2, 224, 24, 41, 1, 2, 336,
                             'Lions', 16, 2, 141, 14, 24, 1, 0, 199)

    teams = set_game_outcome(teams, -2.5,
                             'Texans', 10, 1, 162, 23, 35, 1, 2, 259,
                             'Rams', 22, 3, 225, 20, 32, 2, 2, 344)

    teams = set_game_outcome(teams, -4,
                             'Jaguars', 12, 1, 102, 17, 38, 0, 0, 259,
                             'Falcons', 31, 4, 170, 15, 25, 1, 0, 425)

    teams = set_game_outcome(teams, -2,
                             'Broncos', 20, 2, 216, 20, 34, 1, 3, 340,
                             'Cardinals', 7, 1, 176, 20, 33, 1, 1, 229)

    return teams


def week_1(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week1_schedule())

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
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_2(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week2_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week3_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week4_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week5_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week6_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week7_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week8_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week9_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week10_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week11_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week12_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week13_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week14_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week15_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week16_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_week17_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_wildcard_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_divisional_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_conference_schedule())

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
        Predictor.get_week_probabilities(teams, Playoffs.get_superbowl_schedule())

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
