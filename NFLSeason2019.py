import Projects.nfl.NFL_Prediction.NFLPredictor as Predictor
import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
import Projects.nfl.NFL_Prediction.StandingsHelper as Standings


def season():
    print('Preseason')
    teams = set_up_teams()
    eliminated_teams = list()

    Standings.print_league_details(teams, eliminated_teams, full_standings=False)
    Standings.print_schedule_difficulty(teams)
    print('*' * 120, '\n')

    teams = handle_week(teams, 'Week 1', week_1, eliminated_teams, suppress_probabilities=False)

    Standings.print_schedule_difficulty(teams, remaining=True)
    Playoffs.monte_carlo(teams, trials=100)

    print('Current Playoff Picture')
    afc_playoff_teams, nfc_playoff_teams = Playoffs.get_playoff_picture(teams)
    print('-' * 15 + 'AFC' + '-' * 15)
    Playoffs.create_playoff_bracket(afc_playoff_teams)
    print()
    print('-' * 15 + 'NFC' + '-' * 15)
    Playoffs.create_playoff_bracket(nfc_playoff_teams)


def set_up_teams():
    def create_base_team(team_name, elo):
        return team_name, 0, 0, 0, elo, 0, 0, 0, 0, 0, 0, 0, 0, 0

    teams = list()
    teams.append(create_base_team('49ers', 1431.803))
    teams.append(create_base_team('Bears', 1547.207))
    teams.append(create_base_team('Bengals', 1450.022))
    teams.append(create_base_team('Bills', 1467.667))
    teams.append(create_base_team('Broncos', 1447.034))
    teams.append(create_base_team('Browns', 1465.156))
    teams.append(create_base_team('Buccaneers', 1437.01))
    teams.append(create_base_team('Cardinals', 1414.907))
    teams.append(create_base_team('Chargers', 1566.415))
    teams.append(create_base_team('Chiefs', 1562.643))
    teams.append(create_base_team('Colts', 1533.559))
    teams.append(create_base_team('Cowboys', 1555.546))
    teams.append(create_base_team('Dolphins', 1458.772))
    teams.append(create_base_team('Eagles', 1561.189))
    teams.append(create_base_team('Falcons', 1501.919))
    teams.append(create_base_team('Giants', 1442.666))
    teams.append(create_base_team('Jaguars', 1447.748))
    teams.append(create_base_team('Jets', 1408.145))
    teams.append(create_base_team('Lions', 1476.841))
    teams.append(create_base_team('Packers', 1463.462))
    teams.append(create_base_team('Panthers', 1486.623))
    teams.append(create_base_team('Patriots', 1608.682))
    teams.append(create_base_team('Raiders', 1433.093))
    teams.append(create_base_team('Rams', 1589.463))
    teams.append(create_base_team('Ravens', 1537.051))
    teams.append(create_base_team('Redskins', 1465.572))
    teams.append(create_base_team('Saints', 1592.439))
    teams.append(create_base_team('Seahawks', 1535.119))
    teams.append(create_base_team('Steelers', 1544.046))
    teams.append(create_base_team('Texans', 1515.959))
    teams.append(create_base_team('Titans', 1520.108))
    teams.append(create_base_team('Vikings', 1532.134))
    return teams


def handle_week(teams,
                week_name,
                week,
                eliminated_teams,
                suppress_probabilities=False,
                full_standings=True,
                divisional_view=False,
                model_rankings=False):
    # Print the week name
    print(week_name)

    # Handle the week
    teams = week(teams, suppress_probabilities)

    # Print the tables
    Standings.print_league_details(teams,
                                   eliminated_teams,
                                   full_standings=full_standings,
                                   divisional_view=divisional_view,
                                   model_rankings=model_rankings)

    print('*' * 120, '\n')

    # Return the updated teams
    return teams


def week_1(teams, suppress_probabilities):
    if not suppress_probabilities:
        Predictor.get_week_probabilities(teams, Playoffs.get_2019_week1_schedule())

    teams = set_game_outcome(teams,
                             home_name='Vikings',
                             home_score=17,
                             home_touchdowns=2,
                             home_net_pass_yards=264,
                             home_pass_completions=17,
                             home_pass_attempts=26,
                             home_pass_tds=1,
                             home_interceptions_thrown=0,
                             home_total_yards=407,
                             away_name='Saints',
                             away_score=13,
                             away_touchdowns=1,
                             away_net_pass_yards=217,
                             away_pass_completions=15,
                             away_pass_attempts=22,
                             away_pass_tds=0,
                             away_interceptions_thrown=1,
                             away_total_yards=322)

    teams = set_game_outcome(teams,
                             'Lions', 7, 1, 386, 19, 22, 1, 0, 440,
                             'Bears', 14, 2, 182, 14, 24, 0, 0, 355)

    return teams


def get_team(teams, team_name):
    for team in teams:
        if team[0] == team_name:
            return team


def set_game_outcome(teams,
                     home_name, home_score, home_touchdowns, home_net_pass_yards, home_pass_completions,
                     home_pass_attempts, home_pass_tds, home_interceptions_thrown, home_total_yards,
                     away_name, away_score, away_touchdowns, away_net_pass_yards, away_pass_completions,
                     away_pass_attempts, away_pass_tds, away_interceptions_thrown, away_total_yards):
    teams = Predictor.update_teams(teams,
                                   home_name, home_score, home_touchdowns, home_net_pass_yards, home_pass_completions,
                                   home_pass_attempts, home_pass_tds, home_interceptions_thrown, home_total_yards,
                                   away_name, away_score, away_touchdowns, away_net_pass_yards, away_pass_completions,
                                   away_pass_attempts, away_pass_tds, away_interceptions_thrown, away_total_yards)

    Playoffs.completed_games.append((home_name, home_score, away_name, away_score))

    return teams
