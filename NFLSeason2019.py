import statistics

from prettytable import PrettyTable

import Projects.nfl.NFL_Prediction.NFLPredictor as Predictor


def season():
    eliminated_teams = list()
    teams = handle_week(None, 'Preseason', set_up_teams, eliminated_teams)
    teams = handle_week(teams, 'Week 1', week_1, eliminated_teams, full_standings=True, model_rankings=True)


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
    teams.append(create_base_team('Patriots', 1608.682))
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


def handle_week(teams, week_name, week, eliminated_teams, full_standings=False, model_rankings=False):
    # Print the week name
    print(week_name)

    # If the league already has teams
    if teams:
        # Handle the week
        teams = week(teams)
    else:
        # Otherwise do setup
        teams = week()
    # Print the tables
    print_league_details(teams, eliminated_teams, full_standings=full_standings, model_rankings=model_rankings)

    # Return the updated teams
    return teams


def week_1(teams):
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1*spread otherwise)
    probabilities = list()
    probabilities.append(Predictor.predict_game_outcome(teams, 'Vikings', 'Lions', -6, verbose=True))
    probabilities.append(Predictor.predict_game_outcome(teams, 'Packers', 'Bears', 3))

    probabilities.sort(key=lambda outcome: outcome[0], reverse=True)
    for game in probabilities:
        print(game[1])
    print()

    teams = Predictor.update_teams(teams,
                                   home_name='Vikings',
                                   home_score=17,
                                   home_touchdowns=2,
                                   home_net_pass_yards=264,
                                   home_pass_completions=17,
                                   home_pass_attempts=26,
                                   home_pass_tds=1,
                                   home_interceptions_thrown=0,
                                   home_total_yards=407,
                                   home_first_downs=28,
                                   home_third_down_conversions=9,
                                   home_third_downs=15,
                                   away_name='Lions',
                                   away_score=13,
                                   away_touchdowns=1,
                                   away_net_pass_yards=217,
                                   away_pass_completions=15,
                                   away_pass_attempts=22,
                                   away_pass_tds=0,
                                   away_interceptions_thrown=1,
                                   away_total_yards=322,
                                   away_first_downs=21,
                                   away_third_down_conversions=7,
                                   away_third_downs=13)

    teams = Predictor.update_teams(teams, 'Packers', 7, 1, 386, 19, 22, 1, 0, 440, 28, 11, 18,
                                   'Bears', 14, 2, 182, 14, 24, 0, 0, 355, 26, 10, 15)

    return teams


def print_elo_rankings(teams, eliminated_teams):
    # Sort the teams by elo, then wins, then least losses
    sorted_by_losses = sorted(teams, key=lambda tup: tup[2])
    sorted_by_wins = sorted(sorted_by_losses, reverse=True, key=lambda tup: tup[1])
    sorted_by_elo = sorted(sorted_by_wins, reverse=True, key=lambda tup: tup[4])

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Tier'])
    table.float_format = '0.3'

    # Add the info to the rows
    for rank, team in enumerate(sorted_by_elo):
        row = list()
        row.append(rank + 1)
        team_info = list()
        team_info.append(team[0])
        team_info.append(team[1])
        team_info.append(team[2])
        team_info.append(team[3])
        team_info.append(round(team[4], 3))
        team_info.append(get_tier(teams, team))
        row = row + team_info

        # Color the row based of the teams tier
        row = [get_tier_color(row[-1], val) for val in row]

        # Add the row to the table if the team isnt eliminated
        if team[0] not in eliminated_teams:
            table.add_row(row)

    # Print the table
    print('Elo Rankings')
    print(table)
    print()


def print_standings(teams, eliminated_teams):
    # Sort the teams by wins, then least losses, then point differential, then total yards
    sorted_by_yards = sorted(teams, reverse=True, key=lambda tup: tup[13])
    sorted_by_point_diff = sorted(sorted_by_yards, reverse=True, key=lambda tup: tup[5] - tup[6])
    sorted_by_losses = sorted(sorted_by_point_diff, key=lambda tup: tup[2])
    sorted_by_wins = sorted(sorted_by_losses, reverse=True, key=lambda tup: tup[1])

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo'])
    table.float_format = '0.3'

    # Add the info to the rows
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

        # Add the row to the table if the team isnt eliminated
        if team[0] not in eliminated_teams:
            table.add_row(row)

    # Print the table
    print('Standings')
    print(table)
    print()


def print_full_standings(teams, eliminated_teams):
    # Sort the teams by wins, then least losses, then point differential, then total yards
    sorted_by_yards = sorted(teams, reverse=True, key=lambda tup: tup[13])
    sorted_by_point_diff = sorted(sorted_by_yards, reverse=True, key=lambda tup: tup[5] - tup[6])
    sorted_by_losses = sorted(sorted_by_point_diff, key=lambda tup: tup[2])
    sorted_by_wins = sorted(sorted_by_losses, reverse=True, key=lambda tup: tup[1])

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Points', 'Points Against', 'Point Diff.',
                         'Touchdowns', 'Passer Rating', 'Total Yards', 'First Downs', '3rd Down %'])
    table.float_format = '0.3'

    # Add the info to the rows
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

        # Add the row to the table if the team isnt eliminated
        if team[0] not in eliminated_teams:
            table.add_row(row)

    # Print the table
    print('Standings')
    print(table)
    print()


def print_model_rankings(teams, eliminated_teams):
    # Get every teams chance to beat the perfectly average team in the league
    team_probs = list()
    for team in teams:
        team_probs.append((team, round(get_chance_against_average(teams, team), 3)))

    # Sort the teams by their chance to beat an average team, then by elo
    team_probs.sort(key=lambda tup: tup[0][4], reverse=True)
    team_probs.sort(key=lambda tup: tup[1], reverse=True)

    # Remove the victory probability from the team info
    teams = [team[0] for team in team_probs]

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Tier'])
    table.float_format = '0.3'

    # Add the info to the rows
    for rank, team in enumerate(teams):
        row = list()
        row.append(rank + 1)
        team_info = list()
        team_info.append(team[0])
        team_info.append(team[1])
        team_info.append(team[2])
        team_info.append(team[3])
        team_info.append(round(team[4], 3))
        team_info.append(get_tier(teams, team))
        row = row + team_info

        # Color the row based of the teams tier
        row = [get_tier_color(row[-1], val) for val in row]

        # Add the row to the table if the team isnt eliminated
        if team[0] not in eliminated_teams:
            table.add_row(row)

    # Print the table
    print('Model Rankings')
    print(table)
    print()


def print_league_details(teams, eliminated_teams, full_standings=False, model_rankings=False):
    # Print the standings
    if full_standings:
        print_full_standings(teams, eliminated_teams)
    else:
        print_standings(teams, eliminated_teams)

    # Print the elo rankings
    print_elo_rankings(teams, eliminated_teams)

    # If model rankings are desired, print the model rankings
    if model_rankings:
        print_model_rankings(teams, eliminated_teams)


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


def get_average_team(teams):
    all_wins = list()
    all_losses = list()
    all_ties = list()
    all_elo = list()
    all_average_points_for = list()
    all_average_points_against = list()
    all_average_tds = list()
    all_total_net_pass_yards = list()
    all_total_pass_completions = list()
    all_total_pass_attempts = list()
    all_total_pass_tds = list()
    all_total_interceptions_thrown = list()
    all_average_total_yards = list()
    all_average_first_downs = list()
    all_total_third_down_conversions = list()
    all_total_third_downs = list()

    for team in teams:
        all_wins.append(team[1])
        all_losses.append(team[2])
        all_ties.append(team[3])
        all_elo.append(team[4])
        all_average_points_for.append(team[5])
        all_average_points_against.append(team[6])
        all_average_tds.append(team[7])
        all_total_net_pass_yards.append(team[8])
        all_total_pass_completions.append(team[9])
        all_total_pass_attempts.append(team[10])
        all_total_pass_tds.append(team[11])
        all_total_interceptions_thrown.append(team[12])
        all_average_total_yards.append(team[13])
        all_average_first_downs.append(team[14])
        all_total_third_down_conversions.append(team[15])
        all_total_third_downs.append(team[16])

    avg_wins = statistics.mean(all_wins)
    avg_losses = statistics.mean(all_losses)
    avg_ties = statistics.mean(all_ties)
    avg_elo = statistics.mean(all_elo)
    avg_average_points_for = statistics.mean(all_average_points_for)
    avg_average_points_against = statistics.mean(all_average_points_against)
    avg_average_tds = statistics.mean(all_average_tds)
    avg_total_net_pass_yards = statistics.mean(all_total_net_pass_yards)
    avg_total_pass_completions = statistics.mean(all_total_pass_completions)
    avg_total_pass_attempts = statistics.mean(all_total_pass_attempts)
    avg_total_pass_tds = statistics.mean(all_total_pass_tds)
    avg_total_interceptions_thrown = statistics.mean(all_total_interceptions_thrown)
    avg_average_total_yards = statistics.mean(all_average_total_yards)
    avg_average_first_downs = statistics.mean(all_average_first_downs)
    avg_total_third_down_conversions = statistics.mean(all_total_third_down_conversions)
    avg_total_third_downs = statistics.mean(all_total_third_downs)

    return ('Average', avg_wins, avg_losses, avg_ties, avg_elo, avg_average_points_for, avg_average_points_against,
            avg_average_tds, avg_total_net_pass_yards, avg_total_pass_completions, avg_total_pass_attempts,
            avg_total_pass_tds, avg_total_interceptions_thrown, avg_average_total_yards, avg_average_total_yards,
            avg_average_first_downs, avg_total_third_down_conversions, avg_total_third_downs)


def get_chance_against_average(teams, team):
    # Get a perfectly average team for the leage
    avg_team = get_average_team(teams)

    # Get the teams average point differential
    team_avg_points_diff = team[5] - team[6]

    # Get the "average" teams average point differential
    avg_team_avg_points_diff = avg_team[5] - avg_team[6]

    # The teams expected margin of victory is their point differential less the average teams point differential
    spread = team_avg_points_diff - avg_team_avg_points_diff

    # Calculate the teams chances of victory if they were the home team and if they were the away team
    home_vote_prob, home_lr_prob, home_svc_prob, home_rf_prob = Predictor.predict_home_victory(team, avg_team, -spread)
    away_vote_prob, away_lr_prob, away_svc_prob, away_rf_prob = Predictor.predict_away_victory(avg_team, team, spread)

    # Return the average of the teams chance of victory
    avg_prob = (home_vote_prob + away_vote_prob) / 2
    return avg_prob


class Colors:
    BLACK = '\033[97m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_PURPLE = '\033[95m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_RED = '\033[91m'
    GRAY = '\033[90m'
    BRIGHT_GRAY = '\033[37m'
    WHITE = '\033[30m'
    CYAN = '\033[36m'
    PURPLE = '\033[35m'
    BLUE = '\033[34m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'
