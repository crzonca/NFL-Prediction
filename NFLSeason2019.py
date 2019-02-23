import statistics

from prettytable import PrettyTable

import Projects.nfl.NFL_Prediction.NFLPredictor as Predictor


def season():
    eliminated_teams = list()
    teams = handle_week(None, 'Preseason', set_up_teams, eliminated_teams)
    teams = handle_week(teams, 'Week 1', week_1, eliminated_teams, full_standings=True)


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


def handle_week(teams, week_name, week, eliminated_teams, full_standings=False):
    print(week_name)
    if teams:
        teams = week(teams)
    else:
        teams = week()
    print_league_details(teams, eliminated_teams, full_standings=full_standings)
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
        team_info.append(round(team[4], 3))
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
    table.float_format = '0.3'
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
    table.float_format = '0.3'
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
