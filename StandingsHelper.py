import statistics

from prettytable import PrettyTable

import Projects.nfl.NFL_Prediction.NFLGraph as Graph
import Projects.nfl.NFL_Prediction.NFLPredictor as Predictor


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


def print_elo_rankings(teams, eliminated_teams, include_title=True):
    """
    Displays the team rankings information for a set of teams.

    :param teams: The list of teams to display information for
    :param eliminated_teams: The list of teams to exclude from the standings
    :param include_title: If a header for the rankings information should be included
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs

    # Sort the teams by elo, then by playoff tiebreakers
    teams = Playoffs.sort_by_tiebreakers(teams)
    sorted_by_elo = sorted(teams, reverse=True, key=lambda tup: tup[4])

    # Remove the eliminated teams
    remaining_teams = list(filter(lambda t: t[0] not in eliminated_teams, sorted_by_elo))

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Tier'])
    table.float_format = '0.3'

    # Add the info to the rows for each team that isnt eliminated
    for rank, team in enumerate(sorted_by_elo):
        row = list()
        row.append(rank + 1)
        team_info = list()
        if team not in remaining_teams:
            team_info.append(team[0] + '*')
        else:
            team_info.append(team[0])
        team_info.append(round(team[1]))
        team_info.append(round(team[2]))
        team_info.append(round(team[3]))
        team_info.append('{0:.3f}'.format(team[4]))
        team_info.append(get_tier(teams, team))
        row = row + team_info

        # Color the row based of the teams tier
        row = [get_tier_color(row[-1], val) for val in row]
        table.add_row(row)

    # Print the table
    if include_title:
        print('Elo Rankings')
    print(table)
    print()


def print_standings(teams, eliminated_teams, include_title=True):
    """
    Displays partial team standings information for a set of teams.

    :param teams: The list of teams to display information for
    :param eliminated_teams: The list of teams to exclude from the standings
    :param include_title: If a header for the standings information should be included
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs

    # Remove the eliminated teams
    remaining_teams = list(filter(lambda t: t[0] not in eliminated_teams, teams))

    # Sort the teams by playoff tiebreakers
    teams = Playoffs.sort_by_tiebreakers(teams)

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo'])
    table.float_format = '0.3'

    # Add the info to the rows for each team that isnt eliminated
    for rank, team in enumerate(teams):
        row = list()
        row.append(rank + 1)
        team_info = list()
        if team not in remaining_teams:
            team_info.append(team[0] + '*')
        else:
            team_info.append(team[0])
        team_info.append(round(team[1]))
        team_info.append(round(team[2]))
        team_info.append(round(team[3]))
        team_info.append(team[4])
        row = row + team_info
        table.add_row(row)

    # Print the table
    if include_title:
        print('Standings')
    print(table)
    print()


def calculate_passer_rating(yards, completions, attempts, tds, ints):
    average_completion_pct = completions / attempts if attempts > 0 else 0
    a = (average_completion_pct - .3) * 5

    average_yards_per_pass_attempt = yards / attempts if attempts > 0 else 0
    b = (average_yards_per_pass_attempt - 3) * .25

    average_passing_touchdowns_per_attempt = tds / attempts if attempts > 0 else 0
    c = average_passing_touchdowns_per_attempt * 20

    average_ints_per_attempt = ints / attempts if attempts > 0 else 0
    d = 2.375 - (average_ints_per_attempt * 25)

    a = 2.375 if a > 2.375 else 0 if a < 0 else a
    b = 2.375 if b > 2.375 else 0 if b < 0 else b
    c = 2.375 if c > 2.375 else 0 if c < 0 else c
    d = 2.375 if d > 2.375 else 0 if d < 0 else d

    rating = ((a + b + c + d) / 6) * 100

    return rating


def print_full_standings(teams, eliminated_teams, include_title=True):
    """
    Displays the full team standings information for a set of teams.

    :param teams: The list of teams to display information for
    :param eliminated_teams: The list of teams to exclude from the standings
    :param include_title: If a header for the standings information should be included
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs

    # Remove the eliminated teams
    remaining_teams = list(filter(lambda t: t[0] not in eliminated_teams, teams))

    # Sort the teams by playoff tiebreakers
    teams = Playoffs.sort_by_tiebreakers(teams)

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Points', 'Points Against', 'Point Diff.',
                         'Touchdowns', 'Passer Rating', 'Total Yards'])
    table.float_format = '0.3'

    # Add the info to the rows for each team that isnt eliminated
    for rank, team in enumerate(teams):
        row = list()
        row.append(rank + 1)
        team_info = list()
        if team not in remaining_teams:
            team_info.append(team[0] + '*')
        else:
            team_info.append(team[0])
        team_info.append(round(team[1]))
        team_info.append(round(team[2]))
        team_info.append(round(team[3]))
        team_info.append(team[4])
        team_info.append(round(team[5], 3))
        team_info.append(round(team[6], 3))
        team_info.append(round(team[5] - team[6], 3))
        team_info.append(round(team[7], 3))
        passer_rating = calculate_passer_rating(team[8], team[9], team[10], team[11], team[12])
        team_info.append(round(passer_rating, 3))
        team_info.append(round(team[13], 3))
        row = row + team_info
        table.add_row(row)

    # Print the table
    if include_title:
        print('Standings')
    print(table)
    print()


def get_full_standings_csv(teams):
    """
    Gets the league's full standings and writes it to a csv file.

    :param teams: The list of teams to include in the standings
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
    import pandas as pd

    # Sort the teams by playoff tiebreakers
    teams = Playoffs.sort_by_tiebreakers(teams)

    standings = pd.DataFrame()
    standings['Rank'] = range(1, 33)
    standings['Name'] = [team[0] for team in teams]
    standings['Wins'] = [team[1] for team in teams]
    standings['Losses'] = [team[2] for team in teams]
    standings['Ties'] = [team[3] for team in teams]
    standings['Elo'] = [team[4] for team in teams]
    standings['Points'] = [team[5] for team in teams]
    standings['Points Against'] = [team[6] for team in teams]
    standings['Point Differential'] = [team[5] - team[6] for team in teams]
    standings['Touchdowns'] = [team[7] for team in teams]
    standings['Passer Rating'] = [calculate_passer_rating(team[8], team[9], team[10], team[11], team[12])
                                  for team in teams]
    standings['Total Yards'] = [team[13] for team in teams]

    standings.to_csv('..\\Projects\\nfl\\NFL_Prediction\\2019Ratings\\2019Standings.csv', index=False)


def print_division_rankings(teams):
    """
    Displays the team rankings information for each division.

    :param teams: The list of all the teams in the league
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
    nfl = Playoffs.get_league_structure()
    for conference_name, conference in nfl.items():
        for division_name, division in conference.items():
            division_teams = list()
            for team in division:
                division_teams.append(get_team(teams, team))
            print(division_name)
            other_divisions = list(set(teams) - set(division_teams))
            print_elo_rankings(teams, [team[0] for team in other_divisions], False)


def print_division_standings(teams):
    """
    Displays the team standings information for each division.

    :param teams: The list of all the teams in the league
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
    nfl = Playoffs.get_league_structure()
    for conference_name, conference in nfl.items():
        for division_name, division in conference.items():
            division_teams = list()
            for team in division:
                division_teams.append(get_team(teams, team))
            print(division_name)
            other_divisions = list(set(teams) - set(division_teams))
            print_standings(teams, [team[0] for team in other_divisions], False)


def print_full_division_standings(teams):
    """
    Displays the full team standings information for each division.

    :param teams: The list of all the teams in the league
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
    nfl = Playoffs.get_league_structure()
    for conference_name, conference in nfl.items():
        for division_name, division in conference.items():
            division_teams = list()
            for team in division:
                division_teams.append(get_team(teams, team))
            print(division_name)
            other_divisions = list(set(teams) - set(division_teams))
            print_full_standings(teams, [team[0] for team in other_divisions], False)


def print_model_rankings(teams, eliminated_teams):
    """
    Displays team rankings based off each team's chance against an average team determined by the model.

    :param teams: The list of teams to display information for
    :param eliminated_teams: The list of teams to exclude from the rankings
    :return: Void
    """
    # Get every teams chance to beat the perfectly average team in the league
    team_probs = list()
    for team in teams:
        team_probs.append((team, round(get_chance_against_average(teams, team), 3)))

    # Sort the teams by their chance to beat an average team, then by elo
    team_probs.sort(key=lambda tup: tup[0][4], reverse=True)
    team_probs.sort(key=lambda tup: tup[1], reverse=True)

    # Remove the victory probability from the team info
    teams = [team[0] for team in team_probs]

    # Remove the eliminated teams
    remaining_teams = list(filter(lambda t: t[0] not in eliminated_teams, teams))

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Tier'])
    table.float_format = '0.3'

    # Add the info to the rows for each team that isnt eliminated
    for rank, team in enumerate(teams):
        row = list()
        row.append(rank + 1)
        team_info = list()
        if team not in remaining_teams:
            team_info.append(team[0] + '*')
        else:
            team_info.append(team[0])
        team_info.append(round(team[1]))
        team_info.append(round(team[2]))
        team_info.append(round(team[3]))
        team_info.append(round(team[4], 3))
        team_info.append(get_tier(teams, team))
        row = row + team_info

        # Color the row based of the teams tier
        row = [get_tier_color(row[-1], val) for val in row]
        table.add_row(row)

    # Print the table
    print('Model Rankings')
    print(table)
    print()


def print_monte_carlo_simulation(teams):
    """
    Displays each teams end of season predictions based off the monte carlo simulations.

    :param teams: The list of all the teams in the league, teams here include a playoff chance determined by monte_carlo
    :return: Void
    """
    # Sort the teams by wins, then least losses, then point differential, then total yards
    sorted_by_losses = sorted(teams, key=lambda tup: tup[2])
    sorted_by_wins = sorted(sorted_by_losses, reverse=True, key=lambda tup: tup[1])

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Playoff Chances'])
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
        team_info.append(team[5])
        row = row + team_info

        table.add_row(row)

    # Print the table
    print('Projected Standings')
    print(table)
    print()


def print_schedule_difficulty(teams, remaining=False, completed=False):
    """
    Displays the schedule difficulty information for all teams in the league. Displays schedule difficulty as a
    percentile compared to other difficulties.  Percentile is based off a normal distribution centered at 1500 with a
    deviation derived by bootstrapping other schedule difficulties using typical team strengths.

    :param teams: The list of all the teams in the league
    :param remaining: If the schedule difficulty should only be based off of each teams remaining games
    :param completed: If the schedule difficulty of only completed games should be used
    :return: Void
    """
    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
    from scipy.stats import norm
    import statistics

    # Get each teams schedule difficulty
    team_tuples = list()
    for team in teams:
        if completed:
            remaining = False
            schedule_difficulty, deviation = Playoffs.get_completed_schedule_difficulty(team[0])
        else:
            schedule_difficulty, deviation = Playoffs.get_schedule_difficulty(teams, team[0], remaining)
        team_tuples.append((team, schedule_difficulty, deviation))

    # Sort the teams
    sorted_by_difficulty = sorted(team_tuples, key=lambda tup: tup[1], reverse=True)

    # Create a normal distribution for schedule difficulties
    if remaining:
        # If not remaining games only, use the mean and standard deviation of the current difficulties
        all_schedules = list()
        for team_tuple in sorted_by_difficulty:
            all_schedules.append(team_tuple[1])
        difficulty_deviation = statistics.pstdev(all_schedules) if statistics.pstdev(all_schedules) > 0 else 1e-10
        difficulty_norm = norm(statistics.mean(all_schedules), difficulty_deviation)
    else:
        # If not remaining games only, use a mean of 1500 and a standard deviation of 13.615
        difficulty_norm = norm(1500, 13.615)

    # Standardize each team's difficulty
    schedule_difficulties = list()
    for team_tuple in sorted_by_difficulty:
        team = team_tuple[0]
        schedule_difficulty = team_tuple[1]
        deviation = team_tuple[2]

        schedule_difficulty = difficulty_norm.cdf(schedule_difficulty) * 100

        schedule_difficulties.append((team, schedule_difficulty, deviation))

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Games Played', 'Elo', 'Schedule Diff.'])
    table.float_format = '0.3'

    # Add the info to the rows
    for rank, team_tuple in enumerate(schedule_difficulties):
        row = list()
        team = team_tuple[0]
        schedule_difficulty = team_tuple[1]
        row.append(rank + 1)
        team_info = list()
        team_info.append(team[0])
        games_played = team[1] + team[2] + team[3]
        team_info.append(games_played)
        team_info.append(team[4])
        team_info.append(schedule_difficulty)
        row = row + team_info

        table.add_row(row)

    # Print the table
    if completed:
        print('Completed Schedule Difficulties')
    else:
        if remaining:
            print('Remaining Schedule Difficulties')
        else:
            print('Schedule Difficulties')
    print(table)
    print()


def print_league_details(teams, eliminated_teams, full_standings=True, divisional_view=False, model_rankings=False):
    """
    Display's the current standings and rankings of a set of teams.

    :param teams: The list of teams to display information for
    :param eliminated_teams: The list of teams to exclude from the rankings
    :param full_standings: If the standings should display full details, if not just record and elo are displayed
    :param divisional_view: If the standings should be separated based on the divisions in the league
    :param model_rankings: If rankings based off the model should be included
    :return: Void
    """

    # Print the standings
    if full_standings:
        if divisional_view:
            print_full_division_standings(teams)
        else:
            print_full_standings(teams, eliminated_teams)
    else:
        if divisional_view:
            print_division_standings(teams)
        else:
            print_standings(teams, eliminated_teams)

    # Print the elo rankings
    if divisional_view:
        print_division_rankings(teams)
    else:
        print_elo_rankings(teams, eliminated_teams)

    # If model rankings are desired, print the model rankings
    if model_rankings:
        print_model_rankings(teams, eliminated_teams)


def print_team_pagerank(teams):
    ranks = Graph.page_rank_teams()

    # Create the table header
    table = PrettyTable(['Rank', 'Name', 'Wins', 'Losses', 'Ties', 'Elo', 'Score'])
    table.float_format = '0.3'

    # Add the info to the rows
    for rank, info in enumerate(ranks):
        name = info[0]
        score = round(100 * info[1], 3)
        row = list()
        row.append(rank + 1)
        team = get_team(teams, name)
        team_info = list()
        team_info.append(team[0])
        team_info.append(team[1])
        team_info.append(team[2])
        team_info.append(team[3])
        team_info.append(team[4])
        team_info.append(score)
        row = row + team_info

        table.add_row(row)

    # Print the table
    print('PageRanked Teams')
    print(table)
    print()


def compare_teams(teams, team_name1, team_name2):
    team1 = get_team(teams, team_name1)
    team2 = get_team(teams, team_name2)

    # Create the table header
    table = PrettyTable([team_name1, 'Stat', team_name2])
    table.float_format = '0.3'

    stats = ['Record',
             'Elo',
             'Points For',
             'Points Against',
             'Point Diff.',
             'Total TDs',
             'Pass Yards',
             'Completions',
             'Attempts',
             'Completion %',
             'Pass TDs',
             'Interceptions',
             'Passer Rating',
             'Total Yards']
    
    team1_games_played = team1[1] + team1[2] + team1[3]
    team1_info = [str(team1[1]) + '-' + str(team1[2]) + '-' + str(team1[3]),
                  round(team1[4], 3),
                  int(team1[5] * team1_games_played),
                  int(team1[6] * team1_games_played),
                  int(team1[5] * team1_games_played) - int(team1[6] * team1_games_played),
                  int(team1[7] * team1_games_played),
                  team1[8],
                  team1[9],
                  team1[10],
                  round(100 * team1[9] / team1[10], 3),
                  team1[11],
                  team1[12],
                  round(calculate_passer_rating(team1[8], team1[9], team1[10], team1[11], team1[12]), 3),
                  int(team1[13] * team1_games_played)]

    team2_games_played = team2[1] + team2[2] + team2[3]
    team2_info = [str(team2[1]) + '-' + str(team2[2]) + '-' + str(team2[3]),
                  round(team2[4], 3),
                  int(team2[5] * team2_games_played),
                  int(team2[6] * team2_games_played),
                  int(team2[5] * team2_games_played) - int(team2[6] * team2_games_played),
                  int(team2[7] * team2_games_played),
                  team2[8],
                  team2[9],
                  team2[10],
                  round(100 * team2[9] / team2[10], 3),
                  team2[11],
                  team2[12],
                  round(calculate_passer_rating(team2[8], team2[9], team2[10], team2[11], team2[12]), 3),
                  int(team2[13] * team2_games_played)]

    def compare_fields():
        record_comp = team1[1] / team1_games_played >= team2[1] / team2_games_played
        elo_comp = team1[4] >= team2[4]
        pf_comp = team1[5] >= team2[5]
        pa_comp = team1[6] <= team2[6]
        pd_comp = team1[5] - team1[6] >= team2[5] - team2[6]
        tot_tds_comp = team1[7] >= team2[7]
        pass_yards_comp = team1[8] >= team2[8]
        pass_completions_comp = team1[9] >= team2[9]
        pass_attempts_comp = team1[10] <= team2[10]
        completion_pct_comp = team1[9] / team1[10] >= team2[9] / team2[10]
        pass_tds_comp = team1[11] >= team2[11]
        ints_comp = team1[12] <= team2[12]
        passer_rating_comp = (calculate_passer_rating(team1[8], team1[9], team1[10], team1[11], team1[12]) >=
                              calculate_passer_rating(team2[8], team2[9], team2[10], team2[11], team2[12]))
        total_yards_comp = team1[13] >= team2[13]
        return [record_comp,
                elo_comp,
                pf_comp,
                pa_comp,
                pd_comp,
                tot_tds_comp,
                pass_yards_comp,
                pass_completions_comp,
                pass_attempts_comp,
                completion_pct_comp,
                pass_tds_comp,
                ints_comp,
                passer_rating_comp,
                total_yards_comp]

    team1_info = [Colors.WHITE + str(field) + Colors.ENDC if better
                  else Colors.BRIGHT_GRAY + str(field) + Colors.ENDC
                  for field, better in zip(team1_info, compare_fields())]

    team2_info = [Colors.WHITE + str(field) + Colors.ENDC if not better
                  else Colors.BRIGHT_GRAY + str(field) + Colors.ENDC
                  for field, better in zip(team2_info, compare_fields())]

    table_info = [team1_info, stats, team2_info]

    # Add the info to the rows for each team that isnt eliminated
    for row in list(map(list, zip(*table_info))):
        table.add_row(row)

    # Print the table
    print(team_name1 + ' vs ' + team_name2)
    print(table)
    print()


def get_tier(teams, team):
    """
    "Calculates a team's 'tier' based on the team's elo. Tiers are calculated based on difference from the mean elo,
    expressed in fractions of the standard deviation.

    :param teams: The list of all the teams in the league
    :param team: The team to evaluate
    :return: The team's tier
    """

    # Get the team's elo
    team_elo = team[4]

    # Get all the elos in the league
    elos = [team_info[4] for team_info in teams]

    # Determine the glass delta for the team's elo
    avg_elo = statistics.mean(elos)
    elo_dev = statistics.pstdev(elos)
    glass_delta = (team_elo - avg_elo) / elo_dev

    # Determine the tier based on the effect size
    if glass_delta > 8 / 3:
        tier = 'S+'
    elif glass_delta > 7 / 3:
        tier = 'S'
    elif glass_delta > 2:
        tier = 'S-'
    elif glass_delta > 5 / 3:
        tier = 'A+'
    elif glass_delta > 4 / 3:
        tier = 'A'
    elif glass_delta > 1:
        tier = 'A-'
    elif glass_delta > 2 / 3:
        tier = 'B+'
    elif glass_delta > 1 / 3:
        tier = 'B'
    elif glass_delta >= 0:
        tier = 'B-'
    elif glass_delta > -1 / 3:
        tier = 'C+'
    elif glass_delta > -2 / 3:
        tier = 'C'
    elif glass_delta > -1:
        tier = 'C-'
    elif glass_delta > -4 / 3:
        tier = 'D+'
    elif glass_delta > -5 / 3:
        tier = 'D'
    elif glass_delta > -2:
        tier = 'D-'
    elif glass_delta > -7 / 3:
        tier = 'F+'
    elif glass_delta > -8 / 3:
        tier = 'F'
    else:
        tier = 'F-'
    return tier


def get_tier_color(tier, string):
    """
    Colors a string based on a given 'tier'.

    :param tier: The given, predefined, tier of a team
    :param string: The text to color
    :return: The original text, colored in the 'tier color'
    """
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
    """
    Creates a team with average stats amongst all teams in the league.  Stats are average from teams at the current
    point in the season.

    :param teams: The list of all the teams in the league
    :return: A team with each stat as average in the league at that point in the season
    """

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

    # Get every team's stats
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

    # Average every team's stats
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

    return ('Average', avg_wins, avg_losses, avg_ties, avg_elo, avg_average_points_for, avg_average_points_against,
            avg_average_tds, avg_total_net_pass_yards, avg_total_pass_completions, avg_total_pass_attempts,
            avg_total_pass_tds, avg_total_interceptions_thrown, avg_average_total_yards, avg_average_total_yards)


def get_chance_against_average(teams, team):
    """
    Predicts a team's percent chance to defeat a team that has completely average stats within the league.  Spread
    is determined as the difference in each team's point differential. Match up is treated as a neutral field game.

    :param teams: The list of all the teams in the league
    :param team: The team to evaluate
    :return: The chance of victory of the evaluated team against an average team
    """

    # Get a perfectly average team for the league
    avg_team = get_average_team(teams)

    # Get the teams average point differential
    team_avg_points_diff = team[5] - team[6]

    # Get the "average" teams average point differential
    avg_team_avg_points_diff = avg_team[5] - avg_team[6]

    # The teams expected margin of victory is their point differential less the average teams point differential
    spread = team_avg_points_diff - avg_team_avg_points_diff

    # Calculate the teams chances of victory if they were the home team and if they were the away team
    home_vote_prob, home_lr_prob, home_svc_prob, home_rf_prob = Predictor.predict_home_victory(team, avg_team, -spread)
    away_vote_prob, away_lr_prob, away_svc_prob, away_rf_prob = Predictor.predict_away_victory(avg_team, team, -spread)

    # Return the average of the teams chance of victory
    avg_prob = (home_vote_prob + away_vote_prob) / 2
    return avg_prob


class Colors:
    GRAY_BOX = '\033[100m'
    BLACK = '\033[97m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_PURPLE = '\033[95m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_RED = '\033[91m'
    GRAY = '\033[90m'
    BRIGHT_GRAY_BOX = '\033[47m'
    CYAN_BOX = '\033[46m'
    PURPLE_BOX = '\033[45m'
    BLUE_BOX = '\033[44m'
    YELLOW_BOX = '\033[43m'
    GREEN_BOX = '\033[42m'
    RED_BOX = '\033[41m'
    WHITE_BOX = '\033[40m'
    BRIGHT_GRAY = '\033[37m'
    CYAN = '\033[36m'
    PURPLE = '\033[35m'
    BLUE = '\033[34m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    WHITE = '\033[30m'
    BOX = '\033[7m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'
