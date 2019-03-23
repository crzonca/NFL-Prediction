import random
import statistics
from functools import cmp_to_key

import Projects.nfl.NFL_Prediction.NFL as NFL
import Projects.nfl.NFL_Prediction.StandingsHelper as Standings

completed_games = list()


def get_team(teams, team_name):
    for team in teams:
        if team[0] == team_name:
            return team


def get_league_structure():
    afc = {'AFC North': ['Bengals', 'Browns', 'Ravens', 'Steelers'],
           'AFC East': ['Bills', 'Dolphins', 'Jets', 'Patriots'],
           'AFC South': ['Colts', 'Jaguars', 'Texans', 'Titans'],
           'AFC West': ['Broncos', 'Chargers', 'Chiefs', 'Raiders']}

    nfc = {'NFC North': ['Bears', 'Lions', 'Packers', 'Vikings'],
           'NFC East': ['Cowboys', 'Eagles', 'Giants', 'Redskins'],
           'NFC South': ['Buccaneers', 'Falcons', 'Panthers', 'Saints'],
           'NFC West': ['49ers', 'Cardinals', 'Rams', 'Seahawks']}

    nfl = {'AFC': afc, 'NFC': nfc}
    return nfl


def get_2019_schedule():
    games = list()
    games.append(('Saints', 'Vikings'))
    games.append(('Bears', 'Lions'))
    games.append(('Giants', 'Panthers'))
    games.append(('Seahawks', '49ers'))
    games.append(('Buccaneers', 'Browns'))
    games.append(('Steelers', 'Falcons'))
    games.append(('Bills', 'Dolphins'))
    games.append(('Rams', 'Cardinals'))
    games.append(('Jaguars', 'Broncos'))
    games.append(('Titans', 'Raiders'))
    games.append(('Eagles', 'Packers'))
    games.append(('Redskins', 'Cowboys'))
    games.append(('Texans', 'Colts'))
    games.append(('Patriots', 'Bengals'))
    games.append(('Chiefs', 'Chargers'))
    games.append(('Jets', 'Ravens'))
    games.append(('Titans', 'Steelers'))
    games.append(('Falcons', 'Cardinals'))
    games.append(('Vikings', 'Dolphins'))
    games.append(('Colts', 'Giants'))
    games.append(('Bengals', 'Ravens'))
    games.append(('Packers', 'Bills'))
    games.append(('Cowboys', 'Bears'))
    games.append(('Panthers', 'Buccaneers'))
    games.append(('Browns', 'Chiefs'))
    games.append(('Raiders', 'Rams'))
    games.append(('Broncos', 'Seahawks'))
    games.append(('Chargers', 'Jaguars'))
    games.append(('Jets', 'Patriots'))
    games.append(('Lions', 'Eagles'))
    games.append(('Redskins', 'Texans'))
    games.append(('49ers', 'Saints'))
    games.append(('Chiefs', '49ers'))
    games.append(('Jaguars', 'Eagles'))
    games.append(('Rams', 'Redskins'))
    games.append(('Vikings', 'Lions'))
    games.append(('Buccaneers', 'Steelers'))
    games.append(('Giants', 'Titans'))
    games.append(('Ravens', 'Browns'))
    games.append(('Broncos', 'Colts'))
    games.append(('Patriots', 'Bills'))
    games.append(('Dolphins', 'Jets'))
    games.append(('Cardinals', 'Raiders'))
    games.append(('Panthers', 'Bengals'))
    games.append(('Texans', 'Cowboys'))
    games.append(('Saints', 'Falcons'))
    games.append(('Seahawks', 'Chargers'))
    games.append(('Bears', 'Packers'))
    games.append(('Giants', 'Bears'))
    games.append(('Chargers', 'Cardinals'))
    games.append(('Saints', 'Panthers'))
    games.append(('Rams', 'Seahawks'))
    games.append(('Eagles', 'Redskins'))
    games.append(('Falcons', '49ers'))
    games.append(('Browns', 'Bengals'))
    games.append(('Raiders', 'Texans'))
    games.append(('Steelers', 'Ravens'))
    games.append(('Titans', 'Broncos'))
    games.append(('Packers', 'Lions'))
    games.append(('Bills', 'Jets'))
    games.append(('Jaguars', 'Colts'))
    games.append(('Dolphins', 'Patriots'))
    games.append(('Panthers', 'Bears'))
    games.append(('Ravens', 'Broncos'))
    games.append(('Colts', 'Chiefs'))
    games.append(('Browns', 'Falcons'))
    games.append(('Redskins', 'Packers'))
    games.append(('49ers', 'Eagles'))
    games.append(('Cowboys', 'Titans'))
    games.append(('Cardinals', 'Saints'))
    games.append(('Bills', 'Jaguars'))
    games.append(('Bengals', 'Buccaneers'))
    games.append(('Lions', 'Rams'))
    games.append(('Texans', 'Giants'))
    games.append(('Raiders', 'Chargers'))
    games.append(('Jets', 'Vikings'))
    games.append(('49ers', 'Raiders'))
    games.append(('Giants', 'Lions'))
    games.append(('Vikings', 'Cowboys'))
    games.append(('Steelers', 'Browns'))
    games.append(('Broncos', 'Jets'))
    games.append(('Eagles', 'Falcons'))
    games.append(('Bears', 'Seahawks'))
    games.append(('Patriots', 'Ravens'))
    games.append(('Rams', 'Chargers'))
    games.append(('Redskins', 'Colts'))
    games.append(('Texans', 'Chiefs'))
    games.append(('Packers', 'Dolphins'))
    games.append(('Buccaneers', 'Saints'))
    games.append(('Jaguars', 'Titans'))
    games.append(('Saints', 'Browns'))
    games.append(('Seahawks', 'Cardinals'))
    games.append(('Bears', 'Redskins'))
    games.append(('Chargers', 'Patriots'))
    games.append(('Dolphins', 'Steelers'))
    games.append(('Panthers', '49ers'))
    games.append(('Chiefs', 'Jaguars'))
    games.append(('Buccaneers', 'Rams'))
    games.append(('Packers', 'Vikings'))
    games.append(('Titans', 'Eagles'))
    games.append(('Falcons', 'Bengals'))
    games.append(('Broncos', 'Raiders'))
    games.append(('Bills', 'Ravens'))
    games.append(('Cowboys', 'Giants'))
    games.append(('Raiders', 'Seahawks'))
    games.append(('Jets', 'Packers'))
    games.append(('Saints', 'Steelers'))
    games.append(('Lions', 'Redskins'))
    games.append(('Rams', 'Panthers'))
    games.append(('Chargers', 'Titans'))
    games.append(('Cardinals', 'Buccaneers'))
    games.append(('Cowboys', 'Jaguars'))
    games.append(('Bengals', 'Dolphins'))
    games.append(('49ers', 'Broncos'))
    games.append(('Chiefs', 'Bills'))
    games.append(('Patriots', 'Vikings'))
    games.append(('Colts', 'Texans'))
    games.append(('Packers', 'Cowboys'))
    games.append(('Vikings', 'Cardinals'))
    games.append(('Falcons', 'Buccaneers'))
    games.append(('Ravens', 'Dolphins'))
    games.append(('Texans', 'Chargers'))
    games.append(('Bills', 'Bears'))
    games.append(('Browns', 'Patriots'))
    games.append(('Raiders', 'Chiefs'))
    games.append(('Lions', 'Jets'))
    games.append(('Eagles', 'Colts'))
    games.append(('Panthers', 'Saints'))
    games.append(('Seahawks', 'Giants'))
    games.append(('Bengals', 'Steelers'))
    games.append(('Falcons', 'Ravens'))
    games.append(('Giants', 'Cowboys'))
    games.append(('Bears', 'Vikings'))
    games.append(('Dolphins', 'Titans'))
    games.append(('Buccaneers', 'Panthers'))
    games.append(('Colts', 'Bengals'))
    games.append(('Bills', 'Lions'))
    games.append(('Jaguars', 'Texans'))
    games.append(('49ers', 'Rams'))
    games.append(('Cardinals', 'Seahawks'))
    games.append(('Steelers', 'Patriots'))
    games.append(('Browns', 'Jets'))
    games.append(('Broncos', 'Chiefs'))
    games.append(('Redskins', 'Eagles'))
    games.append(('Dolphins', 'Bears'))
    games.append(('Eagles', 'Giants'))
    games.append(('Jaguars', 'Browns'))
    games.append(('Cowboys', 'Lions'))
    games.append(('Jets', 'Texans'))
    games.append(('Steelers', 'Raiders'))
    games.append(('Chiefs', 'Cardinals'))
    games.append(('Vikings', 'Packers'))
    games.append(('49ers', 'Buccaneers'))
    games.append(('Saints', 'Seahawks'))
    games.append(('Panthers', 'Ravens'))
    games.append(('Bengals', 'Bills'))
    games.append(('Rams', 'Falcons'))
    games.append(('Patriots', 'Colts'))
    games.append(('Titans', 'Redskins'))
    games.append(('Chargers', 'Broncos'))
    games.append(('Jets', 'Bengals'))
    games.append(('Lions', 'Patriots'))
    games.append(('Cowboys', 'Saints'))
    games.append(('Texans', 'Titans'))
    games.append(('Ravens', 'Buccaneers'))
    games.append(('Browns', 'Panthers'))
    games.append(('Redskins', 'Vikings'))
    games.append(('Colts', 'Chargers'))
    games.append(('Falcons', 'Packers'))
    games.append(('Broncos', 'Rams'))
    games.append(('Bears', 'Eagles'))
    games.append(('Raiders', 'Dolphins'))
    games.append(('Giants', 'Jaguars'))
    games.append(('Seahawks', 'Chiefs'))
    games.append(('Bills', 'Steelers'))
    games.append(('Cardinals', '49ers'))
    games.append(('Eagles', 'Texans'))
    games.append(('Vikings', 'Bills'))
    games.append(('Packers', '49ers'))
    games.append(('Dolphins', 'Browns'))
    games.append(('Chiefs', 'Broncos'))
    games.append(('Ravens', 'Steelers'))
    games.append(('Seahawks', 'Panthers'))
    games.append(('Cardinals', 'Rams'))
    games.append(('Bengals', 'Saints'))
    games.append(('Giants', 'Redskins'))
    games.append(('Buccaneers', 'Falcons'))
    games.append(('Chargers', 'Raiders'))
    games.append(('Lions', 'Bears'))
    games.append(('Titans', 'Jaguars'))
    games.append(('Colts', 'Cowboys'))
    games.append(('Patriots', 'Jets'))
    games.append(('Titans', 'Colts'))
    games.append(('Chargers', 'Chiefs'))
    games.append(('Jets', 'Dolphins'))
    games.append(('Bills', 'Browns'))
    games.append(('Lions', 'Packers'))
    games.append(('Steelers', 'Bengals'))
    games.append(('Saints', 'Rams'))
    games.append(('Redskins', 'Buccaneers'))
    games.append(('Cowboys', 'Eagles'))
    games.append(('Cardinals', 'Broncos'))
    games.append(('49ers', 'Seahawks'))
    games.append(('Jaguars', 'Raiders'))
    games.append(('Panthers', 'Falcons'))
    games.append(('Bears', 'Patriots'))
    games.append(('Texans', 'Ravens'))
    games.append(('Vikings', 'Giants'))
    games.append(('Chargers', '49ers'))
    games.append(('Raiders', 'Broncos'))
    games.append(('Panthers', 'Cardinals'))
    games.append(('Bengals', 'Browns'))
    games.append(('Dolphins', 'Bills'))
    games.append(('Steelers', 'Jets'))
    games.append(('Giants', 'Eagles'))
    games.append(('Colts', 'Jaguars'))
    games.append(('Seahawks', 'Falcons'))
    games.append(('Rams', 'Chiefs'))
    games.append(('Ravens', 'Saints'))
    games.append(('Cowboys', 'Redskins'))
    games.append(('Patriots', 'Packers'))
    games.append(('Buccaneers', 'Lions'))
    games.append(('Titans', 'Texans'))
    games.append(('Vikings', 'Bears'))
    games.append(('Steelers', 'Panthers'))
    games.append(('Cardinals', 'Cowboys'))
    games.append(('Buccaneers', 'Seahawks'))
    games.append(('Rams', '49ers'))
    games.append(('Dolphins', 'Lions'))
    games.append(('Browns', 'Ravens'))
    games.append(('Jaguars', 'Redskins'))
    games.append(('Packers', 'Giants'))
    games.append(('Chiefs', 'Titans'))
    games.append(('Bills', 'Patriots'))
    games.append(('Broncos', 'Texans'))
    games.append(('Bengals', 'Chargers'))
    games.append(('Bears', 'Jets'))
    games.append(('Raiders', 'Colts'))
    games.append(('Falcons', 'Saints'))
    games.append(('Eagles', 'Vikings'))
    games.append(('Jets', 'Bills'))
    games.append(('Seahawks', 'Rams'))
    games.append(('Falcons', 'Panthers'))
    games.append(('Packers', 'Bears'))
    games.append(('Eagles', 'Cowboys'))
    games.append(('Redskins', 'Giants'))
    games.append(('Lions', 'Vikings'))
    games.append(('Patriots', 'Dolphins'))
    games.append(('Colts', 'Titans'))
    games.append(('Saints', 'Buccaneers'))
    games.append(('49ers', 'Cardinals'))
    games.append(('Ravens', 'Bengals'))
    games.append(('Broncos', 'Chargers'))
    games.append(('Chiefs', 'Raiders'))
    games.append(('Texans', 'Jaguars'))
    games.append(('Browns', 'Steelers'))
    return games


def monte_carlo(teams, trials=1000000):
    all_trials = list()
    # 1,000,000 Trials
    for trial in range(trials):
        # Get just the name, record and elo of each team
        pseudo_teams = [(team[0], team[1], team[2], team[3], team[4]) for team in teams]

        # For each game in the list of games yet to be played
        for game in get_2019_schedule()[len(completed_games):]:
            # Get the home and away teams
            home = game[0]
            away = game[1]

            home = get_team(pseudo_teams, home)
            away = get_team(pseudo_teams, away)

            # Get the home teams chance of victory and simulate the outcome
            chance = get_pct_chance(home[4], away[4])
            monte = random.random()
            home_victory = monte < chance
            draw = monte == chance

            # Update the teams records based on the simulated outcome
            home_wins = home[1] + 1 if home_victory else home[1]
            home_losses = home[2] + 1 if not home_victory else home[2]
            away_wins = away[1] + 1 if not home_victory else away[1]
            away_losses = away[2] + 1 if home_victory else away[2]
            home_ties = home[3] + 1 if draw else home[3]
            away_ties = away[3] + 1 if draw else away[3]

            # Update the teams elo based on the simulated outcome
            home_elo, away_elo = NFL.get_new_elos(home[4], away[4], home_victory, False, 42)

            # Create an updated team
            new_home = (home[0], home_wins, home_losses, home_ties, home_elo)
            new_away = (away[0], away_wins, away_losses, away_ties, away_elo)

            # Update the pseudo teams with the new teams
            pseudo_teams = [new_home if team == home else team for team in pseudo_teams]
            pseudo_teams = [new_away if team == away else team for team in pseudo_teams]

        # Add all the pseudo teams in the trial to a list
        all_trials.append(pseudo_teams)

    averaged_teams = list()

    # Get a list of team names
    team_names = [team[0] for team in teams]

    # For each team name in the list
    for team_name in team_names:
        team_trials = list()

        # For each trial
        for trial in all_trials:
            # Get the pseudo team
            team = get_team(trial, team_name)

            # Add it to a list of all pseudo teams with that name over all trials
            team_trials.append(team)

        # Get a list of the pseudo teams wins for each trial
        wins = [team[1] for team in team_trials]

        # Get a list of the pseudo teams losses for each trial
        losses = [team[2] for team in team_trials]

        # Get a list of the pseudo teams elos for each trial
        elos = [team[4] for team in team_trials]

        # Create a new pseudo team with the average of each stat
        averaged_team = (team_name, sum(wins) / len(wins), sum(losses) / len(losses), 0, sum(elos) / len(elos))

        # Add it to a final list
        averaged_teams.append(averaged_team)

    # Print the standings
    Standings.print_monte_carlo_simulation(averaged_teams)


def get_pct_chance(home_elo, away_elo):
    q_home = 10 ** (home_elo / 400)
    q_away = 10 ** (away_elo / 400)

    e_home = q_home / (q_home + q_away)

    return e_home


def get_playoff_picture(teams):
    teams = sort_by_tiebreakers(teams)
    league = get_league_structure()
    first_round_byes = list()
    division_leaders = list()
    wild_cards = list()

    for conf_name, conference in league.items():
        conference_teams = list()

        for div_name, division in conference.items():
            division_teams = [get_team(teams, team_name) for team_name in division]
            division_teams = sort_by_tiebreakers(division_teams)
            division_leaders.append(division_teams[0])
            conference_teams.extend(division_teams)

        conference_teams = sort_by_tiebreakers(conference_teams)
        first_round_byes.append(conference_teams[0])

        conf_leader = conference_teams[0][0]
        conf_leaders_division = None
        for div_name, division in conference.items():
            if conf_leader in division:
                conf_leaders_division = division
        for runner_up in range(1, 5):
            runner_up_name = conference_teams[runner_up][0]
            runner_up_division = None
            for div_name, division in conference.items():
                if runner_up_name in division:
                    runner_up_division = division
            if runner_up_division != conf_leaders_division:
                first_round_byes.append(conference_teams[runner_up])
                break

        other_teams = conference_teams.copy()
        other_teams = list(set(other_teams) - set(division_leaders))
        other_teams = sort_by_tiebreakers(other_teams)
        wild_cards.append(other_teams[0])
        wild_cards.append(other_teams[1])

    print('First Round Byes:')
    first_round_byes = [team[0] for team in first_round_byes]
    print('\n'.join(first_round_byes))
    print()

    print('Division Leaders:')
    division_leaders = [team[0] for team in division_leaders]
    print('\n'.join(division_leaders))
    print()

    print('Wild Cards:')
    wild_cards = [team[0] for team in wild_cards]
    print('\n'.join(wild_cards))
    print()


def sort_by_tiebreakers(teams):
    sorted_teams = sorted(teams, key=cmp_to_key(compare_win_pct), reverse=True)
    return sorted_teams


def compare_win_pct(team1, team2):
    team1_games_played = team1[1] + team1[2] + team1[3]
    team2_games_played = team2[1] + team2[2] + team2[3]
    team1_pct = team1[1] / team1_games_played if team1_games_played > 0 else 0
    team2_pct = team2[1] / team2_games_played if team2_games_played > 0 else 0

    if team1_pct - team2_pct == 0:
        return compare_head_to_head(team1, team2)
    return team1_pct - team2_pct


def compare_head_to_head(team1, team2):
    head_to_head_games = list(filter(lambda game: contains_both_teams(team1, team2, game), completed_games))

    team1_victories = list(filter(lambda game: filter_team_victories(team1, game), head_to_head_games))
    team2_victories = list(filter(lambda game: filter_team_victories(team2, game), head_to_head_games))

    if len(team1_victories) - len(team2_victories) == 0:
        return compare_divisional_record(team1, team2)
    return len(team1_victories) - len(team2_victories)


def compare_divisional_record(team1, team2):
    team1_divisional_games = list(filter(lambda game: contains_divisional_teams(team1, game), completed_games))
    team2_divisional_games = list(filter(lambda game: contains_divisional_teams(team2, game), completed_games))

    team1_victories = list(filter(lambda game: filter_team_victories(team1, game), team1_divisional_games))
    team2_victories = list(filter(lambda game: filter_team_victories(team2, game), team2_divisional_games))

    if len(team1_victories) - len(team2_victories) == 0:
        return compare_common_record(team1, team2)
    return len(team1_victories) - len(team2_victories)


def compare_common_record(team1, team2):
    team1_common_games = list(filter(lambda game: contains_common_opponents(team1, team2, game), completed_games))
    team2_common_games = list(filter(lambda game: contains_common_opponents(team2, team1, game), completed_games))

    team1_victories = list(filter(lambda game: filter_team_victories(team1, game), team1_common_games))
    team2_victories = list(filter(lambda game: filter_team_victories(team2, game), team2_common_games))

    if len(team1_victories) - len(team2_victories) == 0:
        return compare_conference_record(team1, team2)
    return len(team1_victories) - len(team2_victories)


def compare_conference_record(team1, team2):
    team1_conference_games = list(filter(lambda game: contains_conference_teams(team1, game), completed_games))
    team2_conference_games = list(filter(lambda game: contains_conference_teams(team2, game), completed_games))

    team1_victories = list(filter(lambda game: filter_team_victories(team1, game), team1_conference_games))
    team2_victories = list(filter(lambda game: filter_team_victories(team2, game), team2_conference_games))

    if len(team1_victories) - len(team2_victories) == 0:
        return compare_strength_of_victory(team1, team2)
    return len(team1_victories) - len(team2_victories)


def compare_strength_of_victory(team1, team2):
    team1_games = list(filter(lambda g: contains_team(team1, g), completed_games))
    team2_games = list(filter(lambda g: contains_team(team2, g), completed_games))

    team1_victories = list(filter(lambda g: filter_team_victories(team1, g), team1_games))
    team2_victories = list(filter(lambda g: filter_team_victories(team2, g), team2_games))

    team1_opponent_victories = list()
    team2_opponent_victories = list()
    for game in team1_victories:
        if game[0] == team1[0]:
            opponent = game[2]
        else:
            opponent = game[0]
        opponent_games = list(filter(lambda g: contains_team_name(opponent, g), completed_games))
        opponent_victories = list(filter(lambda g: filter_team_victories([opponent], g), opponent_games))
        team1_opponent_victories.append(len(opponent_victories))

    for game in team2_victories:
        if game[0] == team2[0]:
            opponent = game[2]
        else:
            opponent = game[0]
        opponent_games = list(filter(lambda g: contains_team_name(opponent, g), completed_games))
        opponent_victories = list(filter(lambda g: filter_team_victories([opponent], g), opponent_games))
        team2_opponent_victories.append(len(opponent_victories))

    if sum(team1_opponent_victories) - sum(team2_opponent_victories) == 0:
        return compare_strength_of_schedule(team1, team2)
    return sum(team1_opponent_victories) - sum(team2_opponent_victories)


def compare_strength_of_schedule(team1, team2):
    team1_games = list(filter(lambda g: contains_team(team1, g), completed_games))
    team2_games = list(filter(lambda g: contains_team(team2, g), completed_games))

    team1_opponent_victories = list()
    team2_opponent_victories = list()
    for game in team1_games:
        if game[0] == team1[0]:
            opponent = game[2]
        else:
            opponent = game[0]
        opponent_games = list(filter(lambda g: contains_team_name(opponent, g), completed_games))
        opponent_victories = list(filter(lambda g: filter_team_victories([opponent], g), opponent_games))
        team1_opponent_victories.append(len(opponent_victories))

    for game in team2_games:
        if game[0] == team2[0]:
            opponent = game[2]
        else:
            opponent = game[0]
        opponent_games = list(filter(lambda g: contains_team_name(opponent, g), completed_games))
        opponent_victories = list(filter(lambda g: filter_team_victories([opponent], g), opponent_games))
        team2_opponent_victories.append(len(opponent_victories))

    if sum(team1_opponent_victories) - sum(team2_opponent_victories) == 0:
        return compare_point_diff(team1, team2)
    return sum(team1_opponent_victories) - sum(team2_opponent_victories)


def compare_point_diff(team1, team2):
    team1_point_diff = team1[5] - team1[6]
    team2_point_diff = team2[5] - team2[6]

    return team1_point_diff - team2_point_diff


def filter_team_victories(team, game):
    if game[0] == team[0]:
        return game[1] > game[3]
    else:
        return game[1] < game[3]


def contains_team(team, game):
    return game[0] == team[0] or game[2] == team[0]


def contains_team_name(team, game):
    return game[0] == team or game[2] == team


def contains_both_teams(team1, team2, game):
    return contains_team(team1, game) and contains_team(team2, game)


def contains_divisional_teams(team, game):
    name = team[0]
    league = get_league_structure()
    teams_division = None
    for conf_name, conference in league.items():
        for div_name, division in conference.items():
            if name in division:
                teams_division = division

    return (game[0] == name and game[2] in teams_division) or (game[0] in teams_division and game[2] == name)


def contains_conference_teams(team, game):
    name = team[0]
    league = get_league_structure()
    teams_conference = None
    for conf_name, conference in league.items():
        for div_name, division in conference.items():
            if name in division:
                teams_conference = conference

    conf_teams = list()
    for div_name, division in teams_conference.items():
        conf_teams.extend(division)

    return (game[0] == name and game[2] in conf_teams) or (game[0] in conf_teams and game[2] == name)


def contains_common_opponents(team1, team2, game):
    team1_name = team1[0]
    team2_name = team2[0]

    home_team = game[0]
    away_team = game[2]

    team1_opponents = set()
    team2_opponents = set()

    for completed_game in completed_games:
        completed_home_team = completed_game[0]
        completed_away_team = completed_game[2]
        if completed_home_team == team1_name:
            team1_opponents.add(completed_away_team)
        if completed_away_team == team1_name:
            team1_opponents.add(completed_home_team)

        if completed_home_team == team2_name:
            team2_opponents.add(completed_away_team)
        if completed_away_team == team2_name:
            team2_opponents.add(completed_home_team)

    common = team1_opponents.intersection(team2_opponents)

    return (home_team == team1_name and away_team in common) or (home_team in common and away_team == team1_name)


def get_schedule_difficulty(teams, team_name):
    teams_schedule = list(filter(lambda g: g[0] == team_name or g[1] == team_name, get_2019_schedule()))
    opponent_elos = list()
    for game in teams_schedule:
        opponent = None
        if game[0] == team_name:
            opponent = get_team(teams, game[1])
        elif game[1] == team_name:
            opponent = get_team(teams, game[0])
        opponent_elos.append(opponent[4])

    return statistics.mean(opponent_elos)


def get_remaining_schedule_difficulty(teams, team_name):
    teams_schedule = list(filter(lambda g: g[0] == team_name or g[1] == team_name,
                                 get_2019_schedule()[len(completed_games):]))
    opponent_elos = list()
    for game in teams_schedule:
        opponent = None
        if game[0] == team_name:
            opponent = get_team(teams, game[1])
        elif game[1] == team_name:
            opponent = get_team(teams, game[0])
        opponent_elos.append(opponent[4])

    return statistics.mean(opponent_elos)
