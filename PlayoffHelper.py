import itertools
import random

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


def get_fake_schedule(teams):
    games = list()

    structure = get_league_structure()
    afc = structure.get('AFC')
    afc_divisions = list()
    for afc_div_name, afc_division in afc.items():
        afc_divisions.append(afc_division)
    afc_teams = [team for afc_division in afc_divisions for team in afc_division]
    for game in itertools.combinations(afc_teams, 2):
        games.append(game)

    nfc = structure.get('NFC')
    nfc_divisions = list()
    for nfc_div_name, nfc_division in nfc.items():
        nfc_divisions.append(nfc_division)
    nfc_teams = [team for nfc_division in nfc_divisions for team in nfc_division]
    for game in itertools.combinations(nfc_teams, 2):
        games.append(game)

    team_names = [team[0] for team in teams]
    for team_name in team_names:
        other_teams = team_names.copy()
        other_teams.remove(team_name)
        games.append((team_name, random.choice(other_teams)))

    return games


def monte_carlo(teams):
    all_trials = list()
    # 100,000 Trials
    for trial in range(100):
        # Get just the name, record and elo of each team
        pseudo_teams = [(team[0], team[1], team[2], team[3], team[4]) for team in teams]

        # For each game in the list of games
        for game in get_fake_schedule(teams):
            # Get the home and away teams
            home = game[0]
            away = game[1]

            home = get_team(pseudo_teams, home)
            away = get_team(pseudo_teams, away)

            # Get the home teams chance of victory and simulate the outcome
            chance = get_pct_chance(home[4], away[4])
            monte = random.random()
            home_victory = monte <= chance

            # Update the teams records based on the simulated outcome
            home_wins = home[1] + 1 if home_victory else home[1]
            home_losses = home[2] + 1 if not home_victory else home[2]
            away_wins = away[1] + 1 if not home_victory else away[1]
            away_losses = away[2] + 1 if home_victory else away[2]

            # Update the teams elo based on the simulated outcome
            home_elo, away_elo = NFL.get_new_elos(home[4], away[4], home_victory, False, 42)

            # Create an updated team
            new_home = (home[0], home_wins, home_losses, home[3], home_elo)
            new_away = (away[0], away_wins, away_losses, away[3], away_elo)

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


def get_divisional_leaders(teams):
    league = get_league_structure()
    afc = league.get('AFC')
    for div_name, division in afc.items():
        teams = [get_team(teams, name) for name in division]
        sorted_by_wins = sorted(teams, reverse=True, key=lambda team: team[1])


def get_divisional_tiebreaker(team1, team2):
    if team1[1] > team2[1]:
        return team1
    elif team1[1] < team2[1]:
        return team2
    else:
        return get_head_to_head(team1, team2)


def get_head_to_head(team1, team2):
    def contains_both_teams(game):
        return game[0] == team1[0] and game[2] == team2[0] or game[0] == team2[0] and game[2] == team1[0]

    head_to_head_games = list(filter(lambda g: contains_both_teams(g), completed_games))

    def get_team_victories(team, game):
        if game[0] == team[0]:
            return game[1] > game[3]
        else:
            return game[1] < game[3]

    team1_victories = list(filter(lambda g: get_team_victories(team1, g), head_to_head_games))
    team2_victories = list(filter(lambda g: get_team_victories(team2, g), head_to_head_games))

    if team1_victories > team2_victories:
        return team1
    elif team1_victories < team2_victories:
        return team2
    else:
        return get_divisional_record(team1, team2)


def get_divisional_record(team1, team2):
    def contains_divisional_teams(team, game):
        def get_division(team):
            name = team[0]
            league = get_league_structure()
            for conf_name, conference in league.items():
                for div_name, division in conference.items():
                    if name in division:
                        return division

        return game[0] == team and game[2] in get_division(team) or game[0] in get_division(team) and game[2] == team

    team1_divisional_games = list(filter(lambda g: contains_divisional_teams(team1, g), completed_games))
    team2_divisional_games = list(filter(lambda g: contains_divisional_teams(team2, g), completed_games))

    def get_team_victories(team, game):
        if game[0] == team[0]:
            return game[1] > game[3]
        else:
            return game[1] < game[3]

    team1_victories = list(filter(lambda g: get_team_victories(team1, g), team1_divisional_games))
    team2_victories = list(filter(lambda g: get_team_victories(team2, g), team2_divisional_games))

    if team1_victories > team2_victories:
        return team1
    elif team1_victories < team2_victories:
        return team2
    else:
        return get_divisional_record(team1, team2)
