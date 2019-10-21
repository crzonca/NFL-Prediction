import random
import statistics
from functools import cmp_to_key

import pandas as pd

import Projects.nfl.NFL_Prediction.Core.NFLDataGroomer as NFL
import Projects.nfl.NFL_Prediction.OddsHelper as Odds
import Projects.nfl.NFL_Prediction.StandingsHelper as Standings

completed_games = pd.DataFrame(columns=['home_team', 'away_team', 'home_score', 'away_score', 'week', 'home_spread',
                                        'home_pass_completions', 'home_pass_attempts', 'home_passing_touchdowns',
                                        'home_interceptions_thrown', 'home_net_passing_yards', 'home_total_yards',
                                        'home_elo',
                                        'away_pass_completions', 'away_pass_attempts', 'away_passing_touchdowns',
                                        'away_interceptions_thrown', 'away_net_passing_yards', 'away_total_yards',
                                        'away_elo'])


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


def get_league_structure():
    """
    Gets a dictionary containing the division and conference structure of the league.

    :return: The league structure
    """

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


def get_schedule():
    """
    Gets the 2019 NFL schedule.

    :return: The 2019 NFL schedule
    """

    games = list()
    games.extend(get_week1_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week2_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week3_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week4_schedule(week_end_date=None, get_odds=False))

    games.extend(get_week5_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week6_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week7_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week8_schedule(week_end_date=None, get_odds=False))

    games.extend(get_week9_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week10_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week11_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week12_schedule(week_end_date=None, get_odds=False))

    games.extend(get_week13_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week14_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week15_schedule(week_end_date=None, get_odds=False))
    games.extend(get_week16_schedule(week_end_date=None, get_odds=False))

    games.extend(get_week17_schedule(week_end_date=None, get_odds=False))
    # games.extend(get_wildcard_schedule(week_end_date=None, get_odds=False))
    # games.extend(get_divisional_schedule(week_end_date=None, get_odds=False))
    # games.extend(get_conference_schedule(week_end_date=None, get_odds=False))
    # games.extend(get_superbowl_schedule(week_end_date=None, get_odds=False))
    return games


def create_match_up(home_name, away_name, home_spread=0.0, odds=None, neutral_location=False):
    """
    Creates a representation of an instance of an NFL game.

    :param home_name: The name of the home team
    :param away_name: The name of the away team
    :param home_spread: The spread of the game from the home teams perspective
    :param odds: The list of odds information for upcoming games
    :param neutral_location: If the game is at a neutral location
    :return: The match up
    """

    if odds:
        odds = list(filter(lambda g: any(home_name in name for name in g[0]) and
                                     any(away_name in name for name in g[0]), odds))
        if len(odds) < 1:
            print(away_name + ' at ' + home_name + ' not found')
        elif len(odds) > 1:
            print('Multiple instances of ' + away_name + ' at ' + home_name + ' found')
        else:
            game = odds[0]
            home_spread = game[2]

    return (home_name, away_name), home_spread, neutral_location


def get_week1_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Bears', 'Packers', odds=odds))
    games.append(create_match_up('Panthers', 'Rams', odds=odds))
    games.append(create_match_up('Eagles', 'Redskins', odds=odds))
    games.append(create_match_up('Jets', 'Bills', odds=odds))
    games.append(create_match_up('Vikings', 'Falcons', odds=odds))
    games.append(create_match_up('Dolphins', 'Ravens', odds=odds))
    games.append(create_match_up('Jaguars', 'Chiefs', odds=odds))
    games.append(create_match_up('Browns', 'Titans', odds=odds))
    games.append(create_match_up('Chargers', 'Colts', odds=odds))
    games.append(create_match_up('Seahawks', 'Bengals', odds=odds))
    games.append(create_match_up('Buccaneers', '49ers', odds=odds))
    games.append(create_match_up('Cowboys', 'Giants', odds=odds))
    games.append(create_match_up('Cardinals', 'Lions', odds=odds))
    games.append(create_match_up('Patriots', 'Steelers', odds=odds))
    games.append(create_match_up('Saints', 'Texans', odds=odds))
    games.append(create_match_up('Raiders', 'Broncos', odds=odds))

    return games


def get_week2_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Panthers', 'Buccaneers', odds=odds))
    games.append(create_match_up('Ravens', 'Cardinals', odds=odds))
    games.append(create_match_up('Redskins', 'Cowboys', odds=odds))
    games.append(create_match_up('Titans', 'Colts', odds=odds))
    games.append(create_match_up('Steelers', 'Seahawks', odds=odds))
    games.append(create_match_up('Giants', 'Bills', odds=odds))
    games.append(create_match_up('Bengals', '49ers', odds=odds))
    games.append(create_match_up('Lions', 'Chargers', odds=odds))
    games.append(create_match_up('Packers', 'Vikings', odds=odds))
    games.append(create_match_up('Texans', 'Jaguars', odds=odds))
    games.append(create_match_up('Dolphins', 'Patriots', odds=odds))
    games.append(create_match_up('Raiders', 'Chiefs', odds=odds))
    games.append(create_match_up('Rams', 'Saints', odds=odds))
    games.append(create_match_up('Broncos', 'Bears', odds=odds))
    games.append(create_match_up('Falcons', 'Eagles', odds=odds))
    games.append(create_match_up('Jets', 'Browns', odds=odds))

    return games


def get_week3_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Jaguars', 'Titans', odds=odds))
    games.append(create_match_up('Bills', 'Bengals', odds=odds))
    games.append(create_match_up('Eagles', 'Lions', odds=odds))
    games.append(create_match_up('Patriots', 'Jets', odds=odds))
    games.append(create_match_up('Vikings', 'Raiders', odds=odds))
    games.append(create_match_up('Chiefs', 'Ravens', odds=odds))
    games.append(create_match_up('Colts', 'Falcons', odds=odds))
    games.append(create_match_up('Packers', 'Broncos', odds=odds))
    games.append(create_match_up('Cowboys', 'Dolphins', odds=odds))
    games.append(create_match_up('Buccaneers', 'Giants', odds=odds))
    games.append(create_match_up('Cardinals', 'Panthers', odds=odds))
    games.append(create_match_up('49ers', 'Steelers', odds=odds))
    games.append(create_match_up('Seahawks', 'Saints', odds=odds))
    games.append(create_match_up('Chargers', 'Texans', odds=odds))
    games.append(create_match_up('Browns', 'Rams', odds=odds))
    games.append(create_match_up('Redskins', 'Bears', odds=odds))

    return games


def get_week4_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Packers', 'Eagles', odds=odds))
    games.append(create_match_up('Falcons', 'Titans', odds=odds))
    games.append(create_match_up('Giants', 'Redskins', odds=odds))
    games.append(create_match_up('Dolphins', 'Chargers', odds=odds))
    games.append(create_match_up('Colts', 'Raiders', odds=odds))
    games.append(create_match_up('Texans', 'Panthers', odds=odds))
    games.append(create_match_up('Lions', 'Chiefs', odds=odds))
    games.append(create_match_up('Ravens', 'Browns', odds=odds))
    games.append(create_match_up('Bills', 'Patriots', odds=odds))
    games.append(create_match_up('Rams', 'Buccaneers', odds=odds))
    games.append(create_match_up('Cardinals', 'Seahawks', odds=odds))
    games.append(create_match_up('Bears', 'Vikings', odds=odds))
    games.append(create_match_up('Broncos', 'Jaguars', odds=odds))
    games.append(create_match_up('Saints', 'Cowboys', odds=odds))
    games.append(create_match_up('Steelers', 'Bengals', odds=odds))

    return games


def get_week5_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Seahawks', 'Rams', odds=odds))
    games.append(create_match_up('Panthers', 'Jaguars', odds=odds))
    games.append(create_match_up('Redskins', 'Patriots', odds=odds))
    games.append(create_match_up('Titans', 'Bills', odds=odds))
    games.append(create_match_up('Steelers', 'Ravens', odds=odds))
    games.append(create_match_up('Bengals', 'Cardinals', odds=odds))
    games.append(create_match_up('Texans', 'Falcons', odds=odds))
    games.append(create_match_up('Saints', 'Buccaneers', odds=odds))
    games.append(create_match_up('Giants', 'Vikings', odds=odds))
    games.append(create_match_up('Raiders', 'Bears', odds=odds, neutral_location=True))
    games.append(create_match_up('Eagles', 'Jets', odds=odds))
    games.append(create_match_up('Chargers', 'Broncos', odds=odds))
    games.append(create_match_up('Cowboys', 'Packers', odds=odds))
    games.append(create_match_up('Chiefs', 'Colts', odds=odds))
    games.append(create_match_up('49ers', 'Browns', odds=odds))

    return games


def get_week6_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Patriots', 'Giants', odds=odds))
    games.append(create_match_up('Buccaneers', 'Panthers', odds=odds, neutral_location=True))
    games.append(create_match_up('Dolphins', 'Redskins', odds=odds))
    games.append(create_match_up('Vikings', 'Eagles', odds=odds))
    games.append(create_match_up('Chiefs', 'Texans', odds=odds))
    games.append(create_match_up('Jaguars', 'Saints', odds=odds))
    games.append(create_match_up('Browns', 'Seahawks', odds=odds))
    games.append(create_match_up('Ravens', 'Bengals', odds=odds))
    games.append(create_match_up('Rams', '49ers', odds=odds))
    games.append(create_match_up('Cardinals', 'Falcons', odds=odds))
    games.append(create_match_up('Jets', 'Cowboys', odds=odds))
    games.append(create_match_up('Broncos', 'Titans', odds=odds))
    games.append(create_match_up('Chargers', 'Steelers', odds=odds))
    games.append(create_match_up('Packers', 'Lions', odds=odds))

    return games


def get_week7_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Broncos', 'Chiefs', odds=odds))
    games.append(create_match_up('Bills', 'Dolphins', odds=odds))
    games.append(create_match_up('Bengals', 'Jaguars', odds=odds))
    games.append(create_match_up('Lions', 'Vikings', odds=odds))
    games.append(create_match_up('Packers', 'Raiders', odds=odds))
    games.append(create_match_up('Falcons', 'Rams', odds=odds))
    games.append(create_match_up('Colts', 'Texans', odds=odds))
    games.append(create_match_up('Redskins', '49ers', odds=odds))
    games.append(create_match_up('Giants', 'Cardinals', odds=odds))
    games.append(create_match_up('Titans', 'Chargers', odds=odds))
    games.append(create_match_up('Bears', 'Saints', odds=odds))
    games.append(create_match_up('Seahawks', 'Ravens', odds=odds))
    games.append(create_match_up('Cowboys', 'Eagles', odds=odds))
    games.append(create_match_up('Jets', 'Patriots', odds=odds))

    return games


def get_week8_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Vikings', 'Redskins', odds=odds))
    games.append(create_match_up('Falcons', 'Seahawks', odds=odds))
    games.append(create_match_up('Titans', 'Buccaneers', odds=odds))
    games.append(create_match_up('Saints', 'Cardinals', odds=odds))
    games.append(create_match_up('Rams', 'Bengals', odds=odds, neutral_location=True))
    games.append(create_match_up('Jaguars', 'Jets', odds=odds))
    games.append(create_match_up('Bills', 'Eagles', odds=odds))
    games.append(create_match_up('Bears', 'Chargers', odds=odds))
    games.append(create_match_up('Lions', 'Giants', odds=odds))
    games.append(create_match_up('Texans', 'Raiders', odds=odds))
    games.append(create_match_up('49ers', 'Panthers', odds=odds))
    games.append(create_match_up('Patriots', 'Browns', odds=odds))
    games.append(create_match_up('Colts', 'Broncos', odds=odds))
    games.append(create_match_up('Chiefs', 'Packers', odds=odds))
    games.append(create_match_up('Steelers', 'Dolphins', odds=odds))

    return games


def get_week9_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Cardinals', '49ers', odds=odds))
    games.append(create_match_up('Jaguars', 'Texans', odds=odds, neutral_location=True))
    games.append(create_match_up('Eagles', 'Bears', odds=odds))
    games.append(create_match_up('Steelers', 'Colts', odds=odds))
    games.append(create_match_up('Dolphins', 'Jets', odds=odds))
    games.append(create_match_up('Chiefs', 'Vikings', odds=odds))
    games.append(create_match_up('Panthers', 'Titans', odds=odds))
    games.append(create_match_up('Bills', 'Redskins', odds=odds))
    games.append(create_match_up('Seahawks', 'Buccaneers', odds=odds))
    games.append(create_match_up('Raiders', 'Lions', odds=odds))
    games.append(create_match_up('Chargers', 'Packers', odds=odds))
    games.append(create_match_up('Broncos', 'Browns', odds=odds))
    games.append(create_match_up('Ravens', 'Patriots', odds=odds))
    games.append(create_match_up('Giants', 'Cowboys', odds=odds))

    return games


def get_week10_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Raiders', 'Chargers', odds=odds))
    games.append(create_match_up('Bengals', 'Ravens', odds=odds))
    games.append(create_match_up('Browns', 'Bills', odds=odds))
    games.append(create_match_up('Packers', 'Panthers', odds=odds))
    games.append(create_match_up('Saints', 'Falcons', odds=odds))
    games.append(create_match_up('Bears', 'Lions', odds=odds))
    games.append(create_match_up('Jets', 'Giants', odds=odds, neutral_location=True))
    games.append(create_match_up('Titans', 'Chiefs', odds=odds))
    games.append(create_match_up('Buccaneers', 'Cardinals', odds=odds))
    games.append(create_match_up('Colts', 'Dolphins', odds=odds))
    games.append(create_match_up('Steelers', 'Rams', odds=odds))
    games.append(create_match_up('Cowboys', 'Vikings', odds=odds))
    games.append(create_match_up('49ers', 'Seahawks', odds=odds))

    return games


def get_week11_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Browns', 'Steelers', odds=odds))
    games.append(create_match_up('Panthers', 'Falcons', odds=odds))
    games.append(create_match_up('Lions', 'Cowboys', odds=odds))
    games.append(create_match_up('Colts', 'Jaguars', odds=odds))
    games.append(create_match_up('Dolphins', 'Bills', odds=odds))
    games.append(create_match_up('Ravens', 'Texans', odds=odds))
    games.append(create_match_up('Vikings', 'Broncos', odds=odds))
    games.append(create_match_up('Redskins', 'Jets', odds=odds))
    games.append(create_match_up('Buccaneers', 'Saints', odds=odds))
    games.append(create_match_up('49ers', 'Cardinals', odds=odds))
    games.append(create_match_up('Raiders', 'Bengals', odds=odds))
    games.append(create_match_up('Eagles', 'Patriots', odds=odds))
    games.append(create_match_up('Rams', 'Bears', odds=odds))
    games.append(create_match_up('Chargers', 'Chiefs', odds=odds, neutral_location=True))

    return games


def get_week12_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Texans', 'Colts', odds=odds))
    games.append(create_match_up('Bills', 'Broncos', odds=odds))
    games.append(create_match_up('Bears', 'Giants', odds=odds))
    games.append(create_match_up('Bengals', 'Steelers', odds=odds))
    games.append(create_match_up('Browns', 'Dolphins', odds=odds))
    games.append(create_match_up('Falcons', 'Buccaneers', odds=odds))
    games.append(create_match_up('Saints', 'Panthers', odds=odds))
    games.append(create_match_up('Redskins', 'Lions', odds=odds))
    games.append(create_match_up('Jets', 'Raiders', odds=odds))
    games.append(create_match_up('Titans', 'Jaguars', odds=odds))
    games.append(create_match_up('Patriots', 'Cowboys', odds=odds))
    games.append(create_match_up('49ers', 'Packers', odds=odds))
    games.append(create_match_up('Eagles', 'Seahawks', odds=odds))
    games.append(create_match_up('Rams', 'Ravens', odds=odds))

    return games


def get_week13_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Lions', 'Bears', odds=odds))
    games.append(create_match_up('Cowboys', 'Bills', odds=odds))
    games.append(create_match_up('Falcons', 'Saints', odds=odds))
    games.append(create_match_up('Colts', 'Titans', odds=odds))
    games.append(create_match_up('Bengals', 'Jets', odds=odds))
    games.append(create_match_up('Panthers', 'Redskins', odds=odds))
    games.append(create_match_up('Ravens', '49ers', odds=odds))
    games.append(create_match_up('Jaguars', 'Buccaneers', odds=odds))
    games.append(create_match_up('Giants', 'Packers', odds=odds))
    games.append(create_match_up('Dolphins', 'Eagles', odds=odds))
    games.append(create_match_up('Chiefs', 'Raiders', odds=odds))
    games.append(create_match_up('Cardinals', 'Rams', odds=odds))
    games.append(create_match_up('Steelers', 'Browns', odds=odds))
    games.append(create_match_up('Broncos', 'Chargers', odds=odds))
    games.append(create_match_up('Texans', 'Patriots', odds=odds))
    games.append(create_match_up('Seahawks', 'Vikings', odds=odds))

    return games


def get_week14_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Bears', 'Cowboys', odds=odds))
    games.append(create_match_up('Falcons', 'Panthers', odds=odds))
    games.append(create_match_up('Buccaneers', 'Colts', odds=odds))
    games.append(create_match_up('Jets', 'Dolphins', odds=odds))
    games.append(create_match_up('Saints', '49ers', odds=odds))
    games.append(create_match_up('Vikings', 'Lions', odds=odds))
    games.append(create_match_up('Texans', 'Broncos', odds=odds))
    games.append(create_match_up('Bills', 'Ravens', odds=odds))
    games.append(create_match_up('Browns', 'Bengals', odds=odds))
    games.append(create_match_up('Packers', 'Redskins', odds=odds))
    games.append(create_match_up('Jaguars', 'Chargers', odds=odds))
    games.append(create_match_up('Cardinals', 'Steelers', odds=odds))
    games.append(create_match_up('Raiders', 'Titans', odds=odds))
    games.append(create_match_up('Patriots', 'Chiefs', odds=odds))
    games.append(create_match_up('Rams', 'Seahawks', odds=odds))
    games.append(create_match_up('Eagles', 'Giants', odds=odds))

    return games


def get_week15_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Ravens', 'Jets', odds=odds))
    games.append(create_match_up('Panthers', 'Seahawks', odds=odds))
    games.append(create_match_up('Redskins', 'Eagles', odds=odds))
    games.append(create_match_up('Titans', 'Texans', odds=odds))
    games.append(create_match_up('Steelers', 'Bills', odds=odds))
    games.append(create_match_up('Giants', 'Dolphins', odds=odds))
    games.append(create_match_up('Chiefs', 'Broncos', odds=odds))
    games.append(create_match_up('Bengals', 'Patriots', odds=odds))
    games.append(create_match_up('Lions', 'Buccaneers', odds=odds))
    games.append(create_match_up('Packers', 'Bears', odds=odds))
    games.append(create_match_up('Raiders', 'Jaguars', odds=odds))
    games.append(create_match_up('Cardinals', 'Browns', odds=odds))
    games.append(create_match_up('49ers', 'Falcons', odds=odds))
    games.append(create_match_up('Cowboys', 'Rams', odds=odds))
    games.append(create_match_up('Chargers', 'Vikings', odds=odds))
    games.append(create_match_up('Saints', 'Colts', odds=odds))

    return games


def get_week16_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Broncos', 'Lions', odds=odds))
    games.append(create_match_up('Chargers', 'Raiders', odds=odds))
    games.append(create_match_up('Redskins', 'Giants', odds=odds))
    games.append(create_match_up('Titans', 'Saints', odds=odds))
    games.append(create_match_up('Jets', 'Steelers', odds=odds))
    games.append(create_match_up('Patriots', 'Bills', odds=odds))
    games.append(create_match_up('49ers', 'Rams', odds=odds))
    games.append(create_match_up('Buccaneers', 'Texans', odds=odds))
    games.append(create_match_up('Falcons', 'Jaguars', odds=odds))
    games.append(create_match_up('Browns', 'Ravens', odds=odds))
    games.append(create_match_up('Colts', 'Panthers', odds=odds))
    games.append(create_match_up('Dolphins', 'Bengals', odds=odds))
    games.append(create_match_up('Seahawks', 'Cardinals', odds=odds))
    games.append(create_match_up('Eagles', 'Cowboys', odds=odds))
    games.append(create_match_up('Bears', 'Chiefs', odds=odds))
    games.append(create_match_up('Vikings', 'Packers', odds=odds))

    return games


def get_week17_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Ravens', 'Steelers', odds=odds))
    games.append(create_match_up('Bills', 'Jets', odds=odds))
    games.append(create_match_up('Buccaneers', 'Falcons', odds=odds))
    games.append(create_match_up('Giants', 'Eagles', odds=odds))
    games.append(create_match_up('Panthers', 'Saints', odds=odds))
    games.append(create_match_up('Bengals', 'Browns', odds=odds))
    games.append(create_match_up('Cowboys', 'Redskins', odds=odds))
    games.append(create_match_up('Lions', 'Packers', odds=odds))
    games.append(create_match_up('Texans', 'Titans', odds=odds))
    games.append(create_match_up('Jaguars', 'Colts', odds=odds))
    games.append(create_match_up('Chiefs', 'Chargers', odds=odds))
    games.append(create_match_up('Vikings', 'Bears', odds=odds))
    games.append(create_match_up('Patriots', 'Dolphins', odds=odds))
    games.append(create_match_up('Broncos', 'Raiders', odds=odds))
    games.append(create_match_up('Rams', 'Cardinals', odds=odds))
    games.append(create_match_up('Seahawks', '49ers', odds=odds))
    return games


def get_wildcard_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('', '', odds=odds))
    games.append(create_match_up('', '', odds=odds))
    games.append(create_match_up('', '', odds=odds))
    games.append(create_match_up('', '', odds=odds))

    return games


def get_divisional_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('', '', odds=odds))
    games.append(create_match_up('', '', odds=odds))
    games.append(create_match_up('', '', odds=odds))
    games.append(create_match_up('', '', odds=odds))

    return games


def get_conference_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('', '', odds=odds))
    games.append(create_match_up('', '', odds=odds))

    return games


def get_superbowl_schedule(week_end_date, get_odds=True):
    if get_odds:
        odds = Odds.get_odds(week_end_date)
    else:
        odds = None

    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('', '', odds=odds, neutral_location=True))

    return games


def monte_carlo(teams, trials=1e3, verbose=False):
    """
    Simulates each game outcome based on every team's elo after every game.  Process is repeated a high number of times
    to determine each teams probable final record and playoff standing.

    :param teams: The list of all the teams in the league
    :param trials: The number of times to repeat the simulation
    :param verbose: If verbose output should be included
    :return: Void
    """

    import maya

    all_trials = list()

    start = maya.now()
    if verbose:
        print(int(trials), 'Trials')
        print('Simulating remaining games...')

    # For each trial
    for trial in range(int(trials)):
        # Get just the name, record and elo of each team
        pseudo_teams = [(team[0], team[1], team[2], team[3], team[4], 0, 0) for team in teams]

        # For each game in the list of games yet to be played
        schedule, spreads, neutral_location = zip(*get_schedule())
        remaining_schedule = schedule[len(completed_games):]
        for game in remaining_schedule:
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
            away_losses = away[2] + 1 if home_victory else away[2]

            home_losses = home[2] + 1 if not home_victory else home[2]
            away_wins = away[1] + 1 if not home_victory else away[1]

            home_ties = home[3] + 1 if draw else home[3]
            away_ties = away[3] + 1 if draw else away[3]

            # Update the teams elo based on the simulated outcome
            home_elo, away_elo = NFL.get_new_elos(home[4], away[4], home_victory, False, 42)

            # Create an updated team
            new_home = (home[0], home_wins, home_losses, home_ties, home_elo, 0, 0)
            new_away = (away[0], away_wins, away_losses, away_ties, away_elo, 0, 0)

            # Update the pseudo teams with the new teams
            pseudo_teams = [new_home if team == home else team for team in pseudo_teams]
            pseudo_teams = [new_away if team == away else team for team in pseudo_teams]

        # Add all the pseudo teams in the trial to a list
        all_trials.append(pseudo_teams)

    if verbose:
        print('Analyzing team outcomes...')

    averaged_teams = list()
    playoff_teams = list()

    # Get a list of team names
    team_names = [team[0] for team in teams]

    # For each team name
    for team_name in team_names:
        if verbose:
            print('Analyzing', team_name)

        team_trials = [get_team(trial, team_name) for trial in all_trials]

        # Get a list of the pseudo teams wins for each trial
        wins = [team[1] for team in team_trials]

        # Get a list of the pseudo teams losses for each trial
        losses = [team[2] for team in team_trials]

        # Get a list of the pseudo teams ties for each trial
        ties = [team[3] for team in team_trials]

        # Get a list of the pseudo teams elos for each trial
        elos = [team[4] for team in team_trials]

        # Create a new pseudo team with the average of each stat
        averaged_team = (team_name,
                         sum(wins) / len(wins),
                         sum(losses) / len(losses),
                         round(sum(ties) / len(ties)),
                         sum(elos) / len(elos),
                         0, 0)

        # Add it to a final list
        averaged_teams.append(averaged_team)

    # For each trial
    for trial_num, trial in enumerate(all_trials):
        if verbose:
            print('Getting playoff picture for trial', trial_num)

        # Get the teams in the playoffs for each trial
        afc_playoff_teams, nfc_playoff_teams = get_playoff_picture(trial)
        trial_playoff_teams = list()
        trial_playoff_teams.extend(afc_playoff_teams)
        trial_playoff_teams.extend(nfc_playoff_teams)
        playoff_teams.append(trial_playoff_teams)

    # Get the percent of trails where each team is in the playoffs
    averaged_teams_with_chances = list()
    for team in averaged_teams:
        playoff_appearances = list(filter(lambda sim: team[0] in [avg[0] for avg in sim], playoff_teams))
        playoff_chances = 100 * len(playoff_appearances) / trials
        team = (team[0], team[1], team[2], team[3], team[4], playoff_chances)
        averaged_teams_with_chances.append(team)

    end = maya.now()
    if verbose:
        print('Elapsed Time', end - start)

    # Print the standings
    Standings.print_monte_carlo_simulation(averaged_teams_with_chances)


def get_pct_chance(home_elo, away_elo):
    """
    Gets a teams percent chance to beat another team, based solely on each team's elo.

    :param home_elo: The first team's elo
    :param away_elo: The second team's elo
    :return: The percent chance of victory for the first team
    """

    # Get a teams expected chance to win based off each teams elo
    q_home = 10 ** (home_elo / 400)
    q_away = 10 ** (away_elo / 400)

    e_home = q_home / (q_home + q_away)

    return e_home


def get_playoff_picture(teams, verbose=False):
    """
    Sorts a list of teams into the official standings and determines the current playoff teams.  Playoff teams are
    divided into 3 categories: 1st round byes, division leaders, and wild cards

    :param teams: The list of all the teams in the league
    :param verbose: If verbose output should be included
    :return: A list of teams that are in the AFC and NFC playoffs
    """

    # Sort the teams by playoff tiebreakers
    teams = sort_by_tiebreakers(teams)

    # Get the league structure
    league = get_league_structure()

    # Lists for each playoff contender type
    first_round_byes = list()
    division_leaders = list()
    wild_cards = list()

    # Lists for playoff contenders split by conference
    afc_division_leaders = list()
    afc_wild_cards = list()

    nfc_division_leaders = list()
    nfc_wild_cards = list()

    # For each conference
    for conf_name, conference in league.items():
        conference_teams = list()

        # For each division
        for div_name, division in conference.items():
            # Get the teams within each division
            division_teams = [get_team(teams, team_name) for team_name in division]

            # Sort the division by the playoff tiebreakers
            division_teams = sort_by_tiebreakers(division_teams)

            # Append the top team to the list of division leaders
            if conf_name == 'AFC':
                afc_division_leaders.append(division_teams[0])
            elif conf_name == 'NFC':
                nfc_division_leaders.append(division_teams[0])
            division_leaders.append(division_teams[0])

            # Add the teams within the division to the list of teams within the conference
            conference_teams.extend(division_teams)

        # Sort the conference by the playoff tiebreakers
        conference_teams = sort_by_tiebreakers(conference_teams)

        # Add the top team to the list of teams with a first round bye
        first_round_byes.append(conference_teams[0])

        # Get the name of the top team in the conference
        conf_leader = conference_teams[0][0]

        # Get the division of the top team in the conference
        conf_leaders_division = None
        for div_name, division in conference.items():
            if conf_leader in division:
                conf_leaders_division = division

        # For each of the next best teams in the conference
        for runner_up in range(1, 5):
            runner_up_name = conference_teams[runner_up][0]

            # Get the division of the next best team
            runner_up_division = None
            for div_name, division in conference.items():
                if runner_up_name in division:
                    runner_up_division = division

            # If the division of the runner up is different from the division of the conference leader
            if runner_up_division != conf_leaders_division:
                # Add the runner up to the list of teams with a first round bye
                first_round_byes.append(conference_teams[runner_up])
                break

        # Get a list of the remaining teams that are not a division leader
        other_teams = conference_teams.copy()
        other_teams = list(set(other_teams) - set(division_leaders))
        other_teams = sort_by_tiebreakers(other_teams)

        # Add the top 2 teams (that are not division leaders) from each conference top the list of wild cards
        if conf_name == 'AFC':
            afc_wild_cards.append(other_teams[0])
            afc_wild_cards.append(other_teams[1])
        elif conf_name == 'NFC':
            nfc_wild_cards.append(other_teams[0])
            nfc_wild_cards.append(other_teams[1])
        wild_cards.append(other_teams[0])
        wild_cards.append(other_teams[1])

    # Print the playoff teams
    if verbose:
        print(Standings.Colors.UNDERLINE + 'First Round Byes:' + Standings.Colors.ENDC)
        first_round_byes = [team[0] for team in first_round_byes]
        print('\n'.join(first_round_byes))
        print()

        print(Standings.Colors.UNDERLINE + 'Division Leaders:' + Standings.Colors.ENDC)
        division_leaders = [team[0] for team in division_leaders]
        print('\n'.join(division_leaders))
        print()

        print(Standings.Colors.UNDERLINE + 'Wild Cards:' + Standings.Colors.ENDC)
        wild_cards = [team[0] for team in wild_cards]
        print('\n'.join(wild_cards))
        print()

    # Create a list of AFC playoff teams
    afc_playoff_teams = list()
    afc_playoff_teams.extend(sort_by_tiebreakers(afc_division_leaders))
    afc_playoff_teams.extend(sort_by_tiebreakers(afc_wild_cards))

    # Create a list of NFC playoff teams
    nfc_playoff_teams = list()
    nfc_playoff_teams.extend(sort_by_tiebreakers(nfc_division_leaders))
    nfc_playoff_teams.extend(sort_by_tiebreakers(nfc_wild_cards))

    # Return all of the playoff teams
    return afc_playoff_teams, nfc_playoff_teams


def sort_by_tiebreakers(teams):
    """
    Sorts a list of teams based on the NFL playoff tiebreaker rules.
    1) Win Percentage
    2) Head to Head Record
    3) Divisional Record
    4) Common Record
    5) Conference Record
    6) Strength of Victory
    7) Strength of Schedule
    8) Point Differential

    :param teams: The list of all the teams in the league
    :return: The list of all the teams in the league sorted by the tiebreaker rules
    """

    # Get a list of teams sorted by the official NFL playoff tiebreakers
    sorted_teams = sorted(teams, key=cmp_to_key(compare_win_pct), reverse=True)
    return sorted_teams


def compare_win_pct(team1, team2):
    """
    Compares two teams based on each teams win percentage.
    Positive: team1 greater win percentage
    Negative: team2 greater win percentage
    Zero: Equal win percentage

    :param team1: The first team to compare
    :param team2: The second team to compare
    :return: A comparison of each team's win percentage
    """

    # Get the percentage of games each team has won
    team1_games_played = team1[1] + team1[2] + team1[3]
    team2_games_played = team2[1] + team2[2] + team2[3]
    team1_win_pct = team1[1] / team1_games_played if team1_games_played > 0 else 0
    team2_win_pct = team2[1] / team2_games_played if team2_games_played > 0 else 0

    # If they have each won the same percentage of games
    if team1_win_pct - team2_win_pct == 0:
        # Get the percentage of games each team has lost
        team1_loss_pct = team1[2] / team1_games_played if team1_games_played > 0 else 0
        team2_loss_pct = team2[2] / team2_games_played if team2_games_played > 0 else 0

        # If they have each lost the same percentage of games
        if team2_loss_pct - team1_loss_pct == 0:
            # Compare the teams based on their head to head games
            return compare_head_to_head(team1, team2)

        # Sort based on which ever team lost least
        return team2_loss_pct - team1_loss_pct

    # Sort based on which ever team won most
    return team1_win_pct - team2_win_pct


def compare_head_to_head(team1, team2):
    """
    Compares two teams based on each teams record against the other team.
    Positive: team1 better head to head record
    Negative: team2 better head to head record
    Zero: Equal head to head record

    :param team1: The first team to compare
    :param team2: The second team to compare
    :return: A comparison of each team's head to head record
    """

    # Get the games between both teams
    head_to_head_games = completed_games.loc[((completed_games['home_team'] == team1[0]) |
                                              (completed_games['away_team'] == team1[0])) &
                                             ((completed_games['home_team'] == team2[0]) |
                                              (completed_games['away_team'] == team2[0]))]

    # Get the number of victories each team had
    team1_victories = get_team_victories(team1[0], head_to_head_games)
    team2_victories = get_team_victories(team2[0], head_to_head_games)

    # If they each won as many as they lost
    if len(team1_victories) - len(team2_victories) == 0:
        # Compare the teams based on their record against teams in their division
        return compare_divisional_record(team1, team2)

    # Sort based on which ever team won most
    return len(team1_victories) - len(team2_victories)


def get_team_victories(team_name, games_df):
    """
    Gets all of the games where a team was victorious.

    :param team_name: The team to get the victorious games
    :param games_df: A dataframe containing the information of all the completed games
    :return: A dataframe containing all the games that the team has won
    """

    # Get the games where the team won
    team_victories = games_df.loc[((games_df['home_team'] == team_name) &
                                   (games_df['home_score'] > games_df['away_score'])) |
                                  ((games_df['away_team'] == team_name) &
                                   (games_df['away_score'] > games_df['home_score']))]
    return team_victories


def compare_divisional_record(team1, team2):
    """
    Compares two teams based on each teams record against teams within the division.
    Positive: team1 better divisional record
    Negative: team2 better divisional record
    Zero: Equal divisional record

    :param team1: The first team to compare
    :param team2: The second team to compare
    :return: A comparison of each team's divisional record
    """

    # Get they games against opponents within each teams division
    team1_divisional_games = get_divisional_games(team1[0], completed_games)
    team2_divisional_games = get_divisional_games(team2[0], completed_games)

    # Get the number of victories each team had
    team1_victories = get_team_victories(team1[0], team1_divisional_games)
    team2_victories = get_team_victories(team2[0], team2_divisional_games)

    # Get the percentage of victories each team had
    team1_divisional_pct = len(team1_victories) / len(team1_divisional_games) if len(team1_divisional_games) > 0 else 0
    team2_divisional_pct = len(team2_victories) / len(team2_divisional_games) if len(team2_divisional_games) > 0 else 0

    # If they have each won the same percentage of games
    if team1_divisional_pct - team2_divisional_pct == 0:
        # Compare the teams based on their record against common opponents
        return compare_common_record(team1, team2)

    # Sort based on which ever team won most
    return team1_divisional_pct - team2_divisional_pct


def get_divisional_games(team_name, games_df):
    """
    Gets all of the games that a team has competed in where the opponent was in the same division.

    :param team_name: The team to get the in division games for
    :param games_df: A dataframe containing the information of all the completed games
    :return: A dataframe containing all the games that the team has competed in against an in division opponent
    """

    # Get the division of the team
    nfl = get_league_structure()
    teams_division = None
    for conf_name, conf in nfl.items():
        for div_name, division in conf.items():
            if team_name in division:
                teams_division = division
                break

    # Get the games where the opponent is in the teams division
    divisional_games = games_df.loc[((games_df['home_team'] == team_name) &
                                     (games_df['away_team'].isin(teams_division))) |
                                    ((games_df['away_team'] == team_name) &
                                     (games_df['home_team'].isin(teams_division)))]

    return divisional_games


def compare_common_record(team1, team2):
    """
    Compares two teams based on each teams record against teams that each team has faced.
    Positive: team1 better common record
    Negative: team2 better common record
    Zero: Equal common record

    :param team1: The first team to compare
    :param team2: The second team to compare
    :return: A comparison of each team's common record
    """

    # Get they games against common opponents
    team1_common_games = get_games_against_common_opponents(team1, team2, completed_games)
    team2_common_games = get_games_against_common_opponents(team2, team1, completed_games)

    # Get the number of victories each team had
    team1_victories = get_team_victories(team1[0], team1_common_games)
    team2_victories = get_team_victories(team2[0], team2_common_games)

    # Get the percentage of victories each team had
    team1_common_pct = len(team1_victories) / len(team1_common_games) if len(team1_common_games) > 0 else 0
    team2_common_pct = len(team2_victories) / len(team2_common_games) if len(team2_common_games) > 0 else 0

    # If they have each won the same percentage of games
    if team1_common_pct - team2_common_pct == 0:
        # Compare the teams based on their record against teams in their conference
        return compare_conference_record(team1, team2)

    # Sort based on which ever team won most
    return team1_common_pct - team2_common_pct


def get_games_against_common_opponents(team1, team2, games_df):
    """
    Gets all of the games that team1 has competed in where the opponent has also faced team2.

    :param team1: The first team to compare
    :param team2: The second team to compare
    :param games_df: A dataframe containing the information of all the completed games
    :return: A dataframe containing all games containing both teams and common opponents
    """

    # Get all of the games each team played in
    team1_games = games_df.loc[(games_df['home_team'] == team1[0]) | (games_df['away_team'] == team1[0])]
    team2_games = games_df.loc[(games_df['home_team'] == team2[0]) | (games_df['away_team'] == team2[0])]

    # Get all of the opponents team 2 faced (excluding team 1)
    team2_opponents = (set(team2_games['home_team'].unique()) |
                       set(team2_games['away_team'].unique())) - {team2[0]} - {team1[0]}

    # Get all the games where team 1 faced an opponent that team 2 faced
    common_opponents = team1_games.loc[(team1_games['home_team'].isin(team2_opponents)) |
                                       (team1_games['away_team'].isin(team2_opponents))]

    return common_opponents


def compare_conference_record(team1, team2):
    """
    Compares two teams based on each teams record against teams within the conference.
    Positive: team1 better conference record
    Negative: team2 better conference record
    Zero: Equal conference record

    :param team1: The first team to compare
    :param team2: The second team to compare
    :return: A comparison of each team's conference record
    """

    # Get they games against opponents within each teams conference
    team1_conference_games = get_conference_games(team1[0], completed_games)
    team2_conference_games = get_conference_games(team2[0], completed_games)

    # Get the number of victories each team had
    team1_victories = get_team_victories(team1[0], team1_conference_games)
    team2_victories = get_team_victories(team2[0], team2_conference_games)

    # Get the percentage of victories each team had
    team1_conference_pct = len(team1_victories) / len(team1_conference_games) if len(team1_conference_games) > 0 else 0
    team2_conference_pct = len(team2_victories) / len(team2_conference_games) if len(team2_conference_games) > 0 else 0

    # If they have each won the same percentage of games
    if team1_conference_pct - team2_conference_pct == 0:
        # Compare the teams based on the total number of wins of the teams they defeated
        return compare_strength_of_victory(team1, team2)

    # Sort based on which ever team won most
    return team1_conference_pct - team2_conference_pct


def get_conference_games(team_name, games_df):
    """
    Gets all of the games that a team has competed in where the opponent was in the same conference.

    :param team_name: The team to get the in conference games for
    :param games_df: A dataframe containing the information of all the completed games
    :return: A dataframe containing all the games that the team has competed in against an in conference opponent
    """

    # Get the conference of the team
    nfl = get_league_structure()
    teams_conference = None
    for conf_name, conf in nfl.items():
        for div_name, division in conf.items():
            if team_name in division:
                teams_conference = conf
                break

    # Get all the teams within the conference
    conference_teams = list()
    for div_name, division in teams_conference.items():
        conference_teams.extend(division)

    # Get the games where the opponent is in the teams conference
    conference_games = games_df.loc[((games_df['home_team'] == team_name) &
                                     (games_df['away_team'].isin(conference_teams))) |
                                    ((games_df['away_team'] == team_name) &
                                     (games_df['home_team'].isin(conference_teams)))]

    return conference_games


def compare_strength_of_victory(team1, team2):
    """
    Compares two teams based on each teams strength of victory. Strength of victory is determined by the combined
    win percentage of all of the opponents a team has defeated.
    Positive: team1 greater strength of victory
    Negative: team2 greater strength of victory
    Zero: Equal strength of victory

    :param team1: The first team to compare
    :param team2: The second team to compare
    :return: A comparison of each team's strength of victory
    """

    import Projects.nfl.NFL_Prediction.NFLSeason2019 as Season

    # Get all the games each team competed in
    team1_games = completed_games.loc[(completed_games['home_team'] == team1[0]) |
                                      (completed_games['away_team'] == team1[0])]

    team2_games = completed_games.loc[(completed_games['home_team'] == team2[0]) |
                                      (completed_games['away_team'] == team2[0])]

    # Get all of the games where each team was victorious
    team1_victories = get_team_victories(team1[0], team1_games)
    team2_victories = get_team_victories(team2[0], team2_games)

    # Get all of the opponents each team defeated
    team1_opponents = (set(team1_victories['home_team'].unique()) |
                       set(team1_victories['away_team'].unique())) - {team1[0]}

    team2_opponents = (set(team2_victories['home_team'].unique()) |
                       set(team2_victories['away_team'].unique())) - {team2[0]}

    # Total the wins, losses and ties of each teams opponents
    team1_opponent_victories = list()
    team1_opponent_losses = list()
    team1_opponent_ties = list()
    for opponent in team1_opponents:
        teams = Season.nfl_teams
        opponent = get_team(teams, opponent)
        team1_opponent_victories.append(opponent[1])
        team1_opponent_losses.append(opponent[2])
        team1_opponent_ties.append(opponent[3])

    team2_opponent_victories = list()
    team2_opponent_losses = list()
    team2_opponent_ties = list()
    for opponent in team2_opponents:
        teams = Season.nfl_teams
        opponent = get_team(teams, opponent)
        team2_opponent_victories.append(opponent[1])
        team2_opponent_losses.append(opponent[2])
        team2_opponent_ties.append(opponent[3])

    # Get the combined win percentage of each teams opponents
    team1_opponent_games_played = sum(team1_opponent_victories) + sum(team1_opponent_losses) + sum(team1_opponent_ties)
    team1_opponent_win_pct = sum(team1_opponent_victories) / team1_opponent_games_played \
        if team1_opponent_games_played > 0 else 0

    team2_opponent_games_played = sum(team2_opponent_victories) + sum(team2_opponent_losses) + sum(team2_opponent_ties)
    team2_opponent_win_pct = sum(team2_opponent_victories) / team2_opponent_games_played \
        if team2_opponent_games_played > 0 else 0

    # If each teams opponents have the same combined win percentage
    if team1_opponent_win_pct - team2_opponent_win_pct == 0:
        # Compare the teams based on the total number of wins of the teams they faced
        return compare_strength_of_schedule(team1, team2)

    # Sort based on which ever team defeated opponents with the highest combined win percentage
    return team1_opponent_win_pct - team2_opponent_win_pct


def compare_strength_of_schedule(team1, team2):
    """
    Compares two teams based on each teams strength of schedule. Strength of schedule is determined by the combined
    win percentage of all of a teams opponents.
    Positive: team1 more difficult schedule
    Negative: team2 more difficult schedule
    Zero: Equal schedule difficulty

    :param team1: The first team to compare
    :param team2: The second team to compare
    :return: A comparison of each team's strength of schedule
    """

    import Projects.nfl.NFL_Prediction.NFLSeason2019 as Season

    # Get all the games each team competed in
    team1_games = completed_games.loc[(completed_games['home_team'] == team1[0]) |
                                      (completed_games['away_team'] == team1[0])]

    team2_games = completed_games.loc[(completed_games['home_team'] == team2[0]) |
                                      (completed_games['away_team'] == team2[0])]

    # Get all of the opponents each team faced
    team1_opponents = (set(team1_games['home_team'].unique()) |
                       set(team1_games['away_team'].unique())) - {team1[0]}

    team2_opponents = (set(team2_games['home_team'].unique()) |
                       set(team2_games['away_team'].unique())) - {team2[0]}

    # Total the wins, losses and ties of each teams opponents
    team1_opponent_victories = list()
    team1_opponent_losses = list()
    team1_opponent_ties = list()
    for opponent in team1_opponents:
        teams = Season.nfl_teams
        opponent = get_team(teams, opponent)
        team1_opponent_victories.append(opponent[1])
        team1_opponent_losses.append(opponent[2])
        team1_opponent_ties.append(opponent[3])

    team2_opponent_victories = list()
    team2_opponent_losses = list()
    team2_opponent_ties = list()
    for opponent in team2_opponents:
        teams = Season.nfl_teams
        opponent = get_team(teams, opponent)
        team2_opponent_victories.append(opponent[1])
        team2_opponent_losses.append(opponent[2])
        team2_opponent_ties.append(opponent[3])

    # Get the combined win percentage of each teams opponents
    team1_opponent_games_played = sum(team1_opponent_victories) + sum(team1_opponent_losses) + sum(team1_opponent_ties)
    team1_opponent_win_pct = sum(team1_opponent_victories) / team1_opponent_games_played \
        if team1_opponent_games_played > 0 else 0

    team2_opponent_games_played = sum(team2_opponent_victories) + sum(team2_opponent_losses) + sum(team2_opponent_ties)
    team2_opponent_win_pct = sum(team2_opponent_victories) / team2_opponent_games_played \
        if team2_opponent_games_played > 0 else 0

    # If each teams opponents have the same combined win percentage
    if team1_opponent_win_pct - team2_opponent_win_pct == 0:
        # Compare the teams based on their point differential
        return compare_point_diff(team1, team2)

    # Sort based on which ever team faced opponents with the highest combined win percentage
    return team1_opponent_win_pct - team2_opponent_win_pct


def compare_point_diff(team1, team2):
    """
    Compares two teams based on each teams point differential.
    Positive: team1 greater point differential
    Negative: team2 greater point differential
    Zero: Equal point differential

    :param team1: The first team to compare
    :param team2: The second team to compare
    :return: A comparison of each team's point differential
    """

    # Get the point differential of each team
    team1_point_diff = team1[5] - team1[6]
    team2_point_diff = team2[5] - team2[6]

    # Sort based on which ever team has the highest point differential
    return team1_point_diff - team2_point_diff


def get_schedule_difficulty(teams, team_name, remaining=False):
    """
    Gets the difficulty of a team's schedule, expressed in the average elo of all of the teams opponents.

    :param teams: The list of all the teams in the league
    :param team_name: The name of the team to get the schedule difficulty for
    :param remaining: If the schedule difficulty should only be based off of the teams remaining games
    :return: The average elo of each opposing team in the team's schedule.
    """

    # Get the schedule
    schedule, spreads, neutral_location = zip(*get_schedule())

    # If only remaining games
    if remaining:
        # Remove the completed games
        schedule = schedule[len(completed_games):]

    # Filter the total schedule down to just the games the team is in
    teams_schedule = list(filter(lambda g: g[0] == team_name or g[1] == team_name, schedule))
    if len(teams_schedule) == 0:
        return 0, 0

    # For each game in the teams schedule
    opponent_elos = list()
    for game in teams_schedule:
        # Get the teams opponent
        opponent = None
        if game[0] == team_name:
            opponent = get_team(teams, game[1])
        elif game[1] == team_name:
            opponent = get_team(teams, game[0])

        # Add the opponents elo to the list
        opponent_elos.append(opponent[4])

    # Get the standard deviation of the opponents
    if len(opponent_elos) > 1:
        deviation = statistics.pstdev(opponent_elos)
    else:
        deviation = 0

    # Return the average opponent elo and the standard deviation
    return statistics.mean(opponent_elos), deviation


def get_completed_schedule_difficulty(team_name):
    """
    Gets the average difficulty of all the opponents a team has faced.

    :param team_name: The name of the team to get the difficulty for
    :return: The average elo of each opposing team that the team has faced.
    """

    team_completed_games = completed_games.loc[(completed_games['home_team'] == team_name) |
                                               (completed_games['away_team'] == team_name)]

    if len(team_completed_games) == 0:
        return 0, 0

    home = pd.Series(team_completed_games['home_team'] == team_name)
    opponent_elos = pd.Series(team_completed_games.lookup(team_completed_games.index, home.map({True: 'away_elo',
                                                                                                False: 'home_elo'})))

    if len(opponent_elos) > 1:
        deviation = statistics.pstdev(opponent_elos)
    else:
        deviation = 0

    return statistics.mean(opponent_elos), deviation


def create_playoff_bracket(teams):
    """
    Creates formatted playoff brackets for the start of the playoffs.  Creates one bracket per conference and places
    teams based on their seed and initial match up. Wildcard teams are marked with an asterisk.  Does not update teams
    midway through the playoffs.

    :param teams: A list of all teams that are currently in the playoffs.
    :return: Void
    """

    # Create a list of playoff teams and their seed
    seeded_teams = list()
    for seed, team in enumerate(teams):
        seeded_teams.append((seed, team))

    # Get the length of the longest name in the list of playoff teams
    max_name_length = max([len(team[0]) for team in teams]) + 3

    # Pad the team names for formatting
    padded_team_names = [team[0].rjust(max_name_length) for team in teams]

    # Put an asterisk next to the wildcard teams
    for seed, name in enumerate(padded_team_names):
        if seed == 4 or seed == 5:
            name = name.strip()
            name = '*' + name
            name = name.rjust(max_name_length)
            padded_team_names[seed] = name

    # Print a bracket
    print(' ' * 5 + padded_team_names[0] + '--|')
    print(' ' * (max_name_length + 7) + '|')
    print(' ' * (max_name_length + 7) + '|----|')
    print(padded_team_names[4] + '--|' + ' ' * 4 + '|' + ' ' * 4 + '|')
    print(' ' * (max_name_length + 2) + '|----|' + ' ' * 4 + '|')
    print(padded_team_names[3] + '--|' + ' ' * 9 + '|')
    print(' ' * (max_name_length + 12) + '|')
    print(' ' * (max_name_length + 12) + '|----')
    print(' ' * (max_name_length + 12) + '|')
    print(padded_team_names[5] + '--|' + ' ' * 9 + '|')
    print(' ' * (max_name_length + 2) + '|----|' + ' ' * 4 + '|')
    print(padded_team_names[2] + '--|' + ' ' * 4 + '|' + ' ' * 4 + '|')
    print(' ' * (max_name_length + 7) + '|----|')
    print(' ' * (max_name_length + 7) + '|')
    print(' ' * 5 + padded_team_names[1] + '--|')
