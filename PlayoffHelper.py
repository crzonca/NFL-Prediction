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


def get_schedule():
    games = list()
    games.extend(get_week1_schedule())
    games.extend(get_week2_schedule())
    games.extend(get_week3_schedule())
    games.extend(get_week4_schedule())

    games.extend(get_week5_schedule())
    games.extend(get_week6_schedule())
    games.extend(get_week7_schedule())
    games.extend(get_week8_schedule())

    games.extend(get_week9_schedule())
    games.extend(get_week10_schedule())
    games.extend(get_week11_schedule())
    games.extend(get_week12_schedule())

    games.extend(get_week13_schedule())
    games.extend(get_week14_schedule())
    games.extend(get_week15_schedule())
    games.extend(get_week16_schedule())

    games.extend(get_week17_schedule())
    games.extend(get_wildcard_schedule())
    games.extend(get_divisional_schedule())
    games.extend(get_conference_schedule())
    games.extend(get_superbowl_schedule())
    return games


def create_match_up(home_name, away_name, home_spread, neutral_location=False):
    return (home_name, away_name), home_spread, neutral_location


def get_pre_week1_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Giants', 'Jets', 0, True))
    games.append(create_match_up('Eagles', 'Titans', 0))
    games.append(create_match_up('Bears', 'Panthers', 0))
    games.append(create_match_up('Lions', 'Patriots', 0))
    games.append(create_match_up('Packers', 'Texans', 0))
    games.append(create_match_up('Saints', 'Vikings', 0))
    games.append(create_match_up('Cardinals', 'Chargers', 0))
    games.append(create_match_up('49ers', 'Cowboys', 0))
    games.append(create_match_up('Seahawks', 'Broncos', 0))
    games.append(create_match_up('Bills', 'Colts', 0))
    games.append(create_match_up('Dolphins', 'Falcons', 0))
    games.append(create_match_up('Ravens', 'Jaguars', 0))
    games.append(create_match_up('Browns', 'Redskins', 0))
    games.append(create_match_up('Steelers', 'Buccaneers', 0))
    games.append(create_match_up('Chiefs', 'Bengals', 0))
    games.append(create_match_up('Raiders', 'Rams', 0))

    return games


def get_pre_week2_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Giants', 'Bears', 0))
    games.append(create_match_up('Redskins', 'Bengals', 0))
    games.append(create_match_up('Vikings', 'Seahawks', 0))
    games.append(create_match_up('Falcons', 'Jets', 0))
    games.append(create_match_up('Panthers', 'Bills', 0))
    games.append(create_match_up('Buccaneers', 'Dolphins', 0))
    games.append(create_match_up('Cardinals', 'Raiders', 0))
    games.append(create_match_up('Rams', 'Cowboys', 0))
    games.append(create_match_up('Ravens', 'Packers', 0))
    games.append(create_match_up('Steelers', 'Chiefs', 0))
    games.append(create_match_up('Texans', 'Lions', 0))
    games.append(create_match_up('Colts', 'Browns', 0))
    games.append(create_match_up('Jaguars', 'Eagles', 0))
    games.append(create_match_up('Titans', 'Patriots', 0))
    games.append(create_match_up('Broncos', '49ers', 0))
    games.append(create_match_up('Chargers', 'Saints', 0))

    return games


def get_pre_week3_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Cowboys', 'Texans', 0))
    games.append(create_match_up('Eagles', 'Ravens', 0))
    games.append(create_match_up('Lions', 'Bills', 0))
    games.append(create_match_up('Vikings', 'Cardinals', 0))
    games.append(create_match_up('Falcons', 'Redskins', 0))
    games.append(create_match_up('Buccaneers', 'Browns', 0))
    games.append(create_match_up('Rams', 'Broncos', 0))
    games.append(create_match_up('Dolphins', 'Jaguars', 0))
    games.append(create_match_up('Patriots', 'Panthers', 0))
    games.append(create_match_up('Jets', 'Saints', 0))
    games.append(create_match_up('Bengals', 'Giants', 0))
    games.append(create_match_up('Colts', 'Bears', 0))
    games.append(create_match_up('Titans', 'Steelers', 0))
    games.append(create_match_up('Chiefs', '49ers', 0))
    games.append(create_match_up('Raiders', 'Packers', 0))
    games.append(create_match_up('Chargers', 'Seahawks', 0))

    return games


def get_pre_week4_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Cowboys', 'Buccaneers', 0))
    games.append(create_match_up('Redskins', 'Ravens', 0))
    games.append(create_match_up('Bears', 'Titans', 0))
    games.append(create_match_up('Packers', 'Chiefs', 0))
    games.append(create_match_up('Panthers', 'Steelers', 0))
    games.append(create_match_up('Saints', 'Dolphins', 0))
    games.append(create_match_up('Chargers', '49ers', 0))
    games.append(create_match_up('Seahawks', 'Raiders', 0))
    games.append(create_match_up('Bills', 'Vikings', 0))
    games.append(create_match_up('Patriots', 'Giants', 0))
    games.append(create_match_up('Jets', 'Eagles', 0))
    games.append(create_match_up('Bengals', 'Colts', 0))
    games.append(create_match_up('Browns', 'Lions', 0))
    games.append(create_match_up('Texans', 'Rams', 0))
    games.append(create_match_up('Jaguars', 'Falcons', 0))
    games.append(create_match_up('Broncos', 'Cardinals', 0))

    return games


def get_week1_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Bears', 'Packers', -3.5))
    games.append(create_match_up('Panthers', 'Rams', 2.5))
    games.append(create_match_up('Eagles', 'Redskins', -8))
    games.append(create_match_up('Jets', 'Bills', -3.5))
    games.append(create_match_up('Vikings', 'Falcons', -4.5))
    games.append(create_match_up('Dolphins', 'Ravens', 3.5))
    games.append(create_match_up('Jaguars', 'Chiefs', 5.5))
    games.append(create_match_up('Browns', 'Titans', -5))
    games.append(create_match_up('Chargers', 'Colts', -3.5))
    games.append(create_match_up('Seahawks', 'Bengals', -7.5))
    games.append(create_match_up('Buccaneers', '49ers', -2.5))
    games.append(create_match_up('Cowboys', 'Giants', -7.5))
    games.append(create_match_up('Cardinals', 'Lions', 0))
    games.append(create_match_up('Patriots', 'Steelers', -6))
    games.append(create_match_up('Saints', 'Texans', -7.5))
    games.append(create_match_up('Raiders', 'Broncos', -2.5))

    return games


def get_week2_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Panthers', 'Buccaneers', 0))
    games.append(create_match_up('Ravens', 'Cardinals', 0))
    games.append(create_match_up('Redskins', 'Cowboys', 0))
    games.append(create_match_up('Titans', 'Colts', 0))
    games.append(create_match_up('Steelers', 'Seahawks', 0))
    games.append(create_match_up('Giants', 'Bills', 0))
    games.append(create_match_up('Bengals', '49ers', 0))
    games.append(create_match_up('Lions', 'Chargers', 0))
    games.append(create_match_up('Packers', 'Vikings', 0))
    games.append(create_match_up('Texans', 'Jaguars', 0))
    games.append(create_match_up('Dolphins', 'Patriots', 0))
    games.append(create_match_up('Raiders', 'Chiefs', 0))
    games.append(create_match_up('Rams', 'Saints', 0))
    games.append(create_match_up('Broncos', 'Bears', 0))
    games.append(create_match_up('Falcons', 'Eagles', 0))
    games.append(create_match_up('Jets', 'Browns', 0))

    return games


def get_week3_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Jaguars', 'Titans', 0))
    games.append(create_match_up('Bills', 'Bengals', 0))
    games.append(create_match_up('Eagles', 'Lions', 0))
    games.append(create_match_up('Patriots', 'Jets', 0))
    games.append(create_match_up('Vikings', 'Raiders', 0))
    games.append(create_match_up('Chiefs', 'Ravens', 0))
    games.append(create_match_up('Colts', 'Falcons', 0))
    games.append(create_match_up('Packers', 'Broncos', 0))
    games.append(create_match_up('Cowboys', 'Dolphins', 0))
    games.append(create_match_up('Buccaneers', 'Giants', 0))
    games.append(create_match_up('Cardinals', 'Panthers', 0))
    games.append(create_match_up('49ers', 'Steelers', 0))
    games.append(create_match_up('Seahawks', 'Saints', 0))
    games.append(create_match_up('Chargers', 'Texans', 0))
    games.append(create_match_up('Browns', 'Rams', 0))
    games.append(create_match_up('Redskins', 'Bears', 0))

    return games


def get_week4_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Packers', 'Eagles', 0))
    games.append(create_match_up('Falcons', 'Titans', 0))
    games.append(create_match_up('Giants', 'Redskins', 0))
    games.append(create_match_up('Dolphins', 'Chargers', 0))
    games.append(create_match_up('Colts', 'Raiders', 0))
    games.append(create_match_up('Texans', 'Panthers', 0))
    games.append(create_match_up('Lions', 'Chiefs', 0))
    games.append(create_match_up('Ravens', 'Browns', 0))
    games.append(create_match_up('Bills', 'Patriots', 0))
    games.append(create_match_up('Rams', 'Buccaneers', 0))
    games.append(create_match_up('Cardinals', 'Seahawks', 0))
    games.append(create_match_up('Bears', 'Vikings', 0))
    games.append(create_match_up('Broncos', 'Jaguars', 0))
    games.append(create_match_up('Saints', 'Cowboys', 0))
    games.append(create_match_up('Steelers', 'Bengals', 0))

    return games


def get_week5_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Seahawks', 'Rams', 0))
    games.append(create_match_up('Panthers', 'Jaguars', 0))
    games.append(create_match_up('Redskins', 'Patriots', 0))
    games.append(create_match_up('Titans', 'Bills', 0))
    games.append(create_match_up('Steelers', 'Ravens', 0))
    games.append(create_match_up('Bengals', 'Cardinals', 0))
    games.append(create_match_up('Texans', 'Falcons', 0))
    games.append(create_match_up('Saints', 'Buccaneers', 0))
    games.append(create_match_up('Giants', 'Vikings', 0))
    games.append(create_match_up('Raiders', 'Bears', 0, True))
    games.append(create_match_up('Eagles', 'Jets', 0))
    games.append(create_match_up('Chargers', 'Broncos', 0))
    games.append(create_match_up('Cowboys', 'Packers', 0))
    games.append(create_match_up('Chiefs', 'Colts', 0))
    games.append(create_match_up('49ers', 'Browns', 0))

    return games


def get_week6_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Patriots', 'Giants', 0))
    games.append(create_match_up('Buccaneers', 'Panthers', 0, True))
    games.append(create_match_up('Dolphins', 'Redskins', 0))
    games.append(create_match_up('Vikings', 'Eagles', 0))
    games.append(create_match_up('Chiefs', 'Texans', 0))
    games.append(create_match_up('Jaguars', 'Saints', 0))
    games.append(create_match_up('Browns', 'Seahawks', 0))
    games.append(create_match_up('Ravens', 'Bengals', 0))
    games.append(create_match_up('Rams', '49ers', 0))
    games.append(create_match_up('Cardinals', 'Falcons', 0))
    games.append(create_match_up('Jets', 'Cowboys', 0))
    games.append(create_match_up('Broncos', 'Titans', 0))
    games.append(create_match_up('Chargers', 'Steelers', 0))
    games.append(create_match_up('Packers', 'Lions', 0))

    return games


def get_week7_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Broncos', 'Chiefs', 0))
    games.append(create_match_up('Bills', 'Dolphins', 0))
    games.append(create_match_up('Bengals', 'Jaguars', 0))
    games.append(create_match_up('Lions', 'Vikings', 0))
    games.append(create_match_up('Packers', 'Raiders', 0))
    games.append(create_match_up('Falcons', 'Rams', 0))
    games.append(create_match_up('Colts', 'Texans', 0))
    games.append(create_match_up('Redskins', '49ers', 0))
    games.append(create_match_up('Giants', 'Cardinals', 0))
    games.append(create_match_up('Titans', 'Chargers', 0))
    games.append(create_match_up('Bears', 'Saints', 0))
    games.append(create_match_up('Seahawks', 'Ravens', 0))
    games.append(create_match_up('Cowboys', 'Eagles', 0))
    games.append(create_match_up('Jets', 'Patriots', 0))

    return games


def get_week8_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Vikings', 'Redskins', 0))
    games.append(create_match_up('Falcons', 'Seahawks', 0))
    games.append(create_match_up('Titans', 'Buccaneers', 0))
    games.append(create_match_up('Saints', 'Cardinals', 0))
    games.append(create_match_up('Rams', 'Bengals', 0, True))
    games.append(create_match_up('Jaguars', 'Jets', 0))
    games.append(create_match_up('Bills', 'Eagles', 0))
    games.append(create_match_up('Bears', 'Chargers', 0))
    games.append(create_match_up('Lions', 'Giants', 0))
    games.append(create_match_up('Texans', 'Raiders', 0))
    games.append(create_match_up('49ers', 'Panthers', 0))
    games.append(create_match_up('Patriots', 'Browns', 0))
    games.append(create_match_up('Colts', 'Broncos', 0))
    games.append(create_match_up('Chiefs', 'Packers', 0))
    games.append(create_match_up('Steelers', 'Dolphins', 0))

    return games


def get_week9_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Cardinals', '49ers', 0))
    games.append(create_match_up('Jaguars', 'Texans', 0, True))
    games.append(create_match_up('Eagles', 'Bears', 0))
    games.append(create_match_up('Steelers', 'Colts', 0))
    games.append(create_match_up('Dolphins', 'Jets', 0))
    games.append(create_match_up('Chiefs', 'Vikings', 0))
    games.append(create_match_up('Panthers', 'Titans', 0))
    games.append(create_match_up('Bills', 'Redskins', 0))
    games.append(create_match_up('Seahawks', 'Buccaneers', 0))
    games.append(create_match_up('Raiders', 'Lions', 0))
    games.append(create_match_up('Chargers', 'Packers', 0))
    games.append(create_match_up('Broncos', 'Browns', 0))
    games.append(create_match_up('Ravens', 'Patriots', 0))
    games.append(create_match_up('Giants', 'Cowboys', 0))

    return games


def get_week10_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Raiders', 'Chargers', 0))
    games.append(create_match_up('Bengals', 'Ravens', 0))
    games.append(create_match_up('Browns', 'Bills', 0))
    games.append(create_match_up('Packers', 'Panthers', 0))
    games.append(create_match_up('Saints', 'Falcons', 0))
    games.append(create_match_up('Bears', 'Lions', 0))
    games.append(create_match_up('Jets', 'Giants', 0, True))
    games.append(create_match_up('Titans', 'Chiefs', 0))
    games.append(create_match_up('Buccaneers', 'Cardinals', 0))
    games.append(create_match_up('Colts', 'Dolphins', 0))
    games.append(create_match_up('Steelers', 'Rams', 0))
    games.append(create_match_up('Cowboys', 'Vikings', 0))
    games.append(create_match_up('49ers', 'Seahawks', 0))

    return games


def get_week11_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Browns', 'Steelers', 0))
    games.append(create_match_up('Panthers', 'Falcons', 0))
    games.append(create_match_up('Lions', 'Cowboys', 0))
    games.append(create_match_up('Colts', 'Jaguars', 0))
    games.append(create_match_up('Dolphins', 'Bills', 0))
    games.append(create_match_up('Ravens', 'Texans', 0))
    games.append(create_match_up('Vikings', 'Broncos', 0))
    games.append(create_match_up('Redskins', 'Jets', 0))
    games.append(create_match_up('Buccaneers', 'Saints', 0))
    games.append(create_match_up('49ers', 'Cardinals', 0))
    games.append(create_match_up('Raiders', 'Bengals', 0))
    games.append(create_match_up('Eagles', 'Patriots', 0))
    games.append(create_match_up('Rams', 'Bears', 0))
    games.append(create_match_up('Chargers', 'Chiefs', 0, True))

    return games


def get_week12_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Texans', 'Colts', 0))
    games.append(create_match_up('Bills', 'Broncos', 0))
    games.append(create_match_up('Bears', 'Giants', 0))
    games.append(create_match_up('Bengals', 'Steelers', 0))
    games.append(create_match_up('Browns', 'Dolphins', 0))
    games.append(create_match_up('Falcons', 'Buccaneers', 0))
    games.append(create_match_up('Saints', 'Panthers', 0))
    games.append(create_match_up('Redskins', 'Lions', 0))
    games.append(create_match_up('Jets', 'Raiders', 0))
    games.append(create_match_up('Titans', 'Jaguars', 0))
    games.append(create_match_up('Patriots', 'Cowboys', 0))
    games.append(create_match_up('49ers', 'Packers', 0))
    games.append(create_match_up('Eagles', 'Seahawks', 0))
    games.append(create_match_up('Rams', 'Ravens', 0))

    return games


def get_week13_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Lions', 'Bears', 0))
    games.append(create_match_up('Cowboys', 'Bills', 0))
    games.append(create_match_up('Falcons', 'Saints', 0))
    games.append(create_match_up('Colts', 'Titans', 0))
    games.append(create_match_up('Bengals', 'Jets', 0))
    games.append(create_match_up('Panthers', 'Redskins', 0))
    games.append(create_match_up('Ravens', '49ers', 0))
    games.append(create_match_up('Jaguars', 'Buccaneers', 0))
    games.append(create_match_up('Giants', 'Packers', 0))
    games.append(create_match_up('Dolphins', 'Eagles', 0))
    games.append(create_match_up('Chiefs', 'Raiders', 0))
    games.append(create_match_up('Cardinals', 'Rams', 0))
    games.append(create_match_up('Steelers', 'Browns', 0))
    games.append(create_match_up('Broncos', 'Chargers', 0))
    games.append(create_match_up('Texans', 'Patriots', 0))
    games.append(create_match_up('Seahawks', 'Vikings', 0))

    return games


def get_week14_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Bears', 'Cowboys', 0))
    games.append(create_match_up('Falcons', 'Panthers', 0))
    games.append(create_match_up('Buccaneers', 'Colts', 0))
    games.append(create_match_up('Jets', 'Dolphins', 0))
    games.append(create_match_up('Saints', '49ers', 0))
    games.append(create_match_up('Vikings', 'Lions', 0))
    games.append(create_match_up('Texans', 'Broncos', 0))
    games.append(create_match_up('Bills', 'Ravens', 0))
    games.append(create_match_up('Browns', 'Bengals', 0))
    games.append(create_match_up('Packers', 'Redskins', 0))
    games.append(create_match_up('Jaguars', 'Chargers', 0))
    games.append(create_match_up('Cardinals', 'Steelers', 0))
    games.append(create_match_up('Raiders', 'Titans', 0))
    games.append(create_match_up('Patriots', 'Chiefs', 0))
    games.append(create_match_up('Rams', 'Seahawks', 0))
    games.append(create_match_up('Eagles', 'Giants', 0))

    return games


def get_week15_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Ravens', 'Jets', 0))
    games.append(create_match_up('Panthers', 'Seahawks', 0))
    games.append(create_match_up('Redskins', 'Eagles', 0))
    games.append(create_match_up('Titans', 'Texans', 0))
    games.append(create_match_up('Steelers', 'Bills', 0))
    games.append(create_match_up('Giants', 'Dolphins', 0))
    games.append(create_match_up('Chiefs', 'Broncos', 0))
    games.append(create_match_up('Bengals', 'Patriots', 0))
    games.append(create_match_up('Lions', 'Buccaneers', 0))
    games.append(create_match_up('Packers', 'Bears', 0))
    games.append(create_match_up('Raiders', 'Jaguars', 0))
    games.append(create_match_up('Cardinals', 'Browns', 0))
    games.append(create_match_up('49ers', 'Falcons', 0))
    games.append(create_match_up('Cowboys', 'Rams', 0))
    games.append(create_match_up('Chargers', 'Vikings', 0))
    games.append(create_match_up('Saints', 'Colts', 0))

    return games


def get_week16_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Broncos', 'Lions', 0))
    games.append(create_match_up('Chargers', 'Raiders', 0))
    games.append(create_match_up('Redskins', 'Giants', 0))
    games.append(create_match_up('Titans', 'Saints', 0))
    games.append(create_match_up('Jets', 'Steelers', 0))
    games.append(create_match_up('Patriots', 'Bills', 0))
    games.append(create_match_up('49ers', 'Rams', 0))
    games.append(create_match_up('Buccaneers', 'Texans', 0))
    games.append(create_match_up('Falcons', 'Jaguars', 0))
    games.append(create_match_up('Browns', 'Ravens', 0))
    games.append(create_match_up('Colts', 'Panthers', 0))
    games.append(create_match_up('Dolphins', 'Bengals', 0))
    games.append(create_match_up('Seahawks', 'Cardinals', 0))
    games.append(create_match_up('Eagles', 'Cowboys', 0))
    games.append(create_match_up('Bears', 'Chiefs', 0))
    games.append(create_match_up('Vikings', 'Packers', 0))

    return games


def get_week17_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    games.append(create_match_up('Ravens', 'Steelers', 0))
    games.append(create_match_up('Bills', 'Jets', 0))
    games.append(create_match_up('Buccaneers', 'Falcons', 0))
    games.append(create_match_up('Giants', 'Eagles', 0))
    games.append(create_match_up('Panthers', 'Saints', 0))
    games.append(create_match_up('Bengals', 'Browns', 0))
    games.append(create_match_up('Cowboys', 'Redskins', 0))
    games.append(create_match_up('Lions', 'Packers', 0))
    games.append(create_match_up('Texans', 'Titans', 0))
    games.append(create_match_up('Jaguars', 'Colts', 0))
    games.append(create_match_up('Chiefs', 'Chargers', 0))
    games.append(create_match_up('Vikings', 'Bears', 0))
    games.append(create_match_up('Patriots', 'Dolphins', 0))
    games.append(create_match_up('Broncos', 'Raiders', 0))
    games.append(create_match_up('Rams', 'Cardinals', 0))
    games.append(create_match_up('Seahawks', '49ers', 0))
    return games


def get_wildcard_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    # games.append(create_match_up('Cowboys', 'Buccaneers', 0))
    # games.append(create_match_up('Redskins', 'Ravens', 0))
    # games.append(create_match_up('Bears', 'Titans', 0))
    # games.append(create_match_up('Packers', 'Chiefs', 0))

    return games


def get_divisional_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    # games.append(create_match_up('Cowboys', 'Buccaneers', 0))
    # games.append(create_match_up('Redskins', 'Ravens', 0))
    # games.append(create_match_up('Bears', 'Titans', 0))
    # games.append(create_match_up('Packers', 'Chiefs', 0))

    return games


def get_conference_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    # games.append(create_match_up('Bears', 'Titans', 0))
    # games.append(create_match_up('Packers', 'Chiefs', 0))

    return games


def get_superbowl_schedule():
    games = list()
    # Games are listed as: Home Team, Away Team, Spread if Home is favored (-1 * spread otherwise), neutral location
    # games.append(create_match_up('Bears', 'Titans', 0))

    return games


def monte_carlo(teams, trials=100000):
    all_trials = list()
    # 1,000,000 Trials
    for trial in range(trials):
        # Get just the name, record and elo of each team
        pseudo_teams = [(team[0], team[1], team[2], team[3], team[4], 0, 0) for team in teams]

        # For each game in the list of games yet to be played
        schedule, spreads, neutral_location = zip(*get_schedule())
        for game in schedule[len(completed_games):]:
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
            home_ties = home[3] + 1 if draw else home[3]
            away_wins = away[1] + 1 if not home_victory else away[1]
            away_losses = away[2] + 1 if home_victory else away[2]
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

    # Get the teams in the playoffs for each trial
    playoff_teams = list()
    for trial in all_trials:
        afc_playoff_teams, nfc_playoff_teams = get_playoff_picture(trial)
        trial_playoff_teams = list()
        trial_playoff_teams.extend(afc_playoff_teams)
        trial_playoff_teams.extend(nfc_playoff_teams)
        playoff_teams.append(trial_playoff_teams)

    # Get the percent of trails where each team is in the playoffs
    averaged_teams_with_chances = list()
    for team in averaged_teams:
        playoff_appearances = list(filter(lambda s: team[0] in [avg[0] for avg in s], playoff_teams))
        playoff_chances = 100 * len(playoff_appearances) / trials
        team = (team[0], team[1], team[2], team[3], team[4], playoff_chances)
        averaged_teams_with_chances.append(team)

    # Print the standings
    Standings.print_monte_carlo_simulation(averaged_teams_with_chances)


def get_pct_chance(home_elo, away_elo):
    q_home = 10 ** (home_elo / 400)
    q_away = 10 ** (away_elo / 400)

    e_home = q_home / (q_home + q_away)

    return e_home


def get_playoff_picture(teams, verbose=False):
    teams = sort_by_tiebreakers(teams)
    league = get_league_structure()

    first_round_byes = list()
    division_leaders = list()
    wild_cards = list()

    afc_division_leaders = list()
    afc_wild_cards = list()

    nfc_division_leaders = list()
    nfc_wild_cards = list()

    for conf_name, conference in league.items():
        conference_teams = list()

        for div_name, division in conference.items():
            division_teams = [get_team(teams, team_name) for team_name in division]
            division_teams = sort_by_tiebreakers(division_teams)

            if conf_name == 'AFC':
                afc_division_leaders.append(division_teams[0])
            elif conf_name == 'NFC':
                nfc_division_leaders.append(division_teams[0])
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

        if conf_name == 'AFC':
            afc_wild_cards.append(other_teams[0])
            afc_wild_cards.append(other_teams[1])
        elif conf_name == 'NFC':
            nfc_wild_cards.append(other_teams[0])
            nfc_wild_cards.append(other_teams[1])
        wild_cards.append(other_teams[0])
        wild_cards.append(other_teams[1])

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

    afc_playoff_teams = list()
    afc_playoff_teams.extend(sort_by_tiebreakers(afc_division_leaders))
    afc_playoff_teams.extend(sort_by_tiebreakers(afc_wild_cards))

    nfc_playoff_teams = list()
    nfc_playoff_teams.extend(sort_by_tiebreakers(nfc_division_leaders))
    nfc_playoff_teams.extend(sort_by_tiebreakers(nfc_wild_cards))

    return afc_playoff_teams, nfc_playoff_teams


def sort_by_tiebreakers(teams):
    sorted_teams = sorted(teams, key=cmp_to_key(compare_win_pct), reverse=True)
    return sorted_teams


def compare_win_pct(team1, team2):
    team1_games_played = team1[1] + team1[2]
    team2_games_played = team2[1] + team2[2]
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


def get_schedule_difficulty(teams, team_name, remaining=False):
    schedule, spreads, neutral_location = zip(*get_schedule())
    if remaining:
        schedule = schedule[len(completed_games):]

    teams_schedule = list(filter(lambda g: g[0] == team_name or g[1] == team_name, schedule))
    if len(teams_schedule) == 0:
        return 0

    opponent_elos = list()
    for game in teams_schedule:
        opponent = None
        if game[0] == team_name:
            opponent = get_team(teams, game[1])
        elif game[1] == team_name:
            opponent = get_team(teams, game[0])
        opponent_elos.append(opponent[4])

    if len(opponent_elos) > 1:
        deviation = statistics.stdev(opponent_elos)
    else:
        deviation = 0

    return statistics.mean(opponent_elos), deviation


def create_playoff_bracket(teams):
    seeded_teams = list()
    for seed, team in enumerate(teams):
        seeded_teams.append((seed, team))

    max_name_length = max([len(team[0]) for team in teams]) + 3

    padded_team_names = [team[0].rjust(max_name_length) for team in teams]
    for index, name in enumerate(padded_team_names):
        if index == 4 or index == 5:
            name = name.strip()
            name = '*' + name
            name = name.rjust(max_name_length)
            padded_team_names[index] = name

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
