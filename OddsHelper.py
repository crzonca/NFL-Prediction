import statistics

import maya
import requests

domain = 'https://api.the-odds-api.com'


def get_odds():
    # Request the odds for the games
    params = {'apiKey': '60cf84589508c25d9471beafbf3c3201',
              'sport': 'americanfootball_nfl',
              'region': 'us',
              'mkt': 'spreads'}
    req = requests.get(domain + '/v3/odds/', params=params)

    # Get info on request usages remaining
    requests_used = req.headers.get('x-requests-used')
    requests_remaining = req.headers.get('x-requests-remaining')

    # Get the odds data
    resp = req.json()
    data = resp.get('data')

    # For each game in the odds data
    games = list()
    for game in data:
        # Get the teams
        teams = game.get('teams')
        favored_team = teams[0]
        underdog_team = teams[1]
        home_team = game.get('home_team')

        # Get the time of the game
        game_time = game.get('commence_time')
        game_time = maya.when(str(game_time), timezone='US/Central')

        # For eacch odds site
        sites = game.get('sites')
        home_spreads = list()
        for site in sites:
            # Get when the odds were last updated
            odds_updated_time = site.get('last_update')
            odds_updated_time = maya.when(str(odds_updated_time), timezone='US/Central')

            # If the odds werent updated in the last day
            if odds_updated_time < maya.now().add(days=-1):
                # Skip it
                continue

            # Get the point spreads for the game
            odds = site.get('odds')
            spreads = odds.get('spreads')
            points = spreads.get('points')

            # Get the spread from the home team perspective
            if favored_team == home_team:
                home_spread = points[0]
            else:
                home_spread = points[1]

            # Add the home team spread
            home_spreads.append(float(home_spread))

        if home_spreads:
            home_spread = statistics.mode(home_spreads)
        else:
            home_spread = 0

        games.append((teams, game_time, home_spread))
    return games
