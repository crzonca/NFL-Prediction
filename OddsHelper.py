import statistics

import maya
import requests

domain = 'https://api.the-odds-api.com'


def get_odds(week_end_date):
    # Request the odds for the games
    params = {'apiKey': '60cf84589508c25d9471beafbf3c3201',
              'sport': 'americanfootball_nfl',
              'region': 'us',
              'mkt': 'spreads'}
    req = requests.get(domain + '/v3/odds/', params=params)

    # Get info on request usages remaining
    requests_remaining = req.headers.get('x-requests-remaining')
    if int(requests_remaining) < 5:
        print(requests_remaining + ' Requests remaining')

    # Get the odds data
    resp = req.json()

    # Verify the status of getting the spread data
    if not resp.get('success'):
        raise Exception('Unable to fetch spread data')

    data = resp.get('data')

    # For each game in the odds data
    games = list()
    for game in data:
        # Get the teams
        teams = game.get('teams')
        first_team = teams[0]
        home_team = game.get('home_team')

        # Get the time of the game
        game_time = game.get('commence_time')
        game_time = maya.when(str(game_time), timezone='US/Central')

        # Skip games that are after the week end date
        if game_time > week_end_date:
            continue

        # For each odds site
        sites = game.get('sites')
        home_spreads = list()
        for site in sites:
            # Get when the odds were last updated
            odds_updated_time = site.get('last_update')
            odds_updated_time = maya.when(str(odds_updated_time), timezone='US/Central')

            # If the odds weren't updated in the last day
            if odds_updated_time < maya.now().add(days=-1):
                # Skip it
                continue

            # Get the point spreads for the game
            odds = site.get('odds')
            spreads = odds.get('spreads')
            points = spreads.get('points')

            # Get the spread from the home team perspective
            if first_team == home_team:
                home_spread = points[0]
            else:
                home_spread = points[1]

            # Add the home team spread
            home_spreads.append(float(home_spread))

        # Get the average home spread
        if home_spreads:
            home_spread = statistics.mean(home_spreads)
        else:
            home_spread = 0

        games.append((teams, game_time, home_spread))
    return games
