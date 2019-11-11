import networkx as nx

nfl = nx.MultiDiGraph()


def create_games_graph():
    """
    Creates a graph for storing team games and data.

    :return: The graph with the league info for the season
    """

    global nfl

    nfl.add_nodes_from(create_all_nfl_teams())


def create_all_nfl_teams():
    """
    Initializes a list of dicts for the teams in the league.  Each dict represents a team.

    :return: A list of team representations
    """

    teams = list()
    teams.append(create_team_node('Bengals', 'AFC', 'North', elo=1459.57, colors=['#FB4F14', '#000000']))
    teams.append(create_team_node('Browns', 'AFC', 'North', elo=1514.419, colors=['#311D00', '#FF3C00']))
    teams.append(create_team_node('Ravens', 'AFC', 'North', elo=1534.302, colors=['#241773', '#9E7C0C', '#000000']))
    teams.append(create_team_node('Steelers', 'AFC', 'North', elo=1542.941, colors=['#FFB612', '#101820']))

    teams.append(create_team_node('Bills', 'AFC', 'East', elo=1462.998, colors=['#00338D', '#C60C30']))
    teams.append(create_team_node('Dolphins', 'AFC', 'East', elo=1380.449, colors=['#008E97', '#FC4C02', '#005778']))
    teams.append(create_team_node('Jets', 'AFC', 'East', elo=1421.038, colors=['#125740', '#000000']))
    teams.append(create_team_node('Patriots', 'AFC', 'East', elo=1593.129, colors=['#002244', '#C60C30', '#B0B7BC']))

    teams.append(create_team_node('Colts', 'AFC', 'South', elo=1470.814, colors=['#002C5F', '#A2AAAD']))
    teams.append(create_team_node('Jaguars', 'AFC', 'South', elo=1471.362, colors=['#D7A22A', '#006778', '#101820']))
    teams.append(create_team_node('Texans', 'AFC', 'South', elo=1528.406, colors=['#03202F', '#A71930']))
    teams.append(create_team_node('Titans', 'AFC', 'South', elo=1489.737, colors=['#0C2340', '#418FDE', '#C8102E']))

    teams.append(create_team_node('Broncos', 'AFC', 'West', elo=1506.329, colors=['#FB4F14', '#002244']))
    teams.append(create_team_node('Chargers', 'AFC', 'West', elo=1570.778, colors=['#FFC20E', '#002A5E', '#0080C6']))
    teams.append(create_team_node('Chiefs', 'AFC', 'West', elo=1577.908, colors=['#E31837', '#FFB81C']))
    teams.append(create_team_node('Raiders', 'AFC', 'West', elo=1410.342, colors=['#A5ACAF', '#000000']))

    teams.append(create_team_node('Bears', 'NFC', 'North', elo=1557.476, colors=['#0B162A', '#C83803']))
    teams.append(create_team_node('Lions', 'NFC', 'North', elo=1459.433, colors=['#0076B6', '#B0B7BC']))
    teams.append(create_team_node('Packers', 'NFC', 'North', elo=1532.109, colors=['#203731', '#FFB612']))
    teams.append(create_team_node('Vikings', 'NFC', 'North', elo=1531.423, colors=['#4F2683', '#FFC62F']))

    teams.append(create_team_node('Cowboys', 'NFC', 'East', elo=1528.68, colors=['#041E42', '#869397']))
    teams.append(create_team_node('Eagles', 'NFC', 'East', elo=1545.958, colors=['#004C54', '#A5ACAF']))
    teams.append(create_team_node('Giants', 'NFC', 'East', elo=1460.393, colors=['#0B2265', '#A71930', '#A5ACAF']))
    teams.append(create_team_node('Redskins', 'NFC', 'East', elo=1430.362, colors=['#773141', '#FFB612']))

    teams.append(create_team_node('Buccaneers', 'NFC', 'South', elo=1474.242, colors=['#D50A0A', '#0A0A08', '#FF7900']))
    teams.append(create_team_node('Falcons', 'NFC', 'South', elo=1474.105, colors=['#A71930', '#000000', '#A5ACAF']))
    teams.append(create_team_node('Panthers', 'NFC', 'South', elo=1509.894, colors=['#0085CA', '#101820', '#BFC0BF']))
    teams.append(create_team_node('Saints', 'NFC', 'South', elo=1587.095, colors=['#D3BC8D', '#101820']))

    teams.append(create_team_node('49ers', 'NFC', 'West', elo=1469.443, colors=['#AA0000', '#B3995D']))
    teams.append(create_team_node('Cardinals', 'NFC', 'West', elo=1397.315, colors=['#97233F', '#000000', '#FFB612']))
    teams.append(create_team_node('Rams', 'NFC', 'West', elo=1571.6, colors=['#002244', '#866D4B']))
    teams.append(create_team_node('Seahawks', 'NFC', 'West', elo=1535.948, colors=['#002244', '#69BE28', '#A5ACAF']))

    return teams


def create_team_node(name,
                     conference,
                     division,
                     wins=0,
                     losses=0,
                     ties=0,
                     elo=1500.0,
                     points_for=0.0,
                     points_against=0.0,
                     touchdowns=0.0,
                     net_pass_yards=0.0,
                     pass_completions=0.0,
                     pass_attempts=0.0,
                     pass_touchdowns=0.0,
                     interceptions_thrown=0.0,
                     total_yards=0.0,
                     colors=None):
    """
    Creates an individual dict containing the team information.

    :param name: The name of the team, this is the label of the node
    :param conference: The team's conference
    :param division: The team's division
    :param wins: The number of wins the team has
    :param losses: The number of losses the team has
    :param ties: The number of ties the team has
    :param elo: The elo of the team
    :param points_for: The average number of points the team has scored
    :param points_against: The average number of points the team has allowed
    :param touchdowns: The average number of offensive touchdowns the team has scored
    :param net_pass_yards: The average net passing yards of the team
    :param pass_completions: The average number of pass completions of the team
    :param pass_attempts: The average number of pass attempts of the team
    :param pass_touchdowns: The average number of passing touchdowns of the team
    :param interceptions_thrown: The average number of interceptions thrown by the team
    :param total_yards: The average number of total yards of the team
    :param colors: Array of the hex values of the team's colors
    :return: A tuple with the team name and a dict for the team info
    """

    team_dict = dict()

    team_dict['Division'] = division
    team_dict['Conference'] = conference
    team_dict['Wins'] = wins
    team_dict['Losses'] = losses
    team_dict['Ties'] = ties
    team_dict['Elo'] = elo
    team_dict['Points For'] = points_for
    team_dict['Points Against'] = points_against
    team_dict['Touchdowns'] = touchdowns
    team_dict['Net Passing Yards'] = net_pass_yards
    team_dict['Pass Completions'] = pass_completions
    team_dict['Pass Attempts'] = pass_attempts
    team_dict['Passing Touchdowns'] = pass_touchdowns
    team_dict['Interceptions Thrown'] = interceptions_thrown
    team_dict['Total Yards'] = total_yards
    team_dict['Primary'] = colors[0] if colors else '#000000'
    team_dict['Secondary'] = colors[1] if len(colors) > 1 else '#000000'
    team_dict['Tertiary'] = colors[2] if len(colors) > 2 else '#000000'

    return name, team_dict


def set_game_outcome(teams, home_name, home_score, away_name, away_score):
    """
    Updates the game graph with the team information following the completion of a game. Updates individual nodes with
    current information, adds an edge from the winning team to the losing team.  Edge weight is the sum of the net
    points scored by the winning team in all of their victories over the losing team.

    :param teams: A list of all the teams in the league
    :param home_name: The name of the home team
    :param home_score: The score of the home team
    :param away_name: The name of the away team
    :param away_score: The score of the away team
    :return: Void
    """

    global nfl

    # Get the teams
    home_team = get_team(teams, home_name)
    away_team = get_team(teams, away_name)

    # Update each node
    home_team_dict = dict()
    home_team_dict['Wins'] = home_team[1]
    home_team_dict['Losses'] = home_team[2]
    home_team_dict['Ties'] = home_team[3]
    home_team_dict['Elo'] = home_team[4]
    home_team_dict['Points For'] = home_team[5]
    home_team_dict['Points Against'] = home_team[6]
    home_team_dict['Touchdowns'] = home_team[7]
    home_team_dict['Net Passing Yards'] = home_team[8]
    home_team_dict['Pass Completions'] = home_team[9]
    home_team_dict['Pass Attempts'] = home_team[10]
    home_team_dict['Passing Touchdowns'] = home_team[11]
    home_team_dict['Interceptions Thrown'] = home_team[12]
    home_team_dict['Total Yards'] = home_team[13]
    for key, val in home_team_dict.items():
        nx.set_node_attributes(nfl, {home_team[0]: val}, key)

    away_team_dict = dict()
    away_team_dict['Wins'] = away_team[1]
    away_team_dict['Losses'] = away_team[2]
    away_team_dict['Ties'] = away_team[3]
    away_team_dict['Elo'] = away_team[4]
    away_team_dict['Points For'] = away_team[5]
    away_team_dict['Points Against'] = away_team[6]
    away_team_dict['Touchdowns'] = away_team[7]
    away_team_dict['Net Passing Yards'] = away_team[8]
    away_team_dict['Pass Completions'] = away_team[9]
    away_team_dict['Pass Attempts'] = away_team[10]
    away_team_dict['Passing Touchdowns'] = away_team[11]
    away_team_dict['Interceptions Thrown'] = away_team[12]
    away_team_dict['Total Yards'] = away_team[13]
    for key, val in away_team_dict.items():
        nx.set_node_attributes(nfl, {away_team[0]: val}, key)

    # Add or update the edges based on the game outcome
    if home_score > away_score:
        nfl.add_edge(away_name, home_name, weight=(home_score - away_score))
    elif away_score > home_score:
        nfl.add_edge(home_name, away_name, weight=(away_score - home_score))


def page_rank_teams():
    """
    Calculates the pagerank of each team. Displays rankings with values multiplied by the number of nodes in the graph.

    :return: Void
    """

    global nfl

    ranks = nx.pagerank_numpy(nfl)
    return sorted(ranks.items(), key=lambda kv: kv[1], reverse=True)


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


def parity_clock():
    global nfl

    # Create graph copy with no duplicate directed edges
    nfl_copy = nfl.copy()
    nfl_copy.remove_edges_from(list(filter(lambda e: e[2] > 0, nfl.edges)))

    # Iterate over all cycles and find the one with the longest length
    longest_cycle_length = 0
    cycle = None
    for simple_cycle in nx.simple_cycles(nfl_copy):
        if len(simple_cycle) > longest_cycle_length:
            longest_cycle_length = len(simple_cycle)
            cycle = simple_cycle
            if longest_cycle_length == 30:
                break

    # If there are any cycles
    if cycle:
        print('Parity Clock')

        # If all 32 teams are in the cycle, it's a full parity clock
        if longest_cycle_length == 32:
            print('A full parity clock has been completed!')

        # Reverse the cycle direction
        cycle = list(reversed(cycle))

        # Add the starting team to the end to complete the loop
        cycle.append(cycle[0])

        # Format new lines if the length of the cycle is too long to print in one line
        if len(cycle) > 8:
            cycle[8] = '\n' + cycle[8]
        if len(cycle) > 16:
            cycle[16] = '\n' + cycle[16]
        if len(cycle) > 24:
            cycle[24] = '\n' + cycle[24]

        # Print the cycle
        print(' -> '.join(cycle))
        print()


def persist_graph():
    import json

    global nfl

    data = nx.node_link_data(nfl)
    with open('..\\Projects\\nfl\\NFL_Prediction\\2019Ratings\\LeagueGraph.json', 'w') as file:
        json.dump(data, file, indent=4, sort_keys=True)
