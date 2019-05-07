import statistics

import matplotlib.pyplot as plt
import maya
import numpy as np

import Projects.nfl.NFL_Prediction.NFLPredictor as Predictor
import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
import Projects.nfl.NFL_Prediction.StandingsHelper as Standings


def season():
    # Setup
    teams = set_up_teams()
    eliminated_teams = list()

    # Preseason
    print('Preseason')
    teams = handle_week(teams, 'Preseason Week 1', pre_week_1, eliminated_teams, '6 August 2019')
    teams = handle_week(teams, 'Preseason Week 2', pre_week_2, eliminated_teams, '13 August 2019')
    teams = handle_week(teams, 'Preseason Week 3', pre_week_3, eliminated_teams, '20 August 2019')
    handle_week(teams, 'Preseason Week 4', pre_week_4, eliminated_teams, '27 August 2019')

    # Reset teams after preseason
    teams = set_up_teams()

    # Preseason Info
    Standings.print_league_details(teams, eliminated_teams)
    Standings.print_schedule_difficulty(teams)
    print('*' * 120, '\n')

    # Regular Season
    print('Regular Season')
    teams = handle_week(teams, 'Week 1', week_1, eliminated_teams, '3 September 2019')

    teams = handle_week(teams, 'Week 2', week_2, eliminated_teams, '10 September 2019')

    teams = handle_week(teams, 'Week 3', week_3, eliminated_teams, '17 September 2019')

    teams = handle_week(teams, 'Week 4', week_4, eliminated_teams, '24 September 2019')

    teams = handle_week(teams, 'Week 5', week_5, eliminated_teams, '1 October 2019')

    teams = handle_week(teams, 'Week 6', week_6, eliminated_teams, '8 October 2019')

    teams = handle_week(teams, 'Week 7', week_7, eliminated_teams, '15 October 2019')

    teams = handle_week(teams, 'Week 8', week_8, eliminated_teams, '22 October 2019')

    teams = handle_week(teams, 'Week 9', week_9, eliminated_teams, '29 October 2019')

    teams = handle_week(teams, 'Week 10', week_10, eliminated_teams, '5 November 2019')

    teams = handle_week(teams, 'Week 11', week_11, eliminated_teams, '12 November 2019')

    teams = handle_week(teams, 'Week 12', week_12, eliminated_teams, '19 November 2019')
    eliminated_teams.extend([])

    teams = handle_week(teams, 'Week 13', week_13, eliminated_teams, '26 November 2019')

    teams = handle_week(teams, 'Week 14', week_14, eliminated_teams, '3 December 2019')

    teams = handle_week(teams, 'Week 15', week_15, eliminated_teams, '10 December 2019')

    teams = handle_week(teams, 'Week 16', week_16, eliminated_teams, '17 December 2019')

    teams = handle_week(teams, 'Week 17', week_17, eliminated_teams, '24 December 2019')

    # Regular Season Info
    Standings.print_schedule_difficulty(teams, remaining=True)

    # Playoff Info
    afc_playoff_teams, nfc_playoff_teams = Playoffs.get_playoff_picture(teams)
    print('Current Playoff Picture')
    print('-' * 15 + 'AFC' + '-' * 15)
    Playoffs.create_playoff_bracket(afc_playoff_teams)
    print()
    print('-' * 15 + 'NFC' + '-' * 15)
    Playoffs.create_playoff_bracket(nfc_playoff_teams)
    print()
    print('*' * 120, '\n')

    # Playoffs
    print('Playoffs')
    teams = handle_week(teams, 'Wildcard Weekend', wildcard, eliminated_teams, '31 December 2019')

    teams = handle_week(teams, 'Divisional Round', divisional, eliminated_teams, '7 January 2020')

    teams = handle_week(teams, 'Conference Finals', conference, eliminated_teams, '14 January 2020')

    teams = handle_week(teams, 'Superbowl', superbowl, eliminated_teams, '28 January 2020')

    # Final Outcome
    Playoffs.monte_carlo(teams)
    plot_elo_function(teams)


def set_up_teams():
    def create_base_team(team_name, elo):
        return team_name, 0, 0, 0, elo, 0, 0, 0, 0, 0, 0, 0, 0, 0

    teams = list()
    teams.append(create_base_team('49ers', 1431.803))
    teams.append(create_base_team('Bears', 1547.207))
    teams.append(create_base_team('Bengals', 1450.022))
    teams.append(create_base_team('Bills', 1467.667))
    teams.append(create_base_team('Broncos', 1447.034))
    teams.append(create_base_team('Browns', 1465.156))
    teams.append(create_base_team('Buccaneers', 1437.01))
    teams.append(create_base_team('Cardinals', 1414.907))
    teams.append(create_base_team('Chargers', 1566.415))
    teams.append(create_base_team('Chiefs', 1562.643))
    teams.append(create_base_team('Colts', 1533.559))
    teams.append(create_base_team('Cowboys', 1555.546))
    teams.append(create_base_team('Dolphins', 1458.772))
    teams.append(create_base_team('Eagles', 1561.189))
    teams.append(create_base_team('Falcons', 1501.919))
    teams.append(create_base_team('Giants', 1442.666))
    teams.append(create_base_team('Jaguars', 1447.748))
    teams.append(create_base_team('Jets', 1408.145))
    teams.append(create_base_team('Lions', 1476.841))
    teams.append(create_base_team('Packers', 1463.462))
    teams.append(create_base_team('Panthers', 1486.623))
    teams.append(create_base_team('Patriots', 1608.682))
    teams.append(create_base_team('Raiders', 1433.093))
    teams.append(create_base_team('Rams', 1589.463))
    teams.append(create_base_team('Ravens', 1537.051))
    teams.append(create_base_team('Redskins', 1465.572))
    teams.append(create_base_team('Saints', 1592.439))
    teams.append(create_base_team('Seahawks', 1535.119))
    teams.append(create_base_team('Steelers', 1544.046))
    teams.append(create_base_team('Texans', 1515.959))
    teams.append(create_base_team('Titans', 1520.108))
    teams.append(create_base_team('Vikings', 1532.134))
    return teams


def handle_week(teams,
                week_name,
                week,
                eliminated_teams,
                week_start_date,
                full_standings=True,
                divisional_view=False,
                model_rankings=False):
    # Get the starting time of the week
    week_start = maya.when(week_start_date, timezone='US/Central')

    # If the week has started
    if maya.now() >= week_start:
        # Get the ending time of the week
        week_end_date = week_start.add(weeks=1)

        # Print the week name
        print(week_name)

        # Handle the week
        teams = week(teams, week_end_date)

        # Print the tables
        Standings.print_league_details(teams,
                                       eliminated_teams,
                                       full_standings=full_standings,
                                       divisional_view=divisional_view,
                                       model_rankings=model_rankings)

        # Print a separator
        print('*' * 120, '\n')

        # If the week has not ended
        if maya.now() < week_end_date:
            # Plot the team's percentiles based on their elo
            plot_elo_function(teams, week_name)
            plot_elo_function(teams, week_name, absolute=True)

    # Return the updated teams
    return teams


def plot_elo_function(teams, week_name, absolute=False):
    # Get the elo rating of each team
    actual_elos = [round(team[4]) for team in teams]
    avg_elo = statistics.mean(actual_elos)

    if absolute:
        elo_dev = 82.889
    else:
        elo_dev = statistics.pstdev(actual_elos)
    elo_dev_third = statistics.pstdev(actual_elos) / 3

    # Cumulative distribution function for a normal deviation
    def cdf(rating):
        from scipy.stats import norm
        adj_rating = (rating - avg_elo) / elo_dev
        x = norm.cdf(adj_rating)
        return x * 100

    # Get each teams name and percentile
    team_names = [team[0] for team in teams]
    percents = [cdf(rating) for rating in actual_elos]

    # Sort the names, ratings and percentiles by the rating
    zipped = list(zip(team_names, actual_elos, percents))
    zipped.sort(key=lambda tup: tup[1])
    team_names, actual_elos, percents = zip(*zipped)

    # Plot everywhere within 3.89 standard deviations of the mean (99.99% coverage)
    bottom = round(avg_elo - 3.89 * elo_dev)
    top = round(avg_elo + 3.89 * elo_dev)

    # Get the values that match the boundaries of the tiers
    s_plus = avg_elo + elo_dev_third * 8
    s = avg_elo + elo_dev_third * 7
    s_minus = avg_elo + elo_dev_third * 6
    a_plus = avg_elo + elo_dev_third * 5
    a = avg_elo + elo_dev_third * 4
    a_minus = avg_elo + elo_dev_third * 3
    b_plus = avg_elo + elo_dev_third * 2
    b = avg_elo + elo_dev_third * 1
    b_minus = avg_elo
    c_plus = avg_elo - elo_dev_third * 1
    c = avg_elo - elo_dev_third * 2
    c_minus = avg_elo - elo_dev_third * 3
    d_plus = avg_elo - elo_dev_third * 4
    d = avg_elo - elo_dev_third * 5
    d_minus = avg_elo - elo_dev_third * 6
    f_plus = avg_elo - elo_dev_third * 4
    f = avg_elo - elo_dev_third * 5
    f_minus = avg_elo - elo_dev_third * 6

    # Get the range for the curve and the location of each team
    vals = range(bottom, top)
    markers = [vals.index(elo) for elo in actual_elos]
    vals = np.arange(bottom, top)

    # Plot the CDF and mark each team
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(vals, cdf(vals), 'k', markevery=markers, marker='|')

    # Color each section by the tier color
    ax.axvspan(bottom, f_minus, alpha=0.6, color='#000000')
    ax.axvspan(f_minus, f, alpha=0.6, color='#696969')
    ax.axvspan(f, f_plus, alpha=0.6, color='#808080')
    ax.axvspan(f_plus, d_minus, alpha=0.6, color='#A9A9A9')
    ax.axvspan(d_minus, d, alpha=0.5, color='#C0C0C0')
    ax.axvspan(d, d_plus, alpha=0.5, color='#FF7F50')
    ax.axvspan(d_plus, c_minus, alpha=0.5, color='#F08080')
    ax.axvspan(c_minus, c, alpha=0.5, color='#FFA500')
    ax.axvspan(c, c_plus, alpha=0.5, color='#FFFF00')
    ax.axvspan(c_plus, b_minus, alpha=0.5, color='#ADFF2F')
    ax.axvspan(b_minus, b, alpha=0.5, color='#008B8B')
    ax.axvspan(b, b_plus, alpha=0.5, color='#00FFFF')
    ax.axvspan(b_plus, a_minus, alpha=0.5, color='#4169E1')
    ax.axvspan(a_minus, a, alpha=0.5, color='#6495ED')
    ax.axvspan(a, a_plus, alpha=0.5, color='#DA70D6')
    ax.axvspan(a_plus, s_minus, alpha=0.5, color='#EE82EE')
    ax.axvspan(s_minus, s, alpha=0.5, color='#F5F5DC')
    ax.axvspan(s, s_plus, alpha=0.5, color='#FFFFF0')
    ax.axvspan(s_plus, top, alpha=0.5, color='#FFFFFF')

    # Label each teams marker with the team name
    for i, percent in enumerate(percents):
        if i % 2 == 0:
            offset = (30, -5)
        else:
            offset = (-60, 0)
        ax.annotate(s=team_names[i],
                    xy=(actual_elos[i], percent),
                    xytext=offset,
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='-'))

    # Add titles
    ax.set_title('Elo Ratings: ' + week_name)
    ax.set_xlabel('Elo Rating')
    if absolute:
        ax.set_ylabel('Absolute Percentile')
    else:
        ax.set_ylabel('Relative Percentile')

    # Plot
    week_name = week_name.replace(' ', '_')

    if absolute:
        plt.savefig('..\\Projects\\nfl\\NFL_Prediction\\2019Ratings\\Absolute\\Elo_Ratings_' + week_name + '.png',
                    dpi=300)
    else:
        plt.savefig('..\\Projects\\nfl\\NFL_Prediction\\2019Ratings\\Relative\\Elo_Ratings_' + week_name + '.png',
                    dpi=300)
    plt.show()


def pre_week_1(teams, week_end_date):
    # If the week has not ended
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_pre_week1_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Giants',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Jets',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def pre_week_2(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_pre_week2_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Giants',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Bears',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def pre_week_3(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_pre_week3_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Cowboys',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Texans',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def pre_week_4(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_pre_week4_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Cowboys',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Buccaneers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_1(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week1_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Packers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_2(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week2_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Panthers',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Buccaneers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_3(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week3_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Jaguars',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Titans',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_4(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week4_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Packers',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Eagles',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_5(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week5_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Seahawks',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Rams',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_6(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week6_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Patriots',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Giants',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    return teams


def week_7(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week7_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Broncos',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Chiefs',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_8(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week8_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Vikings',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Redskins',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_9(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week9_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Cardinals',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='49ers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_10(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week10_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Raiders',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Chargers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_11(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week11_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Browns',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Steelers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_12(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week12_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Texans',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Colts',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bucanneerrs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_13(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week13_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Lions',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Bears',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_14(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week14_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Cowboys',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_15(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week15_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Ravens',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Jets',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_16(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week16_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Broncos',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Lions',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Steelers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Ravens', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def week_17(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_week17_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Ravens',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Steelers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Buccaneers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Falcons', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Giants', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Saints', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Bengals', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Browns', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Cowboys', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Lions', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Packers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Texans', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Titans', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jaguars', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Colts', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Chiefs', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Chargers', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Vikings', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bears', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Patriots', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Dolphins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Broncos', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Raiders', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Cardinals', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Seahawks', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          '49ers', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def wildcard(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_wildcard_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Packers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def divisional(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_divisional_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Packers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Eagles', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Redskins', 0, 0, 0, 0, 0, 0, 0, 0)
    #
    # teams = set_game_outcome(teams,
    #                          'Jets', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Bills', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def conference(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_conference_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Packers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)
    #
    # teams = set_game_outcome(teams,
    #                          'Panthers', 0, 0, 0, 0, 0, 0, 0, 0,
    #                          'Rams', 0, 0, 0, 0, 0, 0, 0, 0)

    return teams


def superbowl(teams, week_end_date):
    if maya.now() < week_end_date:
        Predictor.get_week_probabilities(teams, Playoffs.get_superbowl_schedule())

    # Results
    # teams = set_game_outcome(teams,
    #                          home_name='Bears',
    #                          home_score=0,
    #                          home_touchdowns=0,
    #                          home_net_pass_yards=0,
    #                          home_pass_completions=0,
    #                          home_pass_attempts=0,
    #                          home_pass_tds=0,
    #                          home_interceptions_thrown=0,
    #                          home_total_yards=0,
    #                          away_name='Packers',
    #                          away_score=0,
    #                          away_touchdowns=0,
    #                          away_net_pass_yards=0,
    #                          away_pass_completions=0,
    #                          away_pass_attempts=0,
    #                          away_pass_tds=0,
    #                          away_interceptions_thrown=0,
    #                          away_total_yards=0)

    return teams


def get_team(teams, team_name):
    for team in teams:
        if team[0] == team_name:
            return team


def set_game_outcome(teams,
                     home_name, home_score, home_touchdowns, home_net_pass_yards, home_pass_completions,
                     home_pass_attempts, home_pass_tds, home_interceptions_thrown, home_total_yards,
                     away_name, away_score, away_touchdowns, away_net_pass_yards, away_pass_completions,
                     away_pass_attempts, away_pass_tds, away_interceptions_thrown, away_total_yards):
    teams = Predictor.update_teams(teams,
                                   home_name, home_score, home_touchdowns, home_net_pass_yards, home_pass_completions,
                                   home_pass_attempts, home_pass_tds, home_interceptions_thrown, home_total_yards,
                                   away_name, away_score, away_touchdowns, away_net_pass_yards, away_pass_completions,
                                   away_pass_attempts, away_pass_tds, away_interceptions_thrown, away_total_yards)

    Playoffs.completed_games.append((home_name, home_score, away_name, away_score))

    return teams
