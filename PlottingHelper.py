import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text


def plot_elo_function(teams,
                      plot_name,
                      sub_dir_name,
                      absolute=False,
                      classic_colors=False,
                      show_plot=True,
                      save_dir='..\\Projects\\nfl\\NFL_Prediction\\2019Ratings\\'):
    """
    Plots the rating of a set of teams. X axis indicates each team's object Elo rating. Y axis indicates each team's
    percentile relative to the other teams in the league.  Plot conatins each rating's tier overlaid by a logit function
    where teams fall.

    :param teams: The list of teams to include in the plot
    :param plot_name: The name of the plot
    :param sub_dir_name: The subdirectory to save the plot to
    :param absolute: If the logit function should be based off all NFL teams in history
    :param classic_colors: If the 'classic' color scheme should be used
    :param show_plot: If the plot should be shown or just saved
    :param save_dir: The directory to save the plot to
    :return: Void
    """

    # Set the style
    sns.set(style="ticks")

    # Get the elo rating of each team
    actual_elos = [round(team[4]) for team in teams]

    # Determine the rating scale
    if absolute:
        avg_elo = 1500
        elo_dev = 82.889
    else:
        avg_elo = statistics.mean(actual_elos)
        elo_dev = statistics.pstdev(actual_elos)

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

    # Get the range for the curve and the location of each team
    vals = range(bottom, top)
    markers = [vals.index(elo) for elo in actual_elos]
    vals = np.arange(bottom, top)

    # Get the values that match the boundaries of the tiers
    s_plus = avg_elo + elo_dev * 8 / 3
    s = avg_elo + elo_dev * 7 / 3
    s_minus = avg_elo + elo_dev * 2
    a_plus = avg_elo + elo_dev * 5 / 3
    a = avg_elo + elo_dev * 4 / 3
    a_minus = avg_elo + elo_dev
    b_plus = avg_elo + elo_dev * 2 / 3
    b = avg_elo + elo_dev * 1 / 3
    b_minus = avg_elo
    c_plus = avg_elo - elo_dev * 1 / 3
    c = avg_elo - elo_dev * 2 / 3
    c_minus = avg_elo - elo_dev
    d_plus = avg_elo - elo_dev * 4 / 3
    d = avg_elo - elo_dev * 5 / 3
    d_minus = avg_elo - elo_dev * 2
    f_plus = avg_elo - elo_dev * 7 / 3
    f = avg_elo - elo_dev * 8 / 3
    f_minus = avg_elo - elo_dev * 3

    # Set the colors of the tiers
    if classic_colors:
        s_plus_color = '#FFFFF0'
        s_color = '#F5F5DC'
        s_minus_color = '#FAEBD7'
        a_plus_color = '#EE82EE'
        a_color = '#DA70D6'
        a_minus_color = '#6495ED'
        b_plus_color = '#4169E1'
        b_color = '#00FFFF'
        b_minus_color = '#008B8B'
        c_plus_color = '#ADFF2F'
        c_color = '#FFFF00'
        c_minus_color = '#FFA500'
        d_plus_color = '#FF7F50'
        d_color = '#FF6347'
        d_minus_color = '#A9A9A9'
        f_plus_color = '#808080'
        f_color = '#606060'
        f_minus_color = '#404040'
        bottom_color = '#000000'
    else:
        s_plus_color = '#A64CA6'
        s_color = '#800080'
        s_minus_color = '#660099'
        a_plus_color = '#3300CC'
        a_color = '#0000FF'
        a_minus_color = '#0019CC'
        b_plus_color = '#004C66'
        b_color = '#008000'
        b_minus_color = '#4CA600'
        c_plus_color = '#B2D800'
        c_color = '#FFFF00'
        c_minus_color = '#FFE400'
        d_plus_color = '#FFC000'
        d_color = '#FFA500'
        d_minus_color = '#FF7300'
        f_plus_color = '#FF3100'
        f_color = '#FF0000'
        f_minus_color = '#B20000'
        bottom_color = '#000000'

    # Plot the CDF and mark each team
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(vals, cdf(vals), 'k', markevery=markers, marker='|')

    # Color each section by the tier color
    ax.axvspan(bottom, f_minus, alpha=0.6, color=bottom_color)

    ax.axvspan(f_minus, f, alpha=0.6, color=f_minus_color)
    ax.annotate(s='F-', xy=((f_minus + f) / 2 - 3, -2))

    ax.axvspan(f, f_plus, alpha=0.6, color=f_color)
    ax.annotate(s='F', xy=((f + f_plus) / 2 - 3, -2))

    ax.axvspan(f_plus, d_minus, alpha=0.6, color=f_plus_color)
    ax.annotate(s='F+', xy=((f_plus + d_minus) / 2 - 3, -2))

    ax.axvspan(d_minus, d, alpha=0.5, color=d_minus_color)
    ax.annotate(s='D-', xy=((d_minus + d) / 2 - 3, -2))

    ax.axvspan(d, d_plus, alpha=0.5, color=d_color)
    ax.annotate(s='D', xy=((d + d_plus) / 2 - 3, -2))

    ax.axvspan(d_plus, c_minus, alpha=0.5, color=d_plus_color)
    ax.annotate(s='D+', xy=((d_plus + c_minus) / 2 - 3, -2))

    ax.axvspan(c_minus, c, alpha=0.5, color=c_minus_color)
    ax.annotate(s='C-', xy=((c_minus + c) / 2 - 3, -2))

    ax.axvspan(c, c_plus, alpha=0.5, color=c_color)
    ax.annotate(s='C', xy=((c + c_plus) / 2 - 3, -2))

    ax.axvspan(c_plus, b_minus, alpha=0.5, color=c_plus_color)
    ax.annotate(s='C+', xy=((c_plus + b_minus) / 2 - 3, -2))

    ax.axvspan(b_minus, b, alpha=0.5, color=b_minus_color)
    ax.annotate(s='B-', xy=((b_minus + b) / 2 - 3, -2))

    ax.axvspan(b, b_plus, alpha=0.5, color=b_color)
    ax.annotate(s='B', xy=((b + b_plus) / 2 - 3, -2))

    ax.axvspan(b_plus, a_minus, alpha=0.5, color=b_plus_color)
    ax.annotate(s='B+', xy=((b_plus + a_minus) / 2 - 3, -2))

    ax.axvspan(a_minus, a, alpha=0.5, color=a_minus_color)
    ax.annotate(s='A-', xy=((a_minus + a) / 2 - 3, -2))

    ax.axvspan(a, a_plus, alpha=0.5, color=a_color)
    ax.annotate(s='A', xy=((a + a_plus) / 2 - 3, -2))

    ax.axvspan(a_plus, s_minus, alpha=0.5, color=a_plus_color)
    ax.annotate(s='A+', xy=((a_plus + s_minus) / 2 - 3, -2))

    ax.axvspan(s_minus, s, alpha=0.5, color=s_minus_color)
    ax.annotate(s='S-', xy=((s_minus + s) / 2 - 3, -2))

    ax.axvspan(s, s_plus, alpha=0.5, color=s_color)
    ax.annotate(s='S', xy=((s + s_plus) / 2 - 3, -2))

    ax.axvspan(s_plus, s_plus + elo_dev / 3, alpha=0.5, color=s_plus_color)
    ax.annotate(s='S+', xy=((s_plus + s_plus + elo_dev / 3) / 2 - 3, -2))

    # Label each teams marker with the team name
    team_name_labels = list()
    for i, percent in enumerate(percents):
        if i % 2 == 0:
            offset = (20, -5)
        else:
            chars = min(len(team_names[i]), 8)
            chars = max(chars, 6)
            offset = (chars * -10, 2)
        team_name_labels.append(ax.annotate(s=team_names[i],
                                            xy=(actual_elos[i], percent),
                                            xytext=offset,
                                            textcoords='offset points',
                                            arrowprops=dict(arrowstyle='-',
                                                            color='black')))

    # Adjust name labels so they don't overlap
    adjust_text(texts=team_name_labels,
                autoalign='xy',
                force_text=.01,
                only_move={'text': 'xy'})

    # Add titles
    if plot_name:
        plot_name = plot_name + ' '
    ax.set_title('Elo Ratings: ' + plot_name + sub_dir_name)
    ax.set_xlabel('Elo Rating')
    if absolute:
        ax.set_ylabel('Absolute Percentile')
    else:
        ax.set_ylabel('Relative Percentile')

    # Add the tick marks
    bottom_remainder = 50 - (bottom % 50)
    plt.xticks(range(bottom + bottom_remainder, top + bottom_remainder + 1, 50))
    plt.yticks(range(0, 101, 10))

    # Remove the x margins
    plt.margins(x=0)

    # Change the name of the subdirectory
    sub_dir_name = sub_dir_name.replace(' ', '_')

    # If the plot is being saved
    if save_dir:

        # Determine the directory
        if absolute:
            save_dir = save_dir + 'Absolute\\' + sub_dir_name + '\\'
        else:
            save_dir = save_dir + 'Relative\\' + sub_dir_name + '\\'

        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Get the file name
        if plot_name:
            plot_name = plot_name.strip().replace(' ', '_')
            file_name = save_dir + 'Elo_Ratings_' + plot_name + '.png'
        else:
            file_name = save_dir + 'Elo_Ratings.png'

        # Save the figure
        plt.savefig(file_name, dpi=300)

    # Display the plot if desired
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_conference_elo_function(teams, conference_name, week_name, absolute=False, classic_colors=False):
    """
    Plots the logit function for each team's rating for teams in a given conference.

    :param teams: The list of teams to plot
    :param conference_name: The name of the conference
    :param week_name: The week that the plot is for
    :param absolute: If the logit function should be based off all NFL teams in history
    :param classic_colors: If the 'classic' color scheme should be used
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs

    # Get the teams in the conference
    conference_teams = list()
    league = Playoffs.get_league_structure()
    for conf_name, conference in league.items():
        if conference_name == conf_name:
            for div_name, division in conference.items():
                conference_teams.extend(division)

    # Plot the elo function for each team in the conference
    teams = list(filter(lambda t: t[0] in conference_teams, teams))
    plot_elo_function(teams,
                      conference_name,
                      week_name,
                      absolute=absolute,
                      classic_colors=classic_colors,
                      show_plot=False)


def plot_division_elo_function(teams, division_name, week_name, absolute=False, classic_colors=False):
    """
    Plots the logit function for each team's rating for teams in a given division.

    :param teams: The list of teams to plot
    :param division_name: The name of the division
    :param week_name: The week that the plot is for
    :param absolute: If the logit function should be based off all NFL teams in history
    :param classic_colors: If the 'classic' color scheme should be used
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs

    # Get the teams in the division
    division_teams = list()
    league = Playoffs.get_league_structure()
    for conf_name, conference in league.items():
        for div_name, division in conference.items():
            if div_name == division_name:
                division_teams = division

    # Plot the elo function for each team in the division
    teams = list(filter(lambda t: t[0] in division_teams, teams))
    plot_elo_function(teams,
                      division_name,
                      week_name,
                      absolute=absolute,
                      classic_colors=classic_colors,
                      show_plot=False)


def plot_team_elo_over_season(title, team_names, show_plot=True):
    """
    Plots the change in elo over the season for a set of teams.

    :param title: The title that the team elos are for
    :param team_names: The names of the teams that are included in the plot
    :param show_plot: If the plot should be shown or just saved
    :return: Void
    """

    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
    import Projects.nfl.NFL_Prediction.NFLSeason2019 as Season

    # Set the style
    sns.set(style="ticks")

    # Get all the teams in the league
    teams = Season.nfl_teams

    # For each team in the list of teams
    all_teams_games = list()
    for team_name in team_names:
        # Get all the games each team has completed
        team_games = Playoffs.completed_games.loc[(Playoffs.completed_games['home_team'] == team_name) |
                                                  (Playoffs.completed_games['away_team'] == team_name)]

        # Add each completed game to a list
        all_teams_games.append(team_games)

    # Get the maximum number of games any team has played in
    max_len = max([len(team_games) for team_games in all_teams_games])

    # Fill out data rows
    teams_elos = pd.DataFrame(columns=team_names)

    # For each team in the list of teams
    for team_name in team_names:
        # Get all the games each team has completed
        team_games = Playoffs.completed_games.loc[(Playoffs.completed_games['home_team'] == team_name) |
                                                  (Playoffs.completed_games['away_team'] == team_name)]

        # Get the elo for the team for each game
        home = pd.Series(team_games['home_team'] == team_name)
        elos = pd.Series(team_games.lookup(team_games.index, home.map({True: 'home_elo', False: 'away_elo'})))

        # Get the team from its name
        team = Season.get_team(teams, team_name)

        # Get and add the teams current elo
        current = pd.Series(team[4])
        elos = elos.append(current, ignore_index=True)

        # Store the team and its elo history in a dictionary
        teams_elos[team_name] = elos

    # Extend the line another week
    data = pd.DataFrame(columns=team_names)

    # For each team in the list of teams
    for team_name in team_names:

        # Get the elo history for the team
        elos = teams_elos[team_name]

        # Add 2 more weeks of the most recent elo for formatting
        while len(elos) < max_len + 2:
            last = pd.Series(elos.iloc[-1])
            elos = elos.append(last, ignore_index=True)

        # Update the data frame with the teams elo history
        data[team_name] = elos

    # Plot the data frame
    ax = data.plot.line(figsize=(20, 10))

    # Add titles
    ax.set_title('Team Elos: ' + title)
    ax.set_xlabel('Week')
    ax.set_ylabel('Elo')

    # Add the tick marks
    plt.xticks(range(max_len + 1))

    # Remove the x margins
    plt.margins(x=0)

    # Get the min an max y values
    min_elo = min([team[4] for team in teams])
    max_elo = max([team[4] for team in teams])
    plt.ylim(min_elo - 15, max_elo + 15)

    # Save the figure
    save_name = title.replace(' ', '_')
    plt.savefig('..\\Projects\\nfl\\NFL_Prediction\\2019Ratings\\Trends\\' + save_name + '.png', dpi=300)

    # Display the plot if desired
    if show_plot:
        plt.show()
    else:
        plt.close()
