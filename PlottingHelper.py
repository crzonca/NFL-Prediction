import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_elo_function(teams,
                      plot_name,
                      sub_dir_name,
                      absolute=False,
                      classic_colors=False,
                      show_plot=True,
                      save_dir='..\\Projects\\nfl\\NFL_Prediction\\2019Ratings\\'):
    sns.set(style="ticks")

    # Get the elo rating of each team
    actual_elos = [round(team[4]) for team in teams]

    if absolute:
        avg_elo = 1500
        elo_dev = 82.889
    else:
        avg_elo = statistics.mean(actual_elos)
        elo_dev = statistics.pstdev(actual_elos)
    elo_dev_third = elo_dev / 3

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
    f_plus = avg_elo - elo_dev_third * 7
    f = avg_elo - elo_dev_third * 8
    f_minus = avg_elo - elo_dev_third * 9

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
    # fig.tight_layout()
    ax.plot(vals, cdf(vals), 'k', markevery=markers, marker='|')

    # Color each section by the tier color
    ax.axvspan(bottom, f_minus, alpha=0.6, color=bottom_color)
    # ax.annotate(s='F-', xy=((f_minus - elo_dev_third + f_minus) / 2 - 3, -2))

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

    ax.axvspan(s_plus, s_plus + elo_dev_third, alpha=0.5, color=s_plus_color)
    ax.annotate(s='S+', xy=((s_plus + s_plus + elo_dev_third) / 2 - 3, -2))

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
                    arrowprops=dict(arrowstyle='-',
                                    color='black'))

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

    # Plot
    sub_dir_name = sub_dir_name.replace(' ', '_')

    if save_dir:
        if absolute:
            save_dir = save_dir + 'Absolute\\' + sub_dir_name + '\\'
        else:
            save_dir = save_dir + 'Relative\\' + sub_dir_name + '\\'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if plot_name:
            plot_name = plot_name.strip().replace(' ', '_')
            file_name = save_dir + 'Elo_Ratings_' + plot_name + '.png'
        else:
            file_name = save_dir + 'Elo_Ratings.png'
        plt.savefig(file_name, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_conference_elo_function(teams, conference_name, week_name, absolute=False, classic_colors=False):
    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
    conference_teams = list()

    league = Playoffs.get_league_structure()
    for conf_name, conference in league.items():
        if conference_name == conf_name:
            for div_name, division in conference.items():
                conference_teams.extend(division)

    teams = list(filter(lambda t: t[0] in conference_teams, teams))
    plot_elo_function(teams,
                      conference_name,
                      week_name,
                      absolute=absolute,
                      classic_colors=classic_colors,
                      show_plot=False)


def plot_division_elo_function(teams, division_name, week_name, absolute=False, classic_colors=False):
    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
    division_teams = list()

    league = Playoffs.get_league_structure()
    for conf_name, conference in league.items():
        for div_name, division in conference.items():
            if div_name == division_name:
                division_teams = division

    teams = list(filter(lambda t: t[0] in division_teams, teams))
    plot_elo_function(teams,
                      division_name,
                      week_name,
                      absolute=absolute,
                      classic_colors=classic_colors,
                      show_plot=False)


def plot_team_elo_over_season(title, team_names):
    import Projects.nfl.NFL_Prediction.PlayoffHelper as Playoffs
    sns.set(style="ticks")
    max_len = max([len(Playoffs.team_elos[name]) for name in team_names])

    # Remove other teams
    team_elos = Playoffs.team_elos.copy()
    other_teams = [team for team in team_elos.keys() if team not in team_names]
    for other_team in other_teams:
        del team_elos[other_team]

    # Fill out data rows
    for name in team_names:
        elos = team_elos[name]
        while len(elos) < max_len + 1:
            team_elos[name].append(elos[-1])
    data = pd.DataFrame(team_elos)

    ax = data.plot.line(figsize=(20, 10))

    # Add titles
    ax.set_title('Team Elos: ' + title)
    ax.set_xlabel('Week')
    ax.set_ylabel('Elo')

    # Add the tick marks
    plt.xticks(range(max_len + 1))

    # Remove the x margins
    plt.margins(x=0)

    plt.show()
