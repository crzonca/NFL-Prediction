import statistics

import matplotlib.pyplot as plt
import numpy as np


def plot_elo_function(teams, week_name, absolute=False, classic_colors=False):
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

    # Add the tick marks
    bottom_remainder = 50 - (bottom % 50)
    plt.xticks(range(bottom + bottom_remainder, top + bottom_remainder + 1, 50))
    plt.yticks(range(0, 101, 10))

    # Remove the x margins
    plt.margins(x=0)

    plt.show()
