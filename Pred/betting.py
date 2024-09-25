import math

import pandas as pd
from prettytable import PrettyTable
from scipy.optimize import minimize_scalar
from scipy.stats import skellam
import statsmodels.api as sm
import warnings

import Projects.nfl.NFL_Prediction.OddsHelper as Odds
from Projects.nfl.NFL_Prediction.Pred.helper import Helper


class Bettor:
    def __init__(self, team_df, individual_df, graph, gen_poisson_model):
        self.team_df = team_df
        self.individual_df = individual_df
        self.graph = graph
        self.gen_poisson_model = gen_poisson_model

    def get_vegas_line(self, home_name, away_name, odds):
        matching_odds = [odd for odd in odds if (away_name in odd[0][0] or home_name in odd[0][0]) and
                         (away_name in odd[0][1] or home_name in odd[0][1])]
        if len(matching_odds) < 1:
            print('Odds not found for', away_name, '@', home_name)
            return 0
        else:
            matching_odds = matching_odds[0]
            return matching_odds[-1]

    def get_spread_chance(self, favorite, underdog, spread):
        if spread > 0:
            return

        helper = Helper(self.team_df, self.individual_df, self.graph, self.gen_poisson_model)
        favorite_dist = helper.get_dist_from_gen_poisson_model(favorite, underdog)
        underdog_dist = helper.get_dist_from_gen_poisson_model(underdog, favorite)

        max_points = 101
        cover_chances = list()
        push_chances = list()
        fail_chances = list()
        for score in range(max_points):
            underdog_chance = float(underdog_dist.pmf(score))
            favorite_chance = float(favorite_dist.pmf(score))
            favorite_cover = float(favorite_dist.sf(score - spread))
            favorite_fail = float(favorite_dist.cdf(score - spread) - favorite_chance)

            cover_chance = underdog_chance * favorite_cover
            push_chance = underdog_chance * favorite_chance
            fail_chance = underdog_chance * favorite_fail

            cover_chances.append(cover_chance)
            push_chances.append(push_chance)
            fail_chances.append(fail_chance)

        cover_chance = sum(cover_chances)
        push_chance = sum(push_chances)
        fail_chance = 1 - cover_chance - push_chance
        return cover_chance, push_chance, fail_chance

    def ats_bets(self, ):
        odds = Odds.get_fanduel_odds()

        bets = list()
        for game in odds:
            home_team, away_team, home_spread, away_spread, home_american, away_american = game

            if home_american == 9900 or away_american == 9900:
                continue

            home_spread = float(home_spread)
            away_spread = float(away_spread)

            home_team = home_team.split()[-1]
            away_team = away_team.split()[-1]

            favorite = home_team if home_spread < 0.0 else away_team
            underdog = away_team if home_spread < 0.0 else home_team

            favorite_spread = home_spread if home_spread < 0 else away_spread
            underdog_spread = away_spread if home_spread < 0 else home_spread

            favorite_american = home_american if home_spread < 0 else away_american
            underdog_american = away_american if home_spread < 0 else home_american

            cover_chance, push_chance, fail_chance = self.get_spread_chance(favorite, underdog, favorite_spread)

            favorite_chance = Odds.convert_american_to_probability(favorite_american)
            underdog_chance = Odds.convert_american_to_probability(underdog_american)

            favorite_payout = 1 / favorite_chance
            underdog_payout = 1 / underdog_chance

            expected_favorite_payout = favorite_payout * cover_chance + push_chance
            expected_underdog_payout = underdog_payout * fail_chance + push_chance

            favorite_row = {'Team': favorite,
                            'Spread': favorite_spread,
                            'Opponent': underdog,
                            'American Odds': favorite_american,
                            'Probability': f'{favorite_chance * 100:.3f}' + '%',
                            'Payout': round(favorite_payout, 2),
                            'Poisson Chance': f'{cover_chance * 100:.3f}' + '%',
                            'Expected Return': round(expected_favorite_payout, 2),
                            'Expected Profit': round(expected_favorite_payout, 2) - 1,
                            'Push Chance': f'{push_chance * 100:.3f}' + '%'}

            underdog_row = {'Team': underdog,
                            'Spread': underdog_spread,
                            'Opponent': favorite,
                            'American Odds': underdog_american,
                            'Probability': f'{underdog_chance * 100:.3f}' + '%',
                            'Payout': round(underdog_payout, 2),
                            'Poisson Chance': f'{fail_chance * 100:.3f}' + '%',
                            'Expected Return': round(expected_underdog_payout, 2),
                            'Expected Profit': round(expected_underdog_payout, 2) - 1,
                            'Push Chance': f'{push_chance * 100:.3f}' + '%'}

            bets.append(favorite_row)
            bets.append(underdog_row)

        bet_df = pd.DataFrame(bets)
        bet_df = bet_df.sort_values(by='Expected Return', ascending=False)

        good_bet_df = bet_df.loc[bet_df['Expected Return'] > 1].reset_index(drop=True)
        bad_bet_df = bet_df.loc[bet_df['Expected Return'] <= 1].reset_index(drop=True)

        green = '\033[32m'
        red = '\033[31m'
        stop = '\033[0m'

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)

        print(green)
        print(good_bet_df)
        print(stop)

        print(red)
        print(bad_bet_df)
        print(stop)

    def all_bets(self, pot):
        ats_odds = Odds.get_fanduel_odds(future_days=7)
        h2h_odds = Odds.get_fanduel_odds(future_days=7, bet_type='h2h')

        bets = list()
        for game in ats_odds:
            home_team, away_team, home_spread, away_spread, home_american, away_american = game

            home_spread = float(home_spread)
            away_spread = float(away_spread)

            home_team = home_team.split()[-1]
            away_team = away_team.split()[-1]

            favorite = home_team if home_spread < 0.0 else away_team
            underdog = away_team if home_spread < 0.0 else home_team

            favorite_spread = home_spread if home_spread < 0 else away_spread
            underdog_spread = away_spread if home_spread < 0 else home_spread

            favorite_american = home_american if home_spread < 0 else away_american
            underdog_american = away_american if home_spread < 0 else home_american

            cover_chance, push_chance, fail_chance = self.get_spread_chance(favorite, underdog, favorite_spread)

            favorite_chance = Odds.convert_american_to_probability(favorite_american)
            underdog_chance = Odds.convert_american_to_probability(underdog_american)

            favorite_payout = 1 / favorite_chance
            underdog_payout = 1 / underdog_chance

            expected_favorite_payout = favorite_payout * cover_chance + push_chance
            expected_underdog_payout = underdog_payout * fail_chance + push_chance

            amount = 1.5
            favorite_bet_pct = minimize_scalar(utility,
                                               amount,
                                               bounds=(0.0, 1),
                                               args=(cover_chance, favorite_payout, push_chance, fail_chance, 1))
            underdog_bet_pct = minimize_scalar(utility,
                                               amount,
                                               bounds=(0.0, 1),
                                               args=(fail_chance, underdog_payout, push_chance, cover_chance, 1))

            favorite_bet_amount = favorite_bet_pct.x * pot
            underdog_bet_amount = underdog_bet_pct.x * pot

            favorite_row = {'Team': favorite,
                            'Spread': favorite_spread,
                            'Opponent': underdog,
                            'American Odds': favorite_american,
                            'Probability': favorite_chance,
                            'Payout': favorite_payout,
                            'Model Chance': cover_chance,
                            'Push Chance': push_chance,
                            'Expected Value': expected_favorite_payout,
                            'Bet Percent': favorite_bet_pct.x,
                            'Bet Amount': favorite_bet_amount,
                            'Bet Type': 'ATS'}

            underdog_row = {'Team': underdog,
                            'Spread': underdog_spread,
                            'Opponent': favorite,
                            'American Odds': underdog_american,
                            'Probability': underdog_chance,
                            'Payout': underdog_payout,
                            'Model Chance': fail_chance,
                            'Push Chance': push_chance,
                            'Expected Value': expected_underdog_payout,
                            'Bet Percent': underdog_bet_pct.x,
                            'Bet Amount': underdog_bet_amount,
                            'Bet Type': 'ATS'}

            bets.append(favorite_row)
            bets.append(underdog_row)

        for game in h2h_odds:
            home_team, away_team, _, _, home_american, away_american = game

            home_team = home_team.split()[-1]
            away_team = away_team.split()[-1]

            home_bt = self.team_df.at[home_team, 'Bayes BT']
            away_bt = self.team_df.at[away_team, 'Bayes BT']

            home_chance = Odds.convert_american_to_probability(home_american)
            away_chance = Odds.convert_american_to_probability(away_american)

            home_payout = 1 / home_chance
            away_payout = 1 / away_chance

            home_bt_chance = math.exp(home_bt) / (math.exp(home_bt) + math.exp(away_bt))
            away_bt_chance = math.exp(away_bt) / (math.exp(home_bt) + math.exp(away_bt))

            expected_home_payout = home_payout * home_bt_chance
            expected_away_payout = away_payout * away_bt_chance

            amount = 1.5
            home_bet_pct = minimize_scalar(utility,
                                           amount,
                                           bounds=(0.0, 1),
                                           args=(home_bt_chance, home_payout, 0, 1 - home_bt_chance, 1))
            away_bet_pct = minimize_scalar(utility,
                                           amount,
                                           bounds=(0.0, 1),
                                           args=(away_bt_chance, away_payout, 0, 1 - away_bt_chance, 1))

            home_bet_amount = home_bet_pct.x * pot
            away_bet_amount = away_bet_pct.x * pot

            home_row = {'Team': home_team,
                        'Spread': 0,
                        'Opponent': away_team,
                        'American Odds': home_american,
                        'Probability': home_chance,
                        'Payout': home_payout,
                        'Model Chance': home_bt_chance,
                        'Push Chance': 0,
                        'Expected Value': expected_home_payout,
                        'Bet Percent': home_bet_pct.x,
                        'Bet Amount': home_bet_amount,
                        'Bet Type': 'H2H'}

            away_row = {'Team': away_team,
                        'Spread': 0,
                        'Opponent': home_team,
                        'American Odds': away_american,
                        'Probability': away_chance,
                        'Payout': away_payout,
                        'Model Chance': away_bt_chance,
                        'Push Chance': 0,
                        'Expected Value': expected_away_payout,
                        'Bet Percent': away_bet_pct.x,
                        'Bet Amount': away_bet_amount,
                        'Bet Type': 'H2H'}

            bets.append(home_row)
            bets.append(away_row)

        bet_df = pd.DataFrame(bets)
        bet_df = bet_df.sort_values(by='Bet Percent', ascending=False)
        remaining_pot = pot
        for index, row in bet_df.iterrows():
            bet_amount = bet_df.at[index, 'Bet Percent'] * remaining_pot
            bet_df.at[index, 'Bet Amount'] = bet_amount
            remaining_pot = remaining_pot - bet_amount

        bet_df['Expected Return'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Expected Value'], axis=1)
        bet_df['Expected Profit'] = bet_df.apply(lambda r: r['Expected Return'] - r['Bet Amount'], axis=1)
        bet_df['To Win'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Payout'] - r['Bet Amount'], axis=1)

        bet_df['Swing'] = bet_df.apply(lambda r: r['Bet Amount'] + r['To Win'], axis=1)
        bet_df = bet_df.sort_values(by='Swing', ascending=False)

        good_bet_df = bet_df.loc[bet_df['Expected Value'] > 1].reset_index(drop=True)
        bad_bet_df = bet_df.loc[bet_df['Expected Value'] <= 1].reset_index(drop=True)

        green = '\033[32m'
        red = '\033[31m'
        print_bet_table(good_bet_df, is_ats=True, color=green)
        print_bet_table(bad_bet_df, is_ats=True, color=red)


def utility(amount, pos_chance, pos_payout, push_chance, neg_chance, pot):
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    def crra_utility(x, alpha=1.5):
        # Risk Averse:
        #   alpha > 1
        # Risk Neutral:
        #   alpha = 1
        # Risk Seeking:
        #   alpha < 1
        if alpha == 1:
            return math.log(x)
        return (math.pow(x, 1 - alpha) - 1) / (1 - alpha)

    u = pos_chance * crra_utility(pot + (amount * pos_payout) - amount) + \
        push_chance * crra_utility(pot) + \
        neg_chance * crra_utility(pot - amount)
    return -u


def print_bet_table(df, is_ats=False, color=''):
    columns = ['Num', 'Team', 'Spread', 'Opponent', 'American Odds', 'Probability', 'Payout', 'Model Chance',
               'Push Chance', 'Bet Amount', 'Expected Value', 'Expected Return', 'Expected Profit', 'To Win']

    table = PrettyTable(columns)
    table.float_format = '0.3'

    stop = '\033[0m'

    row_num = 1
    for index, row in df.iterrows():
        table_row = list()
        table_row.append(color + str(row_num) + stop)
        row_num = row_num + 1
        for col in columns[1:]:
            if col == 'American Odds':
                val = '+' + str(row[col]) if row[col] > 0 else str(row[col])
            elif col == 'Spread':
                val = '--' if row['Bet Type'] == 'H2H' else str(row[col])
            elif col == 'Payout' or col == 'Expected Value':
                val = str(round(row[col], 2)) + 'x'
            elif col == 'Probability' or col == 'Model Chance' or col == 'Push Chance':
                val = f'{row[col] * 100:.3f}' + '%'
            elif col == 'Bet Amount' or col == 'Expected Return' or col == 'Expected Profit' or col == 'To Win':
                val = 0 if col == 'Expected Profit' and -.01 < row[col] < 0 else row[col]
                val = '${:,.2f}'.format(val)
            elif col == 'Expected Value':
                val = str(round(row[col], 2))
            else:
                val = str(row[col])
            table_row.append(color + val + stop)

        table.add_row(table_row)

    # Print the table
    print(table)
    print()
