import itertools
import math

import numpy as np
from scipy.stats import rankdata

from Projects.nfl.NFL_Prediction.Pred import league_structure
from Projects.nfl.NFL_Prediction.Redo.poibin import PoiBin


class PlayoffPredictor:
    def __init__(self, team_df, graph):
        self.team_df = team_df
        self.graph = graph

    def get_remaining_win_probs(self, team_name):
        schedule = league_structure.load_schedule().get('weeks')
        opponents = list()
        for week in schedule:
            for game in week:
                if game.get('away') == team_name:
                    opponent = game.get('home')
                    opponents.append(opponent)
                if game.get('home') == team_name:
                    opponent = game.get('away')
                    opponents.append(opponent)

        bt = self.team_df.at[team_name, 'Bayes BT']
        wins = self.team_df.at[team_name, 'Wins']
        losses = self.team_df.at[team_name, 'Losses']
        ties = self.team_df.at[team_name, 'Ties']
        games_played = wins + losses + ties
        if games_played == 17:
            return []

        remaining_opponents = opponents[games_played:17]
        opponent_bts = [self.team_df.at[opponent, 'Bayes BT'] for opponent in remaining_opponents]
        win_probs = [math.exp(bt) / (math.exp(bt) + math.exp(opp_bt)) for opp_bt in opponent_bts]

        return win_probs

    def get_total_wins_chances(self, team):
        wins = self.team_df.at[team, 'Wins']
        wins_dict = {win_total: 0.0 for win_total in range(18)}

        win_probs = self.get_remaining_win_probs(team)
        loss_probs = [1 - win_prob for win_prob in win_probs]

        win_mask = list(itertools.product([0, 1], repeat=len(win_probs)))
        for win_combo in win_mask:
            loss_combo = [0 if game == 1 else 1 for game in win_combo]

            win_combo_probs = list(itertools.compress(win_probs, win_combo))
            loss_combo_probs = list(itertools.compress(loss_probs, loss_combo))
            win_combo_wins = len(win_combo_probs) + wins

            total_wins_prob = np.prod(win_combo_probs)
            total_losses_prob = np.prod(loss_combo_probs)

            combo_prob = total_wins_prob * total_losses_prob

            wins_dict[win_combo_wins] = wins_dict.get(win_combo_wins) + combo_prob

        return wins_dict

    def get_proj_record(self, team_name):
        win_probs = self.get_remaining_win_probs(team_name)

        wins = self.team_df.at[team_name, 'Wins']
        ties = self.team_df.at[team_name, 'Ties']

        expected_wins = sum(win_probs) + wins
        expected_losses = 17 - expected_wins - ties

        return round(expected_wins), round(expected_losses), ties

    def get_first_round_bye_chance(self, team):
        teams_division = league_structure.get_division(team)
        teams_conference = teams_division.split()[0]
        conference_opponents = [opp for opp in self.team_df.index if
                                opp != team and league_structure.get_division(opp).split()[0] == teams_conference]

        tiebreak_opponents = {opp: self.get_tiebreak(team, opp, divisional=False) for opp in conference_opponents}

        total_wins_chances = self.get_total_wins_chances(team)
        opp_wins_chances = {opp: self.get_total_wins_chances(opp) for opp in conference_opponents}

        win_chances = list()
        for win_total, chance in total_wins_chances.items():
            if chance == 0:
                continue

            all_opponent_chances = list()
            for opponent, opp_chances in opp_wins_chances.items():
                if tiebreak_opponents.get(opponent):
                    opponent_chances = {k: v for k, v in opp_chances.items() if k <= win_total}
                else:
                    opponent_chances = {k: v for k, v in opp_chances.items() if k < win_total}

                opponent_chance = sum(opponent_chances.values())
                all_opponent_chances.append(opponent_chance)

            win_chances.append(chance * np.prod(all_opponent_chances))

        first_place_chance = sum(win_chances)
        first_place_chance = min([first_place_chance, 1])
        first_place_chance = max([first_place_chance, 0])
        return first_place_chance

    def get_wildcard_chance(self, team):
        teams_division = league_structure.get_division(team)
        teams_conference = teams_division.split()[0]
        conference_opponents = [opp for opp in self.team_df.index if
                                opp != team and league_structure.get_division(opp).split()[0] == teams_conference]

        tiebreak_opponents = {opp: self.get_tiebreak(team, opp, divisional=False) for opp in conference_opponents}

        total_wins_chances = self.get_total_wins_chances(team)
        opp_wins_chances = {opp: self.get_total_wins_chances(opp) for opp in
                            conference_opponents}

        win_chances = list()
        for win_total, chance in total_wins_chances.items():
            if chance == 0:
                continue

            all_opponent_chances = list()
            for opponent, opp_chances in opp_wins_chances.items():

                # Each opponent's chance to get the required number of wins to beat the team
                if tiebreak_opponents.get(opponent):
                    opponent_chances = {k: v for k, v in opp_chances.items() if k > win_total}
                else:
                    opponent_chances = {k: v for k, v in opp_chances.items() if k >= win_total}

                opponent_chance = sum(opponent_chances.values())

                # Times that opponents chance to lose their division (be in wild card hunt)
                all_opponent_chances.append(
                    opponent_chance * (1 - self.get_division_winner_chance(opponent)))

            # The teams chance to have 2 or fewer teams beat them in the WC race
            pb = PoiBin(all_opponent_chances)
            make_wc_chance = pb.cdf(2)

            # Times the teams chance to get that number of wins
            win_chances.append(chance * make_wc_chance)

        # Total chance to win a WC slot times the teams chance to lose their division (be in wild card hunt)
        wc_chance = sum(win_chances) * (1 - self.get_division_winner_chance(team))
        wc_chance = min([wc_chance, 1])
        wc_chance = max([wc_chance, 0])
        return wc_chance

    def get_division_winner_chance(self, team):
        teams_division = league_structure.get_division(team)

        division_opponents = [opp for opp in self.team_df.index if
                              opp != team and league_structure.get_division(opp) == teams_division]
        tiebreak_opponents = {opp: self.get_tiebreak(team, opp, divisional=True) for opp in division_opponents}

        total_wins_chances = self.get_total_wins_chances(team)
        opp_wins_chances = {opp: self.get_total_wins_chances(opp) for opp in
                            division_opponents}

        win_chances = list()
        for win_total, chance in total_wins_chances.items():
            if chance == 0:
                continue

            all_opponent_chances = list()
            for opponent, opp_chances in opp_wins_chances.items():
                if tiebreak_opponents.get(opponent):
                    opponent_chances = {k: v for k, v in opp_chances.items() if k <= win_total}
                else:
                    opponent_chances = {k: v for k, v in opp_chances.items() if k < win_total}

                opponent_chance = sum(opponent_chances.values())
                all_opponent_chances.append(opponent_chance)

            win_chances.append(chance * np.prod(all_opponent_chances))

        first_place_chance = sum(win_chances)
        first_place_chance = min([first_place_chance, 1])
        first_place_chance = max([first_place_chance, 0])
        return first_place_chance

    def get_tiebreak_head_to_head(self, team, opponent):
        previous_games = [e for e in self.graph.edges if team in e and opponent in e]
        previous_wins = [(loser, winner) for loser, winner, num in previous_games if winner == team]
        previous_losses = [(loser, winner) for loser, winner, num in previous_games if loser == team]

        if len(previous_wins) > len(previous_losses):
            return True

        if len(previous_wins) < len(previous_losses):
            return False

        return None

    def get_tiebreak_divisional_win_pct(self, team, opponent):
        division = league_structure.get_division(team)
        teams_opponents = [opp for opp in self.team_df.index if
                           opp != team and league_structure.get_division(opp) == division]
        opponents_opponents = [opp for opp in self.team_df.index if
                               opp != opponent and league_structure.get_division(opp) == division]

        teams_divisional_games = [e for e in self.graph.edges if team in e and any(opp in e for opp in teams_opponents)]
        opponents_divisional_games = [e for e in self.graph.edges if
                                      opponent in e and any(opp in e for opp in opponents_opponents)]

        team_divisional_wins = [(loser, winner) for loser, winner, num in teams_divisional_games if winner == team]
        team_divisional_losses = [(loser, winner) for loser, winner, num in teams_divisional_games if loser == team]
        team_divisional_games = len(team_divisional_wins) + len(team_divisional_losses)
        if team_divisional_games == 0:
            team_divisional_win_pct = 0
        else:
            team_divisional_win_pct = len(team_divisional_wins) / team_divisional_games

        opponent_divisional_wins = [(loser, winner) for loser, winner, num in opponents_divisional_games if
                                    winner == opponent]
        opponent_divisional_losses = [(loser, winner) for loser, winner, num in opponents_divisional_games if
                                      loser == opponent]
        opponent_divisional_games = len(opponent_divisional_wins) + len(opponent_divisional_losses)
        if opponent_divisional_games == 0:
            opponent_divisional_win_pct = 0
        else:
            opponent_divisional_win_pct = len(opponent_divisional_wins) / opponent_divisional_games

        if team_divisional_win_pct > opponent_divisional_win_pct:
            return True

        if team_divisional_win_pct < opponent_divisional_win_pct:
            return False

        return None

    def get_tiebreak_conference_win_pct(self, team, opponent):
        teams_conference = league_structure.get_division(team).split()[0]
        opponents_conference = league_structure.get_division(opponent).split()[0]
        teams_opponents = [opp for opp in self.team_df.index if opp != team
                           and league_structure.get_division(opp).split()[0] == teams_conference]
        opponents_opponents = [opp for opp in self.team_df.index if opp != opponent
                               and league_structure.get_division(opp).split()[0] == opponents_conference]

        teams_conference_games = [e for e in self.graph.edges if team in e and any(opp in e for opp in teams_opponents)]
        opponents_conference_games = [e for e in self.graph.edges if
                                      opponent in e and any(opp in e for opp in opponents_opponents)]

        team_conference_wins = [(loser, winner) for loser, winner, num in teams_conference_games if winner == team]
        team_conference_losses = [(loser, winner) for loser, winner, num in teams_conference_games if loser == team]
        team_conference_games = len(team_conference_wins) + len(team_conference_losses)
        if team_conference_games == 0:
            team_conference_win_pct = 0
        else:
            team_conference_win_pct = len(team_conference_wins) / team_conference_games

        opponent_conference_wins = [(loser, winner) for loser, winner, num in opponents_conference_games if
                                    winner == opponent]
        opponent_conference_losses = [(loser, winner) for loser, winner, num in opponents_conference_games if
                                      loser == opponent]
        opponent_conference_games = len(opponent_conference_wins) + len(opponent_conference_losses)
        if opponent_conference_games == 0:
            opponent_conference_win_pct = 0
        else:
            opponent_conference_win_pct = len(opponent_conference_wins) / opponent_conference_games

        if team_conference_win_pct > opponent_conference_win_pct:
            return True

        if team_conference_win_pct < opponent_conference_win_pct:
            return False

        return None

    def get_tiebreak_common_win_pct(self, team, opponent, divisional=True):
        teams_games = [e for e in self.graph.edges if team in e]
        opponents_games = [e for e in self.graph.edges if opponent in e]

        teams_opponents = set([loser for loser, winner, num in teams_games]).union(
            set([winner for loser, winner, num in teams_games]))

        opponents_opponents = set([loser for loser, winner, num in opponents_games]).union(
            set([winner for loser, winner, num in opponents_games]))

        common_opponents = teams_opponents.intersection(opponents_opponents)

        teams_common_games = [e for e in self.graph.edges if team in e and
                              any(opp in e for opp in common_opponents)]
        opponents_common_games = [e for e in self.graph.edges if opponent in e and
                                  any(opp in e for opp in common_opponents)]

        if not divisional and len(teams_common_games) + len(opponents_common_games) < 4:
            return None

        team_common_wins = [(loser, winner) for loser, winner, num in teams_common_games if winner == team]
        team_common_losses = [(loser, winner) for loser, winner, num in teams_common_games if loser == team]
        team_common_games = len(team_common_wins) + len(team_common_losses)
        if team_common_games == 0:
            team_common_win_pct = 0
        else:
            team_common_win_pct = len(team_common_wins) / team_common_games

        opponent_common_wins = [(loser, winner) for loser, winner, num in opponents_common_games
                                if winner == opponent]
        opponent_common_losses = [(loser, winner) for loser, winner, num in opponents_common_games
                                  if loser == opponent]
        opponent_common_games = len(opponent_common_wins) + len(opponent_common_losses)
        if opponent_common_games == 0:
            opponent_common_win_pct = 0
        else:
            opponent_common_win_pct = len(opponent_common_wins) / opponent_common_games

        if team_common_win_pct > opponent_common_win_pct:
            return True

        if team_common_win_pct < opponent_common_win_pct:
            return False

        return None

    def get_tiebreak_strength_of_victory(self, team, opponent):
        teams_games = [e for e in self.graph.edges if team in e]
        opponents_games = [e for e in self.graph.edges if opponent in e]

        team_wins = [(loser, winner) for loser, winner, num in teams_games if winner == team]
        opponent_wins = [(loser, winner) for loser, winner, num in opponents_games if winner == opponent]

        team_total_wins = 0
        team_total_losses = 0
        team_total_ties = 0
        for loser, team in team_wins:
            team_total_wins = team_total_wins + self.team_df.at[loser, 'Wins']
            team_total_losses = team_total_losses + self.team_df.at[loser, 'Losses']
            team_total_ties = team_total_ties + self.team_df.at[loser, 'Ties']

        opponent_total_wins = 0
        opponent_total_losses = 0
        opponent_total_ties = 0
        for loser, opponent in opponent_wins:
            opponent_total_wins = opponent_total_wins + self.team_df.at[loser, 'Wins']
            opponent_total_losses = opponent_total_losses + self.team_df.at[loser, 'Losses']
            opponent_total_ties = opponent_total_ties + self.team_df.at[loser, 'Ties']

        team_total_games = team_total_wins + team_total_losses + team_total_ties
        if team_total_games == 0:
            team_total_wp = 0
        else:
            team_total_wp = team_total_wins / team_total_games

        opponent_total_games = opponent_total_wins + opponent_total_losses + opponent_total_ties
        if opponent_total_games == 0:
            opponent_total_wp = 0
        else:
            opponent_total_wp = opponent_total_wins / opponent_total_games

        if team_total_wp > opponent_total_wp:
            return True

        if team_total_wp < opponent_total_wp:
            return False

        return None

    def get_tiebreak_strength_of_schedule(self, team, opponent):
        teams_games = [e for e in self.graph.edges if team in e]
        opponents_games = [e for e in self.graph.edges if opponent in e]

        team_total_wins = 0
        team_total_losses = 0
        team_total_ties = 0
        for loser, winner, weight in teams_games:
            if weight != 0:
                i = 0
            if winner == team:
                other_team = loser
            else:
                other_team = winner
            team_total_wins = team_total_wins + self.team_df.at[other_team, 'Wins']
            team_total_losses = team_total_losses + self.team_df.at[other_team, 'Losses']
            team_total_ties = team_total_ties + self.team_df.at[other_team, 'Ties']

        opponent_total_wins = 0
        opponent_total_losses = 0
        opponent_total_ties = 0
        for loser, winner, weight in opponents_games:
            if winner == opponent:
                other_team = loser
            else:
                other_team = winner
            opponent_total_wins = opponent_total_wins + self.team_df.at[other_team, 'Wins']
            opponent_total_losses = opponent_total_losses + self.team_df.at[other_team, 'Losses']
            opponent_total_ties = opponent_total_ties + self.team_df.at[other_team, 'Ties']

        team_total_games = team_total_wins + team_total_losses + team_total_ties
        if team_total_games == 0:
            team_total_wp = 0
        else:
            team_total_wp = team_total_wins / team_total_games

        opponent_total_games = opponent_total_wins + opponent_total_losses + opponent_total_ties
        if opponent_total_games == 0:
            opponent_total_wp = 0
        else:
            opponent_total_wp = opponent_total_wins / opponent_total_games

        if team_total_wp > opponent_total_wp:
            return True

        if team_total_wp < opponent_total_wp:
            return False

        return None

    def get_tiebreak_conference_point_diff(self, team, opponent):
        import warnings
        warnings.filterwarnings('ignore')

        teams_conference = league_structure.get_division(team).split()[0]
        opponents_conference = league_structure.get_division(opponent).split()[0]

        teams_conf_df = self.team_df.loc[self.team_df['Division'].str.startswith(teams_conference)]
        teams_conf_df['Total Points'] = teams_conf_df.apply(lambda r: r['Avg Points'] * r['Games Played'], axis=1)
        teams_conf_df['Total Points Allowed'] = teams_conf_df.apply(
            lambda r: r['Avg Points Allowed'] * r['Games Played'],
            axis=1)
        teams_conf_df['Total Point Diff'] = teams_conf_df.apply(lambda r: r['Total Points'] - r['Total Points Allowed'],
                                                                axis=1)

        teams_index = list(teams_conf_df.index).index(team)
        teams_point_diff_rank = rankdata(teams_conf_df['Total Point Diff'], method='min')[teams_index]

        opponent_conf_df = self.team_df.loc[self.team_df['Division'].str.startswith(opponents_conference)]
        opponent_conf_df['Total Points'] = opponent_conf_df.apply(lambda r: r['Avg Points'] * r['Games Played'], axis=1)
        opponent_conf_df['Total Points Allowed'] = opponent_conf_df.apply(lambda r: r['Avg Points Allowed'] *
                                                                                    r['Games Played'], axis=1)
        opponent_conf_df['Total Point Diff'] = opponent_conf_df.apply(lambda r: r['Total Points'] -
                                                                                r['Total Points Allowed'], axis=1)

        opponents_index = list(opponent_conf_df.index).index(opponent)
        opponents_point_diff_rank = rankdata(teams_conf_df['Total Point Diff'], method='min')[opponents_index]

        if teams_point_diff_rank > opponents_point_diff_rank:
            return True

        if teams_point_diff_rank < opponents_point_diff_rank:
            return False

        return None

    def get_tiebreak_overall_point_diff(self, team, opponent):
        team_df_copy = self.team_df.copy()

        team_df_copy['Total Points'] = team_df_copy.apply(lambda r: r['Avg Points'] * r['Games Played'],
                                                          axis=1)
        team_df_copy['Total Points Allowed'] = team_df_copy.apply(lambda r: r['Avg Points Allowed'] * r['Games Played'],
                                                                  axis=1)
        team_df_copy['Total Point Diff'] = team_df_copy.apply(lambda r: r['Total Points'] - r['Total Points Allowed'],
                                                              axis=1)

        teams_index = list(team_df_copy.index).index(team)
        opponents_index = list(team_df_copy.index).index(opponent)
        ranking = rankdata(team_df_copy['Total Point Diff'], method='min')
        teams_point_diff_rank = ranking[teams_index]
        opponents_point_diff_rank = ranking[opponents_index]

        if teams_point_diff_rank > opponents_point_diff_rank:
            return True

        if teams_point_diff_rank < opponents_point_diff_rank:
            return False

        return None

    def get_tiebreak(self, team, opponent, divisional=True):
        # This is all done as things are currently, not as they may play out
        h2h = self.get_tiebreak_head_to_head(team, opponent)
        if h2h is not None:
            return h2h

        if divisional:
            divisional_wp = self.get_tiebreak_divisional_win_pct(team, opponent)
            if divisional_wp is not None:
                return divisional_wp
        else:
            conference_wp = self.get_tiebreak_conference_win_pct(team, opponent)
            if conference_wp is not None:
                return conference_wp

        common_wp = self.get_tiebreak_common_win_pct(team, opponent, divisional=divisional)
        if common_wp is not None:
            return common_wp

        if divisional:
            conference_wp = self.get_tiebreak_conference_win_pct(team, opponent)
            if conference_wp is not None:
                return conference_wp

        sov = self.get_tiebreak_strength_of_victory(team, opponent)
        if sov is not None:
            return sov

        sos = self.get_tiebreak_strength_of_schedule(team, opponent)
        if sos is not None:
            return sos

        conf_point_diff = self.get_tiebreak_conference_point_diff(team, opponent)
        if conf_point_diff is not None:
            return conf_point_diff

        overall_point_diff = self.get_tiebreak_overall_point_diff(team, opponent)
        if overall_point_diff is not None:
            return overall_point_diff

        team_bt = self.team_df.at[team, 'Bayes BT']
        opponent_bt = self.team_df.at[opponent, 'Bayes BT']

        if team_bt == opponent_bt:
            return team > opponent
        else:
            return team_bt > opponent_bt
