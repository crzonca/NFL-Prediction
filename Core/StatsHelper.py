def game_point_diff(row):
    return row['home_score'] - row['away_score']


def home_victory(row):
    return 1 if row['game_point_diff'] > 0 else 0


def home_draw(row):
    return 1 if row['game_point_diff'] == 0 else 0


def home_win_pct(row):
    return row['home_wins'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_points_for(row):
    return row['home_total_points_for'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_points_against(row):
    return row['home_total_points_against'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_first_downs(row):
    return row['home_total_first_downs'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_rush_attempts(row):
    return row['home_total_rush_attempts'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_rushing_yards(row):
    return row['home_total_rushing_yards'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_rushing_touchdowns(row):
    return row['home_total_rushing_touchdowns'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_pass_completions(row):
    return row['home_total_pass_completions'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_pass_attempts(row):
    return row['home_total_pass_attempts'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_passing_yards(row):
    return row['home_total_passing_yards'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_passing_touchdowns(row):
    return row['home_total_passing_touchdowns'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_interceptions_thrown(row):
    return row['home_total_interceptions_thrown'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_times_sacked(row):
    return row['home_total_times_sacked'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_sacked_yards(row):
    return row['home_total_sacked_yards'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_net_passing_yards(row):
    return row['home_total_net_passing_yards'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_total_yards(row):
    return row['home_total_total_yards'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_fumbles(row):
    return row['home_total_fumbles'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_fumbles_lost(row):
    return row['home_total_fumbles_lost'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_turnovers(row):
    return row['home_total_turnovers'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_penalties(row):
    return row['home_total_penalties'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_penalty_yards(row):
    return row['home_total_penalty_yards'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_third_down_conversions(row):
    return row['home_total_third_down_conversions'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_third_downs(row):
    return row['home_total_third_downs'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_third_down_pct(row):
    return (row['home_total_third_down_conversions'] / row['home_total_third_downs']
            if row['home_total_third_downs'] > 0 else 0)


def home_average_fourth_down_conversions(row):
    return row['home_total_fourth_down_conversions'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_fourth_downs(row):
    return row['home_total_fourth_downs'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_fourth_down_pct(row):
    return (row['home_total_fourth_down_conversions'] / row['home_total_fourth_downs']
            if row['home_total_fourth_downs'] > 0 else 0)


def home_average_time_of_possession(row):
    return row['home_total_time_of_possession'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_yards_allowed(row):
    return row['home_total_yards_allowed'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_interceptions(row):
    return row['home_total_interceptions'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_fumbles_forced(row):
    return row['home_total_fumbles_forced'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_fumbles_recovered(row):
    return row['home_total_fumbles_recovered'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_turnovers_forced(row):
    return row['home_total_turnovers_forced'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_sacks(row):
    return row['home_total_sacks'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def home_average_sack_yards_forced(row):
    return row['home_total_sack_yards_forced'] / row['home_games_played'] if row['home_games_played'] > 0 else 0


def away_win_pct(row):
    return row['away_wins'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_points_for(row):
    return row['away_total_points_for'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_points_against(row):
    return row['away_total_points_against'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_first_downs(row):
    return row['away_total_first_downs'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_rush_attempts(row):
    return row['away_total_rush_attempts'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_rushing_yards(row):
    return row['away_total_rushing_yards'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_rushing_touchdowns(row):
    return row['away_total_rushing_touchdowns'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_pass_completions(row):
    return row['away_total_pass_completions'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_pass_attempts(row):
    return row['away_total_pass_attempts'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_passing_yards(row):
    return row['away_total_passing_yards'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_passing_touchdowns(row):
    return row['away_total_passing_touchdowns'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_interceptions_thrown(row):
    return row['away_total_interceptions_thrown'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_times_sacked(row):
    return row['away_total_times_sacked'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_sacked_yards(row):
    return row['away_total_sacked_yards'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_net_passing_yards(row):
    return row['away_total_net_passing_yards'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_total_yards(row):
    return row['away_total_total_yards'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_fumbles(row):
    return row['away_total_fumbles'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_fumbles_lost(row):
    return row['away_total_fumbles_lost'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_turnovers(row):
    return row['away_total_turnovers'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_penalties(row):
    return row['away_total_penalties'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_penalty_yards(row):
    return row['away_total_penalty_yards'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_third_down_conversions(row):
    return row['away_total_third_down_conversions'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_third_downs(row):
    return row['away_total_third_downs'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_third_down_pct(row):
    return (row['away_total_third_down_conversions'] / row['away_total_third_downs']
            if row['away_total_third_downs'] > 0 else 0)


def away_average_fourth_down_conversions(row):
    return row['away_total_fourth_down_conversions'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_fourth_downs(row):
    return row['away_total_fourth_downs'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_fourth_down_pct(row):
    return (row['away_total_fourth_down_conversions'] / row['away_total_fourth_downs']
            if row['away_total_fourth_downs'] > 0 else 0)


def away_average_time_of_possession(row):
    return row['away_total_time_of_possession'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_yards_allowed(row):
    return row['away_total_yards_allowed'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_interceptions(row):
    return row['away_total_interceptions'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_fumbles_forced(row):
    return row['away_total_fumbles_forced'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_fumbles_recovered(row):
    return row['away_total_fumbles_recovered'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_turnovers_forced(row):
    return row['away_total_turnovers_forced'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_sacks(row):
    return row['away_total_sacks'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def away_average_sack_yards_forced(row):
    return row['away_total_sack_yards_forced'] / row['away_games_played'] if row['away_games_played'] > 0 else 0


def home_average_yards_per_rush_attempt(row):
    return (row['home_average_rushing_yards'] / row['home_average_rush_attempts']
            if row['home_average_rush_attempts'] > 0 else 0)


def home_average_rushing_touchdowns_per_attempt(row):
    return (row['home_average_rushing_touchdowns'] / row['home_average_rush_attempts']
            if row['home_average_rush_attempts'] > 0 else 0)


def home_average_yards_per_pass_attempt(row):
    return (row['home_average_net_passing_yards'] / row['home_average_pass_attempts']
            if row['home_average_pass_attempts'] > 0 else 0)


def home_average_passing_touchdowns_per_attempt(row):
    return (row['home_average_passing_touchdowns'] / row['home_average_pass_attempts']
            if row['home_average_pass_attempts'] > 0 else 0)


def home_average_yards_per_pass_completion(row):
    return (row['home_average_net_passing_yards'] / row['home_average_pass_completions']
            if row['home_average_pass_completions'] > 0 else 0)


def home_average_rushing_play_pct(row):
    plays = row['home_average_rush_attempts'] + row['home_average_pass_attempts']
    return row['home_average_rush_attempts'] / plays if plays > 0 else 0


def home_average_passing_play_pct(row):
    plays = row['home_average_rush_attempts'] + row['home_average_pass_attempts']
    return row['home_average_pass_attempts'] / plays if plays > 0 else 0


def home_average_rushing_yards_pct(row):
    return (row['home_average_rushing_yards'] / row['home_average_total_yards']
            if row['home_average_total_yards'] > 0 else 0)


def home_average_passing_yards_pct(row):
    return (row['home_average_net_passing_yards'] / row['home_average_total_yards']
            if row['home_average_total_yards'] > 0 else 0)


def home_average_completion_pct(row):
    return (row['home_average_pass_completions'] / row['home_average_pass_attempts']
            if row['home_average_pass_attempts'] > 0 else 0)


def home_average_sacked_pct(row):
    return (row['home_average_times_sacked'] / row['home_average_pass_attempts']
            if row['home_average_pass_attempts'] > 0 else 0)


def home_average_passer_rating(row):
    a = (home_average_completion_pct(row) - .3) * 5
    b = (home_average_yards_per_pass_attempt(row) - 3) * .25
    c = home_average_passing_touchdowns_per_attempt(row) * 20
    ints_per_attempt = (row['home_average_interceptions_thrown'] / row['home_average_pass_attempts']
                        if row['home_average_pass_attempts'] > 0 else 0)
    d = 2.375 - (ints_per_attempt * 25)

    a = 2.375 if a > 2.375 else 0 if a < 0 else a
    b = 2.375 if b > 2.375 else 0 if b < 0 else b
    c = 2.375 if c > 2.375 else 0 if c < 0 else c
    d = 2.375 if d > 2.375 else 0 if d < 0 else d

    return ((a + b + c + d) / 6) * 100


def home_average_touchdowns(row):
    return home_average_rushing_touchdowns(row) + home_average_passing_touchdowns(row)


def home_average_yards_per_point(row):
    return home_average_total_yards(row) / home_average_points_for(row) if home_average_points_for(row) > 0 else 0


def home_average_scoring_margin(row):
    return home_average_points_for(row) - home_average_points_against(row)


def home_average_turnover_margin(row):
    return home_average_turnovers_forced(row) - home_average_turnovers(row)


def away_average_yards_per_rush_attempt(row):
    return (row['away_average_rushing_yards'] / row['away_average_rush_attempts']
            if row['away_average_rush_attempts'] > 0 else 0)


def away_average_rushing_touchdowns_per_attempt(row):
    return (row['away_average_rushing_touchdowns'] / row['away_average_rush_attempts']
            if row['away_average_rush_attempts'] > 0 else 0)


def away_average_yards_per_pass_attempt(row):
    return (row['away_average_net_passing_yards'] / row['away_average_pass_attempts']
            if row['away_average_pass_attempts'] > 0 else 0)


def away_average_passing_touchdowns_per_attempt(row):
    return (row['away_average_passing_touchdowns'] / row['away_average_pass_attempts']
            if row['away_average_pass_attempts'] > 0 else 0)


def away_average_yards_per_pass_completion(row):
    return (row['away_average_net_passing_yards'] / row['away_average_pass_completions']
            if row['away_average_pass_completions'] > 0 else 0)


def away_average_rushing_play_pct(row):
    plays = row['away_average_rush_attempts'] + row['away_average_pass_attempts']
    return row['away_average_rush_attempts'] / plays if plays > 0 else 0


def away_average_passing_play_pct(row):
    plays = row['away_average_rush_attempts'] + row['away_average_pass_attempts']
    return row['away_average_pass_attempts'] / plays if plays > 0 else 0


def away_average_rushing_yards_pct(row):
    return (row['away_average_rushing_yards'] / row['away_average_total_yards']
            if row['away_average_total_yards'] > 0 else 0)


def away_average_passing_yards_pct(row):
    return (row['away_average_net_passing_yards'] / row['away_average_total_yards']
            if row['away_average_total_yards'] > 0 else 0)


def away_average_completion_pct(row):
    return (row['away_average_pass_completions'] / row['away_average_pass_attempts']
            if row['away_average_pass_attempts'] > 0 else 0)


def away_average_sacked_pct(row):
    return (row['away_average_times_sacked'] / row['away_average_pass_attempts']
            if row['away_average_pass_attempts'] > 0 else 0)


def away_average_passer_rating(row):
    a = (away_average_completion_pct(row) - .3) * 5
    b = (away_average_yards_per_pass_attempt(row) - 3) * .25
    c = away_average_passing_touchdowns_per_attempt(row) * 20
    ints_per_attempt = (row['away_average_interceptions_thrown'] / row['away_average_pass_attempts']
                        if row['away_average_pass_attempts'] > 0 else 0)
    d = 2.375 - (ints_per_attempt * 25)

    a = 2.375 if a > 2.375 else 0 if a < 0 else a
    b = 2.375 if b > 2.375 else 0 if b < 0 else b
    c = 2.375 if c > 2.375 else 0 if c < 0 else c
    d = 2.375 if d > 2.375 else 0 if d < 0 else d

    return ((a + b + c + d) / 6) * 100


def away_average_touchdowns(row):
    return away_average_rushing_touchdowns(row) + away_average_passing_touchdowns(row)


def away_average_yards_per_point(row):
    return away_average_total_yards(row) / away_average_points_for(row) if away_average_points_for(row) > 0 else 0


def away_average_scoring_margin(row):
    return away_average_points_for(row) - away_average_points_against(row)


def away_average_turnover_margin(row):
    return away_average_turnovers_forced(row) - away_average_turnovers(row)


def win_pct_diff(row):
    return row['home_win_pct'] - row['away_win_pct']


def elo_diff(row):
    return row['home_elo'] - row['away_elo']


def average_points_for_diff(row):
    return row['home_average_points_for'] - row['away_average_points_for']


def average_points_against_diff(row):
    return row['home_average_points_against'] - row['away_average_points_against']


def average_first_downs_diff(row):
    return row['home_average_first_downs'] - row['away_average_first_downs']


def average_rush_attempts_diff(row):
    return row['home_average_rush_attempts'] - row['away_average_rush_attempts']


def average_rushing_yards_diff(row):
    return row['home_average_rushing_yards'] - row['away_average_rushing_yards']


def average_rushing_touchdowns_diff(row):
    return row['home_average_rushing_touchdowns'] - row['away_average_rushing_touchdowns']


def average_pass_completions_diff(row):
    return row['home_average_pass_completions'] - row['away_average_pass_completions']


def average_pass_attempts_diff(row):
    return row['home_average_pass_attempts'] - row['away_average_pass_attempts']


def average_passing_yards_diff(row):
    return row['home_average_passing_yards'] - row['away_average_passing_yards']


def average_passing_touchdowns_diff(row):
    return row['home_average_passing_touchdowns'] - row['away_average_passing_touchdowns']


def average_interceptions_thrown_diff(row):
    return row['home_average_interceptions_thrown'] - row['away_average_interceptions_thrown']


def average_times_sacked_diff(row):
    return row['home_average_times_sacked'] - row['away_average_times_sacked']


def average_sacked_yards_diff(row):
    return row['home_average_sacked_yards'] - row['away_average_sacked_yards']


def average_net_passing_yards_diff(row):
    return row['home_average_net_passing_yards'] - row['away_average_net_passing_yards']


def average_total_yards_diff(row):
    return row['home_average_total_yards'] - row['away_average_total_yards']


def average_fumbles_diff(row):
    return row['home_average_fumbles'] - row['away_average_fumbles']


def average_fumbles_lost_diff(row):
    return row['home_average_fumbles_lost'] - row['away_average_fumbles_lost']


def average_turnovers_diff(row):
    return row['home_average_turnovers'] - row['away_average_turnovers']


def average_penalties_diff(row):
    return row['home_average_penalties'] - row['away_average_penalties']


def average_penalty_yards_diff(row):
    return row['home_average_penalty_yards'] - row['away_average_penalty_yards']


def average_third_down_pct_diff(row):
    return row['home_average_third_down_pct'] - row['away_average_third_down_pct']


def average_fourth_down_pct_diff(row):
    return row['home_average_fourth_down_pct'] - row['away_average_fourth_down_pct']


def average_time_of_possession_diff(row):
    return row['home_average_time_of_possession'] - row['away_average_time_of_possession']


def average_yards_allowed_diff(row):
    return row['home_average_yards_allowed'] - row['away_average_yards_allowed']


def average_interceptions_diff(row):
    return row['home_average_interceptions'] - row['away_average_interceptions']


def average_fumbles_forced_diff(row):
    return row['home_average_fumbles_forced'] - row['away_average_fumbles_forced']


def average_fumbles_recovered_diff(row):
    return row['home_average_fumbles_recovered'] - row['away_average_fumbles_recovered']


def average_turnovers_forced_diff(row):
    return row['home_average_turnovers_forced'] - row['away_average_turnovers_forced']


def average_sacks_diff(row):
    return row['home_average_sacks'] - row['away_average_sacks']


def average_sack_yards_forced_diff(row):
    return row['home_average_sack_yards_forced'] - row['away_average_sack_yards_forced']


def average_yards_per_rush_attempt_diff(row):
    return row['home_average_yards_per_rush_attempt'] - row['away_average_yards_per_rush_attempt']


def average_rushing_touchdowns_per_attempt_diff(row):
    return row['home_average_rushing_touchdowns_per_attempt'] - row['away_average_rushing_touchdowns_per_attempt']


def average_yards_per_pass_attempt_diff(row):
    return row['home_average_yards_per_pass_attempt'] - row['away_average_yards_per_pass_attempt']


def average_passing_touchdowns_per_attempt_diff(row):
    return row['home_average_passing_touchdowns_per_attempt'] - row['away_average_passing_touchdowns_per_attempt']


def average_yards_per_pass_completion_diff(row):
    return row['home_average_yards_per_pass_completion'] - row['away_average_yards_per_pass_completion']


def average_rushing_play_pct_diff(row):
    return row['home_average_rushing_play_pct'] - row['away_average_rushing_play_pct']


def average_passing_play_pct_diff(row):
    return row['home_average_passing_play_pct'] - row['away_average_passing_play_pct']


def average_rushing_yards_pct_diff(row):
    return row['home_average_rushing_yards_pct'] - row['away_average_rushing_yards_pct']


def average_passing_yards_pct_diff(row):
    return row['home_average_passing_yards_pct'] - row['away_average_passing_yards_pct']


def average_completion_pct_diff(row):
    return row['home_average_completion_pct'] - row['away_average_completion_pct']


def average_sacked_pct_diff(row):
    return row['home_average_sacked_pct'] - row['away_average_sacked_pct']


def average_passer_rating_diff(row):
    return row['home_average_passer_rating'] - row['away_average_passer_rating']


def average_touchdowns_diff(row):
    return row['home_average_touchdowns'] - row['away_average_touchdowns']


def average_yards_per_point_diff(row):
    return row['home_average_yards_per_point'] - row['away_average_yards_per_point']


def average_scoring_margin_diff(row):
    return row['home_average_scoring_margin'] - row['away_average_scoring_margin']


def average_turnover_margin_diff(row):
    return row['home_average_turnover_margin'] - row['away_average_turnover_margin']
