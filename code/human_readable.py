def human_readable(player_obs):
# translates the board encoding into natural numbers
        human_board = []
        for pos in player_obs['board']:
            human_board.append(pos[0] + pos[1] + pos[2] + pos[3] * 2)
        return human_board
        