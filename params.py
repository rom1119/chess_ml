import numpy as np

class POINTS_FIELD():
    ATTACK_VAL =  'ATTACK_VAL'
    DEFFENCE_VAL =  'DEFFENCE_VAL'
    FIGURE =  'FIGURE'
    ATTACK_FIGURE_VAL =  'ATTACK_FIGURE_VAL'

NR_MOVES = 300
LETTER_MAP = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
DIGIT_MAP = ['1', '2', '3', '4', '5', '6', '7', '8']
DIGIT_MAP.reverse()
DIGIT_MAP_NOT_REV = ['1', '2', '3', '4', '5', '6', '7', '8']

FIGURE_VALUES_ENEMY = {
    'r': 0.4,
    'n': 0.25,
    'b': 0.3,
    'q': 0.6,
    'k': 0.9,
    'p': 0.1,
}


FIGURE_VALUES_TO_ATTACK = {
    'r': 4,
    'n': 2,
    'b': 3,
    'q': 6,
    'k': 9,
    'p': 1,
}

FIGURE_VALUES_FOR_NET_MY = FIGURE_VALUES_ENEMY
FIGURE_VALUES_FOR_NET_ENEMY = FIGURE_VALUES_TO_ATTACK


weight_of_field_white = [
    [ 7 for _ in range(8)], 
    [ 6 for _ in range(8)], 
    [ 5 for _ in range(8)], 
    [ 4 for _ in range(8)], 
    [ 3 for _ in range(8)], 
    [ 2 for _ in range(8)], 
    [ 1 for _ in range(8)],
    [ 1 for _ in range(8)]
]


# aaa = [1,2,3,4]
# aaa_rev = [1,2,3,4]
# aaa_rev.reverse
weight_of_field_black = [
    [ 1 for _ in range(8)],
    [ 1 for _ in range(8)],
    [ 2 for _ in range(8)], 
    [ 3 for _ in range(8)], 
    [ 4 for _ in range(8)], 
    [ 5 for _ in range(8)], 
    [ 6 for _ in range(8)], 
    [ 7 for _ in range(8)], 
]

weight_of_field_white = np.array(weight_of_field_white) / 10
weight_of_field_black = np.array(weight_of_field_black) / 10

# START_FEN = '8/5k2/8/8/8/8/8/4K2R w - - 0 0'
# START_FEN = 'r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
START_FEN = None

default_board = [
['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'], 
['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'], 
['.', '.', '.', '.', '.', '.', '.', '.'], 
['.', '.', '.', '.', '.', '.', '.', '.'], 
['.', '.', '.', '.', '.', '.', '.', '.'], 
['.', '.', '.', '.', '.', '.', '.', '.'], 
['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'], 
['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
]

# default_board = [
# ['r', '.', '.', '.', 'k', 'b', 'n', 'r'], 
# ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'], 
# ['.', '.', '.', '.', '.', '.', '.', '.'], 
# ['.', '.', '.', '.', '.', '.', '.', '.'], 
# ['.', '.', '.', '.', '.', '.', '.', '.'], 
# ['.', '.', '.', '.', '.', '.', '.', '.'], 
# ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'], 
# ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
# ]



def getQualitySetup(is_white_move):
    res = 0
    board = default_board

    if is_white_move:
        for idx_digit, digit in enumerate(DIGIT_MAP):
            for idx_letter, letter in enumerate(LETTER_MAP):
                cell = board[idx_digit][idx_letter]
                if cell.isupper():
                    res += FIGURE_VALUES_ENEMY[cell.lower()] * weight_of_field_white[idx_digit][idx_letter]

    else:
        for idx_digit, digit in enumerate(DIGIT_MAP):
            for idx_letter, letter in enumerate(LETTER_MAP):
                cell = board[idx_digit][idx_letter]
                if cell.islower():
                    res += FIGURE_VALUES_ENEMY[cell.lower()] * weight_of_field_black[idx_digit][idx_letter]

    return res