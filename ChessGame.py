import numpy as np

import sys 
sys.path.append("./gemma-demo/") 

import chess
import chess.svg
from cairosvg import svg2png
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from time import sleep
import csv
from neuralNetwork import NeuralNetwork

from utils.utils import *

from params import *
from utils.calcDeffence import DeffenceAlgorithm
from utils.calcAttack import AttackAlgorithm
import random
from bprint import *

class ChessGame():

    last_moves_black = []

    history_boards = []
    selected_moves = []
    current_board_for_y = []
    is_white_move = True
    show_ui = True
    nr_moves_white = 0

    game_move_nr = 0

    def __init__(self, show_ui=False) :
        self.board = chess.Board()
        self.show_ui = show_ui

        if show_ui:
            plt.ion()
            fig, axs = plt.subplots(1,sharex=True)
            self.ax = axs

        self.default_board = np.array(default_board)


    def tryShowUI(self):

        if self.show_ui:
            self.ax.clear()
            svg_text = chess.svg.board(
                self.board,
                # fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
                # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
                # squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
                size=350,
            ) 

            # with open('example-board.svg', 'w') as f:
            #     f.write(svg_text)

            svg2png(bytestring=svg_text, write_to='board.png')
            img = mpimg.imread('board.png')
            imgplot = self.ax.imshow(img)
            plt.pause(0.03)


    def play_ai(self, fen_moves, neuralNetwork : NeuralNetwork):

        self.is_white_move = True
        values_for_target_field = {}
        last_moves = []
        last_moves_black = []
        self.history_boards = []
        self.selected_moves = []

        for fen_move in fen_moves:
            legal_moves = list(self.board.legal_moves)
            move_ai = None
            if self.is_white_move:
                X = createX(self.default_board)
                move_ai = neuralNetwork.predict(X)
            # else :
            #     self.board.push_san(fen_move )


            if move_ai is not None:
                
                generated_board_points, possible_moves, move_value, values_for_target_field, current_board_for_y = prepare_matrix_points_for_ai(self.is_white_move, self.default_board, legal_moves)
                
                # most_quality_legal_move, legal_move_split, DEF_FIGURE = self.generate_and_choose_move(possible_moves, [], self.last_moves_black, values_for_target_field, move_value)
                
                generated_ai_possible_move = revertMove(move_ai)
                self.current_board_for_y = current_board_for_y

                field_val_map, figure_def_val, figure_idx = (generated_ai_possible_move[0], generated_ai_possible_move[1], generated_ai_possible_move[2])
                # field_val = float(round(field_val, 2))
                # figure_val = float(round(figure_val, 2))
                # figure_idx = int(abs(round(figure_idx, 0)))
                if figure_def_val == 0.2:
                    figure_def_val = 0.25
                print(f'!!!!game_move_nr {self.game_move_nr}', end='\n\n')
                print(f'!!!!white_move_nr {len(self.selected_moves)}',end='\n\n')
                # print(f'!!!!possible_move {possible_move}',end='\n\n')
                print(f'!!!!figure_def_val {figure_def_val} figure_idx {figure_idx}',end='\n\n')
                # print(f'!!!!field_val_map {field_val_map.tolist()}',end='\n\n')
                # print(f'!!!!values_for_target_field {values_for_target_field}',end='\n\n')

                field_val = 9999
                # while True:
                field_val_map[field_val_map < 0] = 0
                tmp_map = field_val_map.reshape(64)

                ind = np.argpartition(tmp_map, -15)[-15:]
                top5 = sorted(np.unique(tmp_map[ind]), reverse=True)
                selected_coords = None
                for most_val in top5:
                    if not selected_coords is None: break

                    Y_coords, X_cords = np.where(field_val_map == most_val)
                    print(f"Y_coords {Y_coords}, X_cords{X_cords} - most_val {most_val}")
                    for idx_cord, cord_y in enumerate(Y_coords):
                        if self.current_board_for_y[cord_y][X_cords[idx_cord]] > 0:
                            field_val = self.current_board_for_y[cord_y][X_cords[idx_cord]]
                            selected_coords = f"cord_y {cord_y} cord_x {X_cords[idx_cord]}"
                            break

                # print(f'!!!!field_val_map {field_val_map}',end='\n\n')

                # print(f"current_board_for_y {self.current_board_for_y}")
                # print(f"selected field_val {field_val} selected_coords {selected_coords}")
                
                if field_val in values_for_target_field:
                    most_val_moves = values_for_target_field[field_val]
                    print(f'!!!!GREAT field_val {field_val}')
                else :
                    most_val_moves = values_for_target_field[np.max(list(values_for_target_field.keys()))]

                # print(f'!!!!most_val_moves {most_val_moves}',end='\n\n')

                    

                if self.is_white_move:
                    # if self.nr_moves_white % 5 == 0:
                    if figure_def_val in most_val_moves:
                        min_figure = figure_def_val
                        print(f'!!!!GREAT figure_def_val {figure_def_val}')

                        # min_figure = np.random.choice(list(most_val_moves.keys()))
                    else:
                        min_figure = np.min(list(most_val_moves.keys()))
                    
                    self.nr_moves_white = self.nr_moves_white + 1


                legal_random_move_list = most_val_moves[min_figure]
                # print(f'!!!!legal_random_move_list {legal_random_move_list}')

                if figure_idx > len(legal_random_move_list) - 1:
                    legal_random_move_idx = len(legal_random_move_list) - 1
                elif figure_idx < 0:
                    legal_random_move_idx = 0
                else:
                    legal_random_move_idx = figure_idx
                    print(f'!!!!GREAT figure_idx {figure_idx}/{len(legal_random_move_list)}')
                    # legal_random_move_idx = random.randint(0, len(legal_random_move_list) - 1)
                # print( 'len(legal_random_move_list)', len(legal_random_move_list))
                # print( 'legal_random_move_idx', legal_random_move_idx)

                legal_random_move = legal_random_move_list[int(legal_random_move_idx)]
                legal_move_split = (legal_random_move['legal_move'][0] + legal_random_move['legal_move'][1], legal_random_move['legal_move'][2] + legal_random_move['legal_move'][3])
                most_quality_legal_move = legal_random_move

                # selectMove(self.current_board_for_y, legal_move_split, most_quality_legal_move, legal_random_move_idx)
            
            else :

                generated_board_points, possible_moves, move_value, values_for_target_field, current_board_for_y = prepare_matrix_points(self.is_white_move, self.default_board, legal_moves, fen_move)
                print_m('generated_board_points ', generated_board_points)
                print_m('possible_moves ', possible_moves)
                print_m('move_value ', move_value)
                print_m('values_for_target_field ', values_for_target_field)
                print_m('current_board_for_y ', current_board_for_y)
                # print_m('last_moves ', last_moves)
                print_m('possible_moves ', possible_moves)
                print_m('legal_moves ', legal_moves)
                
                
                most_quality_legal_move, legal_move_split, DEF_FIGURE, legal_random_move_idx, history_boards = generate_and_choose_move(self.default_board, self.is_white_move, self.history_boards, fen_move, possible_moves, last_moves, last_moves_black, values_for_target_field, move_value)
            
            # print('last_moves', last_moves)


            column_move_difference = np.abs(LETTER_MAP.index(legal_move_split[1][0].upper()) - LETTER_MAP.index(legal_move_split[0][0].upper()))

            print('i=' + str(self.game_move_nr))
            # print('quality_me=' + str(quality_me))

            print(legal_move_split[0] + ' -> ' + legal_move_split[1])

            self.default_board = move(legal_move_split[0], legal_move_split[1], self.default_board)
            self.board.push_san(most_quality_legal_move['legal_move'] )

            print(self.board, end='\n\n')
            # print(default_board, end='\n\n')
            # print('DEF_FIGURE', DEF_FIGURE)
            DEF_FIGURE = most_quality_legal_move['def_figure']
            initPromotionFigureAndCastling(self.default_board, self.is_white_move, self.board, legal_move_split, DEF_FIGURE, column_move_difference)

            self.is_white_move = not self.is_white_move
            self.game_move_nr = self.game_move_nr + 1

            if len(list(legal_moves)) == 0:
                if self.is_white_move:
                    print('WHITE WIN !!!!!')
                else:
                    print('BLACK WIN !!!!!')

                return False

            colour = 'WHITE'
            if not self.is_white_move:
                colour = 'BLACK'

            print('====== MOVE =================' , colour )
            # display.update(board.fen())
            # ax.plot(predictedX, predictedY)
            self.tryShowUI()

                # sleep(1)
        return True


    def play_auto(self, fen_moves):

        self.is_white_move = True
        values_for_target_field = {}
        last_moves = []
        last_moves_black = []
        self.history_boards = []
        self.selected_moves = []
        print_m('len moves ', len(fen_moves))

        for fen_move in fen_moves:
            # fen_move = target_move_fen.split('.')[1]
            legal_moves = list(self.board.legal_moves)
            if len(list(legal_moves)) == 0:
                # print_m('i=' + str(_))

                print_m(self.board, end='\n\n')
                if self.is_white_move:
                    print_m('WIN BLACK')
                    return False
                else:
                    print_m('WIN WHITE')
                    return True
            if self.is_white_move:
                print_m('MOVE WHITE', fen_move, True)
            else:
                print_m('MOVE BLACK', fen_move, True)
            # self.board.push_san(fen_move )

            # fen_move = fen_moves[2]
            # print_m('self.is_white_move ', self.is_white_move)
            print_m('MOVE ', fen_move, True)
            print_m('fen_move', fen_move, True)
            # print_m('legal_moves', legal_moves)

            for legal_move in legal_moves:
                legal_move = str(legal_move)
                if len(legal_move) == 5:
                    print_m('legal_move 5 !!!!!', legal_move)

            # print_m('is_checkmate', self.board.is_checkmate())
            # print_m('is_check()', self.board.is_check())
            # # print_m('self.board ')
            # # print_m( self.board)
  

            # # print_m('self.board ', self.default_board)

          
            generated_board_points, possible_moves, move_value, values_for_target_field, current_board_for_y = prepare_matrix_points(self.is_white_move, self.default_board, legal_moves, fen_move)
            print_m('generated_board_points ', generated_board_points)
            print_m('possible_moves ', possible_moves)
            print_m('move_value ', move_value)
            print_m('values_for_target_field ', values_for_target_field)
            print_m('current_board_for_y ', current_board_for_y)
            print_m('last_moves ', last_moves)
            print_m('possible_moves ', possible_moves)
            print_m('legal_moves ', legal_moves)
            
            
            most_quality_legal_move, legal_move_split, DEF_FIGURE, legal_random_move_idx, history_boards = generate_and_choose_move(self.default_board, self.is_white_move, self.history_boards, fen_move, possible_moves, last_moves, last_moves_black, values_for_target_field, move_value)
            print_m('most_quality_legal_move ', most_quality_legal_move)
            print_m('most_quality_legal_move[legal_move] ', most_quality_legal_move['legal_move'])
            # print_m('self.board ', self.board)
            # self.board.push_san(most_quality_legal_move['legal_move'] )

            print_m('============================')
            print_m('============================')
            print_m('============================')
            print_m('============================')
            print_m('============================')
            # return True
            self.history_boards = history_boards
            self.current_board_for_y = current_board_for_y
            # # print_m(default_board, end='\n\n')
            # # print_m(board, end='\n\n')

            if self.is_white_move:
                # cell_nr = generate_cell_nr(legal_move_split[1][1], legal_move_split[1][0].upper())
    
                res = selectMove(self.current_board_for_y, legal_move_split, most_quality_legal_move, legal_random_move_idx)
                self.selected_moves.append(res)



            column_move_difference = np.abs(LETTER_MAP.index(legal_move_split[1][0].upper()) - LETTER_MAP.index(legal_move_split[0][0].upper()))

            # # print_m('column_move_difference=' + str(column_move_difference))
            # # print_m('move_value=' + str(move_value))
            # # print_m('i=' + str(_))
            # # print_m('quality_me=' + str(quality_me))
            # # print_m('quality_enemy=' + str(quality_enemy))
            # # print_m('move_value=' + str(move_value))
            # # print_m('is_white_move=' + str(is_white_move))
            # # print_m(legal_move_split[0] + ' -> ' + legal_move_split[1])

            self.default_board = move(legal_move_split[0], legal_move_split[1], self.default_board)
            self.board.push_san(most_quality_legal_move['legal_move'] )

            print_m(self.default_board, None, True)
            print_m(self.board, None, True)
            # # print_m('DEF_FIGURE', DEF_FIGURE)

            initPromotionFigureAndCastling(self.default_board, self.is_white_move, self.board,legal_move_split, DEF_FIGURE, column_move_difference)

            self.is_white_move = not self.is_white_move
   

            self.tryShowUI()
            break
            # sleep(1)
        # print_m('i=' + str(_))

        print_m(self.board, None, True)
        print_m(self.default_board, None, True)

        return False


file_path = '/Users/romanpytka/ai/csv/chess/test.txt'

# game.auto_play_chess()

with open(file_path, newline='\n') as csvfile:

    reader = csv.DictReader(csvfile)
    games = []
    for row in reader:
        # print(row)
        # break
        row = row['# #']
        fen = row.split('###')
        if len(fen) > 1:
            fen_clear = fen[1].strip()
            if len(fen_clear):
                games.append(fen_clear)
        # break


    # print_m('el=',curr_game)
    checkmate_games = []
    for game in games:
        # print('game', game)
        # print('game 22', game[len(game) - 1])
        if game[len(game) - 1] != '#':
            continue
        # print('game 33333333', game[len(game) - 1])
        moves = []
        # game_moves = []
        for el in game.split(' '):
            moves.append(el.split('.')[1])
            print_m('el=',el)
# maasive pixel creation 
        checkmate_games.append(moves)
    # curr_game = games[9].split(' ')[:2]
    curr_game = checkmate_games[1]

    print(curr_game)

    game = ChessGame(True)

    # game.play_auto(curr_game)

    nn = NeuralNetwork()

    # game.play_ai(curr_game, nn)

    # X = np.array(game.history_boards) / 10
    # Y = np.array(game.selected_moves) / 10
    # np.save('./data/X.txt', X)
    # np.save('./data/Y.txt', Y)

    X = np.load('./data/X.txt.npy')
    Y = np.load('./data/Y.txt.npy')
    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))
    Y_norm = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    X = X_norm
    Y = Y_norm
    size_x = 10
    size_y = 10

    xx = [np.linspace(1, size_x, size_x)]
    yy = [np.linspace(1, size_y, size_y)]

    for x in X:
        xx = np.append(xx, [x[:size_x]], axis=0)
        
    for y in Y:
        yy = np.append(yy, [y[:size_y]], axis=0)
        # print('loop', x[:50])

    xx = np.delete(xx, 0, 0)
    yy = np.delete(yy, 0, 0)

    # print('X', xx)

    Y_index = 18


    y_1 = Y[Y_index]
    y_1 = np.delete(y_1, len(y_1) - 1, 0)
    y_1 = np.delete(y_1, len(y_1) - 1, 0)
    
    y_1_norm = Y_norm[Y_index]
    y_1_norm = np.delete(y_1_norm, len(y_1_norm) - 1, 0)
    y_1_norm = np.delete(y_1_norm, len(y_1_norm) - 1, 0)

    # with open("./data/X.txt", "w") as txt_file:
    #     for line in X:
    #         txt_file.write(" ".join(line) + "\n")

    # with open("./data/Y.txt", "w") as txt_file:
    #     for line in Y:
    #         txt_file.write(" ".join(line) + "\n")
    # sleep(3)
    print('X', X[Y_index].reshape(8,8))
    print('X_norm', X_norm[Y_index].reshape(8,8))
    print('Y', Y[Y_index])
    print('Y_norm', Y_norm[Y_index])
    print('Y 1', y_1.reshape(8,8))
    print('Y_norm 2', trunc(y_1_norm.reshape(8,8), 1))
    # print('Y', Y[1])
    # print('Y', Y[2])

    print('X len', len(xx), xx.shape)
    print('Y len', len(Y), Y.shape)

    print(curr_game)
    print('len', len(checkmate_games))
    # nn.train(xx, yy)

# TRAIN
    nn.train(X, Y)


# PREDS
    # preds = np.array(nn.predict(X_norm[Y_index]))  
    # print('preds', preds)

    # preds = np.delete(preds, len(preds) - 1, 0)
    # preds = np.delete(preds, len(preds) - 1, 0)
    # q = Y[Y_index]
    # q = np.delete(q, len(q) - 1, 0)
    # q = np.delete(q, len(q) - 1, 0)

    # print('X', X[Y_index].reshape(8,8))
    # print('Y', q.reshape(8,8))

    # print('preds', trunc(preds, 1).reshape(8,8))
        


