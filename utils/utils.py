
from params import *
from utils.calcDeffence import DeffenceAlgorithm
from utils.calcAttack import AttackAlgorithm
import random


deffenceCalc = DeffenceAlgorithm()
attackCalc = AttackAlgorithm()

def print_m(text, text_t = None, is_turn = False):
    if is_turn:
        print(text)
        if text_t:
            print(text_t)
def generate_clear_board_points(): 
    new_board = {}

    for digit in DIGIT_MAP:
        new_board[digit] = {}
        for letter in LETTER_MAP:
            new_board[digit][letter] = {
                'attack_figure_value': None,
                'deffence_figure_value': None,
                'attack_value': [0],
                'defence_value': [0],
                'figure_attack': None,
                'figure_deffence': None,
            }
    return  new_board

def generate_board_points_and_clear_board_field(default_board, is_white_move, board_points, enemy_figure_indexes, my_figure_indexes): 
    board = default_board
    if is_white_move:

        for idx_digit, digit in enumerate(DIGIT_MAP):
            for idx_letter, letter in enumerate(LETTER_MAP):
                cell = board[idx_digit][idx_letter]
                new_att_val = -1
                new_def_val = -1
                if cell.isupper():
                    new_att_val = -1
                    new_def_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                    board_points[digit][letter]['figure_deffence'] = cell
                    deffenceCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), is_white_move, board_points, my_figure_indexes)
                elif cell == '.':
                    # new_val = 0.01 * weight_of_field_white[idx_digit][idx_letter]
                    new_att_val = 0
                    new_def_val = 0
                else:
                    new_att_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                    new_def_val = -1
                    board_points[digit][letter]['figure_attack'] = cell
                    attackCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), is_white_move, board_points, enemy_figure_indexes)
                board_points[digit][letter]['attack_figure_value'] = new_att_val
                board_points[digit][letter]['deffence_figure_value'] = new_def_val


    else:
        for idx_digit, digit in enumerate(DIGIT_MAP):
            for idx_letter, letter in enumerate(LETTER_MAP):
                cell = board[idx_digit][idx_letter]
                new_att_val = -1
                new_def_val = -1
                if cell.islower():
                    new_att_val = -1
                    new_def_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                    board_points[digit][letter]['figure_deffence'] = cell
                    deffenceCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), is_white_move, board_points, my_figure_indexes)
                elif cell == '.':
                    # new_val = 0.01 * weight_of_field_black[idx_digit][idx_letter]
                    new_att_val = 0
                    new_def_val = 0
                else:
                    new_att_val = FIGURE_VALUES_TO_ATTACK[cell.lower()]
                    new_def_val = -1
                    board_points[digit][letter]['figure_attack'] = cell
                    attackCalc.generateValueForField(cell.lower(), (idx_digit, idx_letter), is_white_move, board_points, enemy_figure_indexes)
                board_points[digit][letter]['attack_figure_value'] = new_att_val
                board_points[digit][letter]['deffence_figure_value'] = new_def_val


    return board_points



def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def revertMove(predict):
    Y = np.array(predict).tolist()

    predict_y_def_figure = trunc(np.round(Y[len(Y) - 2], 1), 1)
    predict_y_move_idx = np.round(Y[len(Y) - 1], 1) * 10

    del Y[len(Y) - 1]
    # print_m(f"predict AF DEL= {predict}")
    del Y[len(Y) - 1]

    Y = np.array(Y)
    outputY = np.round(Y.reshape(8,8),2)

    outputY = trunc(outputY, 1)
    return (outputY, predict_y_def_figure, predict_y_move_idx)

def selectMove(current_board_for_y, legal_move_split, most_quality_legal_move, legal_random_move_idx):
    targetY = DIGIT_MAP.index(legal_move_split[1][1])
    targetX = LETTER_MAP.index(legal_move_split[1][0].upper())
    current_board_for_y[targetY][targetX] = 4
    # print_m(f"current_board_for_y {self.current_board_for_y}")
    current_board_for_y = current_board_for_y.reshape(64)
    current_board_for_y = np.append(current_board_for_y, most_quality_legal_move['figure_def_val'])
    current_board_for_y = np.append(current_board_for_y, legal_random_move_idx / 10)
    # print_m(f"current_board_for_y AF {self.current_board_for_y}")
    # print_m(f"most_quality_legal_move AF {most_quality_legal_move['curr_val']}")
    a = current_board_for_y
    
    return a

def prepare_matrix_points_for_ai(is_white_move, default_board, legal_moves):

        uniq_figures = np.array(list(FIGURE_VALUES_ENEMY.keys()))
        if is_white_move:
            enemy_figures = np.char.lower(uniq_figures)
            my_figures = np.char.upper(uniq_figures)
        else:
            enemy_figures = np.char.upper(uniq_figures)
            my_figures = np.char.lower(uniq_figures)

        enemy_figure_indexes = []
        my_figure_indexes = []

        for figure in enemy_figures:
            t_enemy = np.where(default_board == figure)
            for idx, item in enumerate(t_enemy[0]): 
                enemy_figure_indexes.append([t_enemy[0][idx], t_enemy[1][idx]])
        for figure in my_figures:
            t_my = np.where(default_board == figure)
            for idx, item in enumerate(t_my[0]): 
                my_figure_indexes.append([t_my[0][idx], t_my[1][idx]])

        clear_board_points = generate_clear_board_points()
        generated_board_points = generate_board_points_and_clear_board_field(default_board, is_white_move,clear_board_points, enemy_figure_indexes, my_figure_indexes)
        # print(generated_board_points)

        most_quality_legal_move = 0
        move_value = -100
        values_for_target_field = {}

        # quality_me = getQualitySetup(is_white_move)
        # quality_enemy = getQualitySetup(not is_white_move)

        attack_figure = []
        deffence_figure = []
        attack = []
        deffence = []
        move_values = []

        for row in generated_board_points.keys():
            attFigArr = []
            deffFigArr = []
            deffArr = []
            attackArr = []
            move_valuesArr = []
            # attack_figure.append([])
            for col in generated_board_points[row].keys():
                # print(generated_board_points[row][col])
                attFigArr.append(generated_board_points[row][col]['attack_figure_value'])
                deffFigArr.append(generated_board_points[row][col]['deffence_figure_value'])
                deffArr.append(round(np.sum(generated_board_points[row][col]['defence_value']), 3))
                attackArr.append(round(np.sum(generated_board_points[row][col]['attack_value']), 3))
                move_valuesArr.append([])
            attack_figure.append(attFigArr)
            deffence_figure.append(deffFigArr)
            deffence.append(deffArr)
            attack.append(attackArr)
            move_values.append(move_valuesArr)
            # attack.put(values=attArr)
            # deffence.put(values=deffArr)
            # # attack.put

        deffence = np.array(deffence)
        attack = np.array(attack)
        attack_figure = np.array(attack_figure)
        deffence_figure = np.array(deffence_figure)


        # print('is_white_move=' + str(is_white_move))
        # print('============== attack_figure ===================')
        # print (attack_figure, sep=' ')
        # print('============== deffence_figure ===================')
        # print (deffence_figure, sep=' ')
        # print('============== attack ===================')
        # print (attack, sep=' ')
        # print('============== deffence ===================')
        # print (deffence, sep=' ')
        
        attack_calc = attack_figure + attack
        deffence_calc = deffence_figure + deffence
        # print('============== attack_figure + attack ===================')
        # print (attack_calc)
        # print('============== deffence_figure + deffence ===================')
        # print (deffence_calc)

        calc_def_minus_att =  deffence_calc - attack_calc 
        # print('============== deffence_calc - attack_calc ===================')
        # print (calc_def_minus_att)
        
        calc_att_minusd_def = attack_calc - deffence_calc
        # print('============== attack_calc - deffence_calc ===================')
        # print (calc_att_minusd_def)  
        
        calc_white = attack_figure - attack
        # calc_white_t = - attack_figure + attack
        # print('============== attack_figure - attack ===================')
        # print (calc_white)        
        # print (calc_white_t)        

        # print('count legal_moves=' + str(len(list(legal_moves))))

        possible_moves = np.array(['o' for _ in range(64)])
        possible_moves = possible_moves.reshape(8,8)
        values_for_target_field = {}
        current_board_for_y = np.zeros(64).reshape(8,8)

        poss_moves = []
        # print(f'generated_board_points {generated_board_points} ')
        for legal_move in legal_moves:
            legal_move = str(legal_move)
            # if len(legal_move) == 5 and not legal_move[4].upper() == 'Q':
            #     continue
            poss_moves.append(legal_move)
            # attack_figure_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_figure_value']
            # attack_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_value']
            # print(f'legal_move {legal_move} ')

            figure_def = generated_board_points[legal_move[1]][legal_move[0].upper()]['figure_deffence']

            targetY = DIGIT_MAP.index(legal_move[3])
            targetX = LETTER_MAP.index(legal_move[2].upper())

            figure_def_Val = FIGURE_VALUES_ENEMY[figure_def.lower()]
        # =======================================================
            # curr_val = attack_figure_val - (np.mean(attack_val)) - (figure_def_Val)
            # curr_val = attack_figure_val - attack_calc[targetY][targetX] - figure_def_Val
            # print(f"calc_white[targetY][targetX] {calc_white[targetY][targetX]}")

            curr_val = calc_white[targetY][targetX] - figure_def_Val + 3
            current_board_for_y[targetY][targetX] = curr_val
            # if self.is_white_move:
            # else:
            #     curr_val = calc_def_minus_att[targetY][targetX] - figure_def_Val

        # =======================================================
            curr_val = round(curr_val, 3)
            move_values[targetY][targetX].append(curr_val)

            if curr_val not in values_for_target_field.keys():
                values_for_target_field[curr_val] = {}
                
            if figure_def_Val not in values_for_target_field[curr_val].keys():
                values_for_target_field[curr_val][figure_def_Val] = []

            legalMoveDict = {
                'def_figure': figure_def,
                'legal_move': legal_move,
                'figure_def_val': figure_def_Val,
                'curr_val': curr_val,
            }
            values_for_target_field[curr_val][figure_def_Val].append(legalMoveDict)

            if curr_val > move_value:
                move_value = curr_val
        
        # print(f'possible_moves {poss_moves}')
        return generated_board_points, possible_moves, move_value, values_for_target_field, current_board_for_y

def prepare_matrix_points(is_white_move, default_board, legal_moves, target_move_fen):

        uniq_figures = np.array(list(FIGURE_VALUES_ENEMY.keys()))
        if is_white_move:
            enemy_figures = np.char.lower(uniq_figures)
            my_figures = np.char.upper(uniq_figures)
        else:
            enemy_figures = np.char.upper(uniq_figures)
            my_figures = np.char.lower(uniq_figures)

        enemy_figure_indexes = []
        my_figure_indexes = []

        for figure in enemy_figures:
            t_enemy = np.where(default_board == figure)
            for idx, item in enumerate(t_enemy[0]): 
                enemy_figure_indexes.append([t_enemy[0][idx], t_enemy[1][idx]])
        for figure in my_figures:
            t_my = np.where(default_board == figure)
            for idx, item in enumerate(t_my[0]): 
                my_figure_indexes.append([t_my[0][idx], t_my[1][idx]])

        clear_board_points = generate_clear_board_points()
        generated_board_points = generate_board_points_and_clear_board_field(default_board,is_white_move, clear_board_points, enemy_figure_indexes, my_figure_indexes)
        # print_m(generated_board_points)

        most_quality_legal_move = 0
        move_value = -100
        values_for_target_field = {}

        # quality_me = getQualitySetup(is_white_move)
        # quality_enemy = getQualitySetup(not is_white_move)

        attack_figure = []
        deffence_figure = []
        attack = []
        deffence = []
        move_values = []

        for row in generated_board_points.keys():
            attFigArr = []
            deffFigArr = []
            deffArr = []
            attackArr = []
            move_valuesArr = []
            # attack_figure.append([])
            for col in generated_board_points[row].keys():
                # print_m(generated_board_points[row][col])
                attFigArr.append(generated_board_points[row][col]['attack_figure_value'])
                deffFigArr.append(generated_board_points[row][col]['deffence_figure_value'])
                deffArr.append(round(np.sum(generated_board_points[row][col]['defence_value']), 3))
                attackArr.append(round(np.sum(generated_board_points[row][col]['attack_value']), 3))
                move_valuesArr.append([])
            attack_figure.append(attFigArr)
            deffence_figure.append(deffFigArr)
            deffence.append(deffArr)
            attack.append(attackArr)
            move_values.append(move_valuesArr)
            # attack.put(values=attArr)
            # deffence.put(values=deffArr)
            # # attack.put

        deffence = np.array(deffence)
        attack = np.array(attack)
        attack_figure = np.array(attack_figure)
        deffence_figure = np.array(deffence_figure)


        # print_m('is_white_move=' + str(self.is_white_move))
        # print_m('============== attack_figure ===================')
        # print (attack_figure, sep=' ')
        # print_m('============== deffence_figure ===================')
        # print (deffence_figure, sep=' ')
        # print_m('============== attack ===================')
        # print (attack, sep=' ')
        # print_m('============== deffence ===================')
        # print (deffence, sep=' ')
        
        attack_calc = attack_figure + attack
        deffence_calc = deffence_figure + deffence
        # print_m('============== attack_figure + attack ===================')
        # print (attack_calc)
        # print_m('============== deffence_figure + deffence ===================')
        # print (deffence_calc)

        calc_def_minus_att =  deffence_calc - attack_calc 
        # print_m('============== deffence_calc - attack_calc ===================')
        # print (calc_def_minus_att)
        
        calc_att_minusd_def = attack_calc - deffence_calc
        # print_m('============== attack_calc - deffence_calc ===================')
        # print (calc_att_minusd_def)  
        
        calc_white = attack_figure - attack
        # calc_white_t = - attack_figure + attack
        # print_m('============== attack_figure - attack ===================')
        # print (calc_white)        
        # print (calc_white_t)        

        # print_m('count legal_moves=' + str(len(list(legal_moves))))

        possible_moves = np.array(['o' for _ in range(64)])
        possible_moves = possible_moves.reshape(8,8)
        values_for_target_field = {}
        current_board_for_y = np.zeros(64).reshape(8,8)

        poss_moves = []
        # print_m(f'generated_board_points {generated_board_points} ')

        # target_move_fen = 'Ne6+'
        print_m('target_move_fen',target_move_fen)
        target_move_fen = target_move_fen.replace('#', '')
        target_move_fen = target_move_fen.replace('+', '')
        target_move_fen = target_move_fen.replace('x', '')
        castling_figure = None
        if '=' in target_move_fen:
            castling_figure = target_move_fen.split('=')[1]
            print('yes')
            print('castling_figure',castling_figure)
            target_move_fen = target_move_fen.replace('=', '')
            target_move_fen = target_move_fen.replace(castling_figure, '')

        target_move_fen_len = len(target_move_fen)
        print_m('new target_move_fen',target_move_fen, True)
        print_m(' legal_moves',legal_moves, True)
        legal_move = None

        target_field = target_move_fen[len(target_move_fen)-2:]
        legal_moves_dict_for_figure = {}
        legal_moves_dict_for_column = {}

        if (target_move_fen == 'O-O' ) :
            for l_move in legal_moves:
                l_move = str(l_move)

                if is_white_move :

                    if l_move[3] == '1' and l_move[2] == 'g':
                        legal_move = l_move
                else :
                    if l_move[3] == '8' and l_move[2] == 'g':
                        legal_move = l_move

        elif (target_move_fen == 'O-O-O' ) :
            for l_move in legal_moves:
                l_move = str(l_move)

                if is_white_move :
                    if l_move[3] == '1' and l_move[2] == 'c':
                        legal_move = l_move
                else :
                    if l_move[3] == '8' and l_move[2] == 'c':
                        legal_move = l_move
        
        else :

        
            for l_move in legal_moves:
                l_move = str(l_move)
                if len(l_move) == 5 and not l_move[4].upper() == castling_figure:
                    continue
                
                from_field = l_move[:2]
                to_field = l_move[2:4]
                if to_field != target_field:
                    continue
                
                if to_field not in legal_moves_dict_for_figure:
                    legal_moves_dict_for_figure[to_field] = {}
                    legal_moves_dict_for_column[to_field] = {}
                
                figure_from = default_board[DIGIT_MAP.index(from_field[1])][LETTER_MAP.index(from_field[0].upper())]
                col_from =from_field[0].upper()

                if figure_from not in legal_moves_dict_for_figure[to_field]:
                    legal_moves_dict_for_figure[to_field][figure_from.lower()] = []
                    
                if col_from not in legal_moves_dict_for_column[to_field]:
                    legal_moves_dict_for_column[to_field][col_from] = []

                legal_moves_dict_for_figure[to_field][figure_from.lower()].append(l_move)
                legal_moves_dict_for_column[to_field][col_from].append(l_move)

                print_m('l_move', l_move, True)
                print_m('1 legal move', l_move[len(l_move)-2:], True)
                print_m('2', target_field, True)

            item_for_figure = legal_moves_dict_for_figure[target_field]
            item_for_column = legal_moves_dict_for_column[target_field]
            print_m(' legal_moves_dict_for_figure', legal_moves_dict_for_figure, True)
            print_m(' legal_moves_dict_for_column', legal_moves_dict_for_column, True)
            print_m(' item__figure', item_for_figure, True)
            print_m(' item__column', item_for_column, True)

        #  W1.d4 B1.d5 W2.c4 B2.e6 W3.Nc3 B3.Nf6 W4.cxd5 B4.exd5 W5.Bg5 B5.Be7 W6.e3 B6.Ne4 W7.Bxe7 B7.Nxc3 W8.Bxd8 B8.Nxd1 W9.Bxc7 B9.Nxb2 W10.Rb1 B10.Nc4 W11.Bxc4 B11.dxc4 W12.Ne2 B12.O-O W13.Nc3 B13.b6 W14.d5 B14.Na6 W15.Bd6 B15.Rd8 W16.Ba3 B16.Bb7 W17.e4 B17.f6 W18.Ke2 B18.Nc7 W19.Rhd1 B19.Ba6 W20.Ke3 B20.Kf7 W21.g4 B21.g5 W22.h4 B22.h6 W23.Rh1 B23.Re8 W24.f3 B24.Bb7 W25.hxg5 B25.fxg5 W26.d6 B26.Nd5+ W27.Nxd5 B27.Bxd5 W28.Rxh6 B28.c3 W29.d7 B29.Re6 W30.Rh7+ B30.Kg8
#13 2000.06.29 1-0 2851 None 83 date_false result_false welo_false belo_true edate_false setup_false fen_false result2_false oyrange_false blen_false  W1.b3 B1.c5 W2.Bb2 B2.Nc6 W3.g3 B3.d6 W4.Bg2 B4.Nf6 W5.c4 B5.a6 W6.Nc3 B6.e5 W7.d3 B7.Nd4 W8.e3 B8.Bg4 W9.Qd2 B9.Nf5 W10.Nge2 B10.Bxe2 W11.Qxe2 B11.g6 W12.Bxb7 B12.Rb8 W13.Bc6+ B13.Nd7 W14.O-O B14.Bg7 W15.Bg2 B15.O-O W16.Nd5 B16.Nb6 W17.Nxb6 B17.Rxb6 W18.Bh3 B18.Qf6 W19.f4 B19.Rb4 W20.fxe5 B20.dxe5 W21.e4 B21.Qe7 W22.exf5 B22.Kh8 W23.Rae1 B23.Rbb8 W24.f6 B24.Bxf6 W25.Rxf6 B25.Qxf6 W26.Bxe5 B26.Qxe5 W27.Qxe5+ B27.Kg8 W28.Bg2 B28.Rbe8 W29.Qxe8 B29.Rxe8 W30.Rxe8+ B30.Kg7 W31.Bd5 B31.h5 W32.b4 B32.cxb4 W33.c5 B33.b3 W34.Bxb3 B34.f5 W35.c6 B35.f4 W36.c7 B36.fxg3 W37.c8=Q B37.gxh2+ W38.Kxh2 B38.h4 W39.Qe6 B39.Kh6 W40.Rg8 B40.Kg5 W41.Rxg6+ B41.Kh5 W42.Qg4# 

            # if len(item_for_figure) == 1:
            if target_move_fen_len == 2:

                legal_move = item_for_figure['p'][0]

            elif target_move_fen_len == 3 and target_move_fen[0].isupper():
                print_m('this case', legal_move, True)

                # from_field = item[:2]
                # to_field = item[2:]
                    # figure_from = default_board[DIGIT_MAP.index(from_field[1])][LETTER_MAP.index(from_field[0].upper())]
                for k in item_for_figure:
                    if k.upper() == target_move_fen[0]:
                        legal_move = item_for_figure[k][0]
                print_m('figure_from', figure_from, True)
                print_m('target_move_fen', target_move_fen, True)



            elif target_move_fen_len == 3 and target_move_fen[0].islower():

                for k in item_for_column:
                    if k.upper() == target_move_fen[0].upper():
                        legal_move = item_for_column[k][0]
                        
                # if from_field[0] == target_move_fen[0]:
                #     legal_move = l_move
            elif target_move_fen_len == 4:
                print_m('this case 4', legal_move, True)

                figure_from = default_board[DIGIT_MAP.index(from_field[1])][LETTER_MAP.index(from_field[0].upper())]


                for fig in item_for_figure:
                    if fig.upper() == target_move_fen[0]:
                        for col in item_for_column:
                            if col.upper() == target_move_fen[1].upper():
                                legal_move = item_for_column[col][0]

                if legal_move is None:
                    for k in item_for_figure:
                        if k.upper() == target_move_fen[0]:
                            legal_move = item_for_figure[k][0]

                # if figure_from.upper() == target_move_fen[0] and from_field[0] == target_move_fen[1]:
                #     legal_move = l_move
        # else :
        #     if target_move_fen_len == 2:

        #         legal_move = item['p'][0]

        #     elif target_move_fen_len == 3 and target_move_fen[0].isupper():
        #         print_m('this case else', legal_move, True)

        #         for k in item_for_figure:
        #             if k.upper() == target_move_fen[0]:
        #                 legal_move = item_for_figure[k][0]
        #         print_m('figure_from', figure_from, True)
        #         print_m('target_move_fen', target_move_fen, True)
        #     elif target_move_fen_len == 3 and target_move_fen[0].islower():
        #         # figure_from = default_board[DIGIT_MAP.index(from_field[1])][LETTER_MAP.index(from_field[0].upper())]

        #         for k in item_for_column:
        #             if k.upper() == target_move_fen[0].upper():
        #                 legal_move = item_for_column[k][0]
        #     elif target_move_fen_len == 4:
        #         figure_from = default_board[DIGIT_MAP.index(from_field[1])][LETTER_MAP.index(from_field[0].upper())]

        #         if figure_from.upper() == target_move_fen[0] and from_field[0] == target_move_fen[1]:
        #             legal_move = l_move

        print_m('SELECTED MOVE', legal_move, True)
        print_m('SELECTED target_move_fen', target_move_fen, True)


        # for legal_move in legal_moves:

        for lm in legal_moves:
            lm = str(lm)
            # if len(legal_move) == 5 and not legal_move[4].upper() == 'Q':
            #     continue
        # poss_moves.append(legal_move)
        # attack_figure_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_figure_value']
        # attack_val = generated_board_points[legal_move[3]][legal_move[2].upper()]['attack_value']
        # print_m(f'legal_move {legal_move} ')

            f_def = generated_board_points[lm[1]][lm[0].upper()]['figure_deffence']

            tY = DIGIT_MAP.index(lm[3])
            tX = LETTER_MAP.index(lm[2].upper())

            f_def_Val = FIGURE_VALUES_ENEMY[f_def.lower()]
        # =======================================================
            # curr_val = attack_figure_val - (np.mean(attack_val)) - (figure_def_Val)
            # curr_val = attack_figure_val - attack_calc[targetY][targetX] - figure_def_Val
            # print_m(f"calc_white[targetY][targetX] {calc_white[targetY][targetX]}")

            c_val = calc_white[tY][tX] - f_def_Val + 3
            current_board_for_y[tY][tX] = c_val
        # if self.is_white_move:
        # else:
        #     curr_val = calc_def_minus_att[targetY][targetX] - figure_def_Val

    # =======================================================

        figure_def = generated_board_points[legal_move[1]][legal_move[0].upper()]['figure_deffence']

        targetY = DIGIT_MAP.index(legal_move[3])
        targetX = LETTER_MAP.index(legal_move[2].upper())

        figure_def_Val = FIGURE_VALUES_ENEMY[figure_def.lower()]
    # =======================================================
        # curr_val = attack_figure_val - (np.mean(attack_val)) - (figure_def_Val)
        # curr_val = attack_figure_val - attack_calc[targetY][targetX] - figure_def_Val
        # print_m(f"calc_white[targetY][targetX] {calc_white[targetY][targetX]}")

        curr_val = calc_white[targetY][targetX] - figure_def_Val + 3
        current_board_for_y[targetY][targetX] = curr_val
        curr_val = round(curr_val, 3)
        move_values[targetY][targetX].append(curr_val)

        if curr_val not in values_for_target_field.keys():
            values_for_target_field[curr_val] = {}
            
        if figure_def_Val not in values_for_target_field[curr_val].keys():
            values_for_target_field[curr_val][figure_def_Val] = []

        legalMoveDict = {
            'def_figure': figure_def,
            'legal_move': legal_move,
            'figure_def_val': figure_def_Val,
            'curr_val': curr_val,
        }
        values_for_target_field[curr_val][figure_def_Val].append(legalMoveDict)

        if curr_val > move_value:
            move_value = curr_val
        
        # print_m(f'possible_moves {poss_moves}')
        return generated_board_points, possible_moves, move_value, values_for_target_field, current_board_for_y


def createX(default_board):
    X = []
    for row in default_board:
        r = []
        for cell in row:
            if cell.isupper():
                val = FIGURE_VALUES_FOR_NET_MY[cell.lower()]
            elif cell == '.':
                # new_val = 0.01 * weight_of_field_white[idx_digit][idx_letter]
                val = 0
            else:
                val = FIGURE_VALUES_FOR_NET_ENEMY[cell.lower()]
            r.append(val)

        X.append(r)
    X = np.array(X)

    # li = np.array(X)
    # res = 0
    # for ids, x in enumerate(li):
    #     idx = ids + 1
    #     v = (0.000001 * idx) + (x + (idx ** 2)) 
    #     # print_m(f"idx={idx} x={x} (0.0001 * idx)={(0.000001 * idx)} v={v}")
    #     res += v
    # print_m(f"x")
    # print_m(f"{X}")
    x_resh = X.reshape(64)
    
    return x_resh

def generate_and_choose_move(default_board, is_white_move, history_boards, fen_move, possible_moves,  last_moves, last_moves_black, values_for_target_field, move_value):

    most_val_moves = values_for_target_field[move_value]

    min_figure = np.min(list(most_val_moves.keys()))
    # min_figure = most_val_moves[min_figure]


    # legal_random_move = np.random.choice(most_val_moves)
    legal_random_move_list = most_val_moves[min_figure]
    legal_random_move_idx = random.randint(0, len(legal_random_move_list) - 1)
    # print_m( 'len(legal_random_move_list)', len(legal_random_move_list))
    # print_m( 'legal_random_move_idx', legal_random_move_idx)

    legal_random_move = legal_random_move_list[legal_random_move_idx]

    legal_move_split = (legal_random_move['legal_move'][0] + legal_random_move['legal_move'][1], legal_random_move['legal_move'][2] + legal_random_move['legal_move'][3])

    most_quality_legal_move = legal_random_move

    # print_m(last_moves)

    DEF_FIGURE = most_quality_legal_move['def_figure']

    # print_m('last_moves', last_moves)
    
    if is_white_move:
        history_boards.append(createX(default_board))

        last_moves.append(most_quality_legal_move['legal_move'])
        if len(last_moves) > 30:

            last_moves = last_moves[len(last_moves) - 30:]
        
    else :
        last_moves_black.append(most_quality_legal_move['legal_move'])
        if len(last_moves_black) > 30:

            last_moves_black = last_moves_black[len(last_moves_black) - 30:]

    
    return (most_quality_legal_move, legal_move_split, DEF_FIGURE, legal_random_move_idx, history_boards)

def move(currPos, targetPos, default_board):

    currentPointer = (LETTER_MAP.index(currPos[0].upper()), DIGIT_MAP.index(currPos[1]))

    currentVal = default_board[currentPointer[1]][currentPointer[0]]
    
    targetPointer = (LETTER_MAP.index(targetPos[0].upper()), DIGIT_MAP.index((targetPos[1])))
    targetVal = default_board[targetPointer[1]][targetPointer[0]]

    default_board[currentPointer[1]][currentPointer[0]] = '.'
    default_board[targetPointer[1]][targetPointer[0]] = currentVal

    return default_board

def initPromotionFigureAndCastling(default_board, is_white_move, board, legal_move_split, DEF_FIGURE, column_move_difference):
    targetRow = legal_move_split[1][1]
    targetCol = legal_move_split[1][0]
    if (DEF_FIGURE == 'P' and is_white_move and targetRow == '8') or (DEF_FIGURE == 'p' and not is_white_move and targetRow == '1'):

        cell_nr = generate_cell_nr(targetRow, targetCol)
        # print_m('DIGIT_MAP', DIGIT_MAP)
        # print_m('digits', digits)
        # print_m('cell_nr', cell_nr)
        # print_m('new_figure', new_figure)
        new_figure = str(board.piece_at(cell_nr))

        if is_white_move :
            default_board[DIGIT_MAP.index(targetRow)][LETTER_MAP.index(targetCol.upper())] = new_figure.upper()
        else :
            default_board[DIGIT_MAP.index(targetRow)][LETTER_MAP.index(targetCol.upper())] = new_figure.lower()
    
    if column_move_difference > 1:
        if (DEF_FIGURE == 'K' and is_white_move and targetRow == '1' and targetCol == 'g' ) :
            if default_board[DIGIT_MAP.index('1')][LETTER_MAP.index('H')] == 'R':
                move('h1', 'f1', default_board)
        elif (DEF_FIGURE == 'K' and is_white_move and targetRow == '1' and targetCol == 'c' ):
            if default_board[DIGIT_MAP.index('1')][LETTER_MAP.index('A')] == 'R':
                move('a1', 'd1', default_board)
        elif (DEF_FIGURE == 'k' and not is_white_move and targetRow == '8' and targetCol == 'g' ) :
            if default_board[DIGIT_MAP.index('8')][LETTER_MAP.index('H')] == 'r':
                move('h8', 'f8', default_board)
        elif (DEF_FIGURE == 'k' and not is_white_move and targetRow == '8' and targetCol == 'c' ):
            if default_board[DIGIT_MAP.index('8')][LETTER_MAP.index('A')] == 'r':
                move('a8', 'd8', default_board)

def generate_cell_nr(targetRow, targetCol):
    return (DIGIT_MAP_NOT_REV.index(targetRow)) * 8 + LETTER_MAP.index(targetCol.upper())
