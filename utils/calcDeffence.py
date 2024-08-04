from params import *



def getY(arg):
    return DIGIT_MAP[arg]

def getX(arg):
    return LETTER_MAP[arg]

def getVal(val, i):
    if type(val) == list:
        return val[i]
    
    return val

def generate_val(figure):
    return FIGURE_VALUES_ENEMY[figure]

class DeffenceAlgorithm:


    def generateValueByCross(self, currentPointer, generated_board_points, enemy_figure_indexes, val):
        i = 0
        x = currentPointer[1] + 1
        y = currentPointer[0] + 1
        while y < 8 and x < 8 :
            generated_board_points[getY(y)][getX(x)]['defence_value'].append(getVal(val, i))
            if [y, x] in enemy_figure_indexes:
                break
            x += 1
            y += 1
            i += 1
        i = 0
        x = currentPointer[1] - 1
        y = currentPointer[0] + 1
        while y < 8 and x < 8 and y >= 0 and x >= 0 :
            generated_board_points[getY(y)][getX(x)]['defence_value'].append(getVal(val, i))
            if [y, x] in enemy_figure_indexes:
                break
            x -= 1
            y += 1
            i += 1
        i = 0
        x = currentPointer[1] + 1
        y = currentPointer[0] - 1
        while y < 8 and x < 8 and y >= 0 and x >= 0 :
            generated_board_points[getY(y)][getX(x)]['defence_value'].append(getVal(val, i))
            if [y, x] in enemy_figure_indexes:
                break
            x += 1
            y -= 1
            i += 1
        i = 0
        x = currentPointer[1] - 1
        y = currentPointer[0] - 1
        while y >= 0 and x >= 0 :
            generated_board_points[getY(y)][getX(x)]['defence_value'].append(getVal(val, i))
            if [y, x] in enemy_figure_indexes:
                break
            x -= 1
            y -= 1
            i += 1


        
    def generateValueByLine(self, currentPointer, generated_board_points, enemy_figure_indexes, val):
        i = 0
        x = currentPointer[1]
        y = currentPointer[0] + 1
        # print(f'x = ${x} and y = ${y} val = ${val}')
        while y < 8 :
            generated_board_points[getY(y)][getX(x)]['defence_value'].append(getVal(val, i))
            if [y, x] in enemy_figure_indexes:
                break
            y += 1
            i += 1

        i = 0
        x = currentPointer[1]
        y = currentPointer[0] - 1

        while y >= 0 :
            generated_board_points[getY(y)][getX(x)]['defence_value'].append(getVal(val, i))
            if [y, x] in enemy_figure_indexes:
                break
            y -= 1
            i += 1
        i = 0
        x = currentPointer[1] + 1
        y = currentPointer[0]
        while x < 8 :
            generated_board_points[getY(y)][getX(x)]['defence_value'].append(getVal(val, i))
            if [y, x] in enemy_figure_indexes:
                break
            x += 1
            i += 1
        i = 0
        x = currentPointer[1] - 1
        y = currentPointer[0]
        while x >= 0 :
            generated_board_points[getY(y)][getX(x)]['defence_value'].append(getVal(val, i))
            if [y, x] in enemy_figure_indexes:
                break
            x -= 1
            i += 1


        # return figure

    def generateValueForField(self, figure, posIdxTouple, is_white_move, generated_board_points, enemy_figure_indexes):
        
        currentPointer = posIdxTouple
        if figure.lower() == 'b': # laufer
            self.generateValueByCross(currentPointer, generated_board_points, enemy_figure_indexes, generate_val('b'))

        elif figure.lower() == 'r': # wierza
            self.generateValueByLine(currentPointer, generated_board_points, enemy_figure_indexes, generate_val('r'))

        elif figure.lower() == 'n': # koń
            field_coords = []
            field_coords.append((currentPointer[1] + 1, currentPointer[0] + 2))
            field_coords.append((currentPointer[1] + 2, currentPointer[0] + 1))
            
            field_coords.append((currentPointer[1] + 1, currentPointer[0] - 2))
            field_coords.append((currentPointer[1] + 2, currentPointer[0] - 1))
            
            field_coords.append((currentPointer[1] - 1, currentPointer[0] + 2))
            field_coords.append((currentPointer[1] - 2, currentPointer[0] + 1))

            field_coords.append((currentPointer[1] - 1, currentPointer[0] - 2))
            field_coords.append((currentPointer[1] - 2, currentPointer[0] - 1))

            for coord in field_coords:
                if (coord[1] < 8 and coord[1] >= 0) and (coord[0] < 8 and coord[0] >= 0) :
                    if [coord[1], coord[0]] in enemy_figure_indexes:
                        continue
                    generated_board_points[getY(coord[1])][getX(coord[0])]['defence_value'].append(generate_val('n'))
        elif figure.lower() == 'k': # king
            field_coords = []
            field_coords.append((currentPointer[1] + 1, currentPointer[0] + 1))
            field_coords.append((currentPointer[1] + 1, currentPointer[0] - 1))

            field_coords.append((currentPointer[1], currentPointer[0] - 1))
            field_coords.append((currentPointer[1], currentPointer[0] + 1))
            
            field_coords.append((currentPointer[1] + 1, currentPointer[0]))
            field_coords.append((currentPointer[1] - 1, currentPointer[0]))

            field_coords.append((currentPointer[1] - 1, currentPointer[0] + 1))
            field_coords.append((currentPointer[1] - 1, currentPointer[0] - 1))


            for coord in field_coords:
                if (coord[1] < 8 and coord[1] >= 0) and (coord[0] < 8 and coord[0] >= 0) :
                    generated_board_points[getY(coord[1])][getX(coord[0])]['defence_value'].append(generate_val('k'))

            newr_val = -0.8
            # # line fields
            # self.generateValueByLine(currentPointer, generated_board_points, enemy_figure_indexes, [newr_val + (0.2*i) for i in range(8)])

            # # cross fields 
            # self.generateValueByCross(currentPointer, generated_board_points, enemy_figure_indexes, [newr_val + (0.2*i) for i in range(8)])

        elif figure.lower() == 'p': # pionek
            field_coords = []

            if is_white_move:
                field_coords.append((currentPointer[1] + 1, currentPointer[0] - 1))
                field_coords.append((currentPointer[1] + 1, currentPointer[0] + 1))
                field_coords.append((currentPointer[1] + 1, currentPointer[0]))
                if currentPointer[1] == 6:
                    field_coords.append((currentPointer[1] + 2, currentPointer[0]))

            elif not is_white_move :
                
                field_coords.append((currentPointer[1] - 1, currentPointer[0] - 1))
                field_coords.append((currentPointer[1] - 1, currentPointer[0] + 1))
                field_coords.append((currentPointer[1] - 1, currentPointer[0]))
                if currentPointer[1] == 1:
                    field_coords.append((currentPointer[1] - 2, currentPointer[0]))


            for coord in field_coords:
                if (coord[1] < 8 and coord[1] >= 0) and (coord[0] < 8 and coord[0] >= 0) :
                    generated_board_points[getY(coord[1])][getX(coord[0])]['defence_value'].append(generate_val('p'))

        elif figure.lower() == 'q': # queen
            # line fields
            self.generateValueByLine(currentPointer, generated_board_points, enemy_figure_indexes, generate_val('q'))

            # cross fields 
            self.generateValueByCross(currentPointer, generated_board_points, enemy_figure_indexes, generate_val('q'))

                    
        return generated_board_points