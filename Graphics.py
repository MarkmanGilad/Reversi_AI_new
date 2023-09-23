from Constant import *
import numpy as np
import pygame

pygame.init()

class Graphics:
    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Reversi')

    def draw_Lines(self):
        for i in range(ROWS):
            pygame.draw.line(self.win, BLACK, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE , WIDTH), width=LINE_WIDTH)
            pygame.draw.line(self.win, BLACK, (0, i * SQUARE_SIZE), (HEIGHT, i * SQUARE_SIZE ), width=LINE_WIDTH)

    def draw_all_pieces(self, board):
        for row in range(ROWS):
            for col in range(COLS):
                if board[row][col] != 0 :
                    self.draw_piece((row, col), board[row][col])
            
    def draw_piece(self, row_col, player):
        center = self.calc_pos(row_col)
        radius = (SQUARE_SIZE) // 2 - PADDING
        color = self.calc_color(player)
        pygame.draw.circle(self.win, RED, center, radius+2)
        pygame.draw.circle(self.win,color , center, radius)

    def calc_pos(self, row_col):
        row, col = row_col
        y = row * SQUARE_SIZE + SQUARE_SIZE//2
        x = col * SQUARE_SIZE + SQUARE_SIZE//2
        return x, y

    def calc_base_pos(self, row_col):
        row, col = row_col
        y = row * SQUARE_SIZE
        x = col * SQUARE_SIZE
        return x, y

    def calc_row_col(self, pos):
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return row, col

    def calc_color(self, player):
        if player == 1:
            return WHITE
        elif player == -1:
            return BLACK
        else:
            return LIGHTGRAY

    def draw(self, board):
        self.win.fill(LIGHTGRAY)
        self.draw_Lines()
        self.draw_all_pieces(board)

    def draw_square(self, row_col, color):
        pos = self.calc_base_pos(row_col)
        pygame.draw.rect(self.win, color, (*pos, SQUARE_SIZE, SQUARE_SIZE))

    def blink(self, row_col, color):
        row, col = row_col
        player = self.board[row][col]
        for i in range (2):
            self.draw_square(row_col, color)
            if player:
                self.draw_piece(row_col, player) 
            pygame.display.update()
            pygame.time.delay(200)
            self.draw_square(row_col, LIGHTGRAY)
            if player:
                self.draw_piece(row_col, player) 
            pygame.display.update()
            pygame.time.delay(200)