import pygame
from Graphics import Graphics

class Human_Agent:

    def __init__(self, player: int) -> None:
        self.player = player

    def get_Action (self, events= None, graphics: Graphics = None, state = None, train = False):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row_col = graphics.calc_row_col(pos)
                pygame.time.delay(200) 
                return row_col
        return None
            