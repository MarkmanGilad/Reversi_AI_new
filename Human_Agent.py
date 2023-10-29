import pygame
from Graphics import Graphics
from State import State


class Human_Agent:

    def __init__(self, player: int) -> None:
        self.player = player

    def get_Action (self, events= None, graphics: Graphics = None, state : State = None, train = False):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row_col = graphics.calc_row_col(pos)
                if row_col in state.legal_actions:
                    return row_col
        return None
            