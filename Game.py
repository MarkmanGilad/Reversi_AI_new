import pygame
from Graphics import *
from Reversi import Reversi
from Human_Agent import Human_Agent
from Constant import *

env = Reversi()
graphics = Graphics()
player1 = Human_Agent(player = 1)
player2 = Human_Agent(player = -1)

def main ():
    run = True
    clock = pygame.time.Clock()
    graphics.draw(env.state.board)
    player = player1

    while(run):
        clock.tick(FPS)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               run = False
               
        action = player.get_Action(events=events, graphics= graphics, state= env.state)
        if action:
            if (env.move(action, env.state)):
                graphics.blink(action, GREEN)
                player = switchPlayers(player)
            else:
                graphics.blink(action, RED)
        
        graphics.draw(env.state.board)
        pygame.display.update()
        if env.is_end_of_game(env.state):
            # run = False
            score = env.state.score()
            print ("score = ", score)
            pygame.time.delay(200)
            env.state = env.get_init_state()
            graphics.board = env.state.board
            player = player1
            graphics.draw(env.state.board)
    pygame.time.delay(2000) 
    pygame.quit()
    print("End of game")
    score = env.state.score()
    print ("score = ", score)
    
def switchPlayers(player):
    if player == player1:
       return player2
    else:
        return player1

if __name__ == '__main__':
    main()