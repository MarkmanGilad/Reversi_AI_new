from Random_Agent import Random_Agent
from Fix_Agent import Fix_Agent
from Reversi import Reversi

class Tester:
    def __init__(self, env, player1, player2) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2
        

    def test (self, games_num):
        env = self.env
        player = self.player1
        player1_win = 0
        player2_win = 0
        games = 0
        while games < games_num:
            action = player.get_Action(state=env.state, train = False)
            env.move(action, env.state)
            player = self.switchPlayers(player)
            if env.is_end_of_game(env.state):
                score = env.state.score()
                if score > 0:
                    player1_win += 1
                elif score < 0:
                    player2_win += 1
                env.state = env.get_init_state()
                games += 1
                player = self.player1
        return player1_win, player2_win        

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

    def __call__(self, games_num):
        return self.test(games_num)

if __name__ == '__main__':
    env = Reversi()
    player1 = Random_Agent(env, player=1)
    player2 = Fix_Agent(env, player=-1)
    test = Tester(env,player1, player2)
    print(test.test(100))
    player1 = Fix_Agent(env, player=1)
    player2 = Random_Agent(env, player=-1)
    test = Tester(env,player1, player2)
    print(test.test(100))