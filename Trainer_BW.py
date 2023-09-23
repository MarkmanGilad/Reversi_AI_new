from Reversi import Reversi

from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
from Fix_Agent import Fix_Agent
import torch
from Tester import Tester

epochs = 2000000
start_epoch = 0
C = 1000
learning_rate = 0.01
batch_size = 64
env = Reversi()
MIN_Buffer = 4000

path_load= None
path_Save='Data/params_4.pth'
path_best = 'Data/best_params_4.pth'
buffer_path = 'Data/buffer_4.pth'
results_path='Data/results_4.pth'
random_results_path = 'Data/random_results_4.pth'
path_best_random = 'Data/best_random_params_4.pth'


def main ():
    
    player1 = DQN_Agent(player=1, env=env,parametes_path=path_load)
    player2 = DQN_Agent(player=-1, env=env,parametes_path=None)
    player2.DQN = player1.DQN

    buffer = ReplayBuffer(path=None)
    Q = player1.DQN
    Q_hat = Q.copy()
    Q_hat.train = False
    
    results = []
    avgLosses = []
    avgLoss = 0
    loss = torch.Tensor([0])
    res = 0
    best_res = -200
    loss_count = 0
    tester_random = Tester(player1=player1, player2=Random_Agent(player=-1, env=env), env=env)
    # tester_fix = Tester(player1=Fix_Agent(env=env,player=1, train=False), player2=player2, env=env)
    random_results = []
    best_random = -100
        
    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,1000, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[30*50000, 30*100000, 30*250000, 30*500000], gamma=0.5)
    
    for epoch in range(start_epoch, epochs):
        print(f'epoch = {epoch}', end='\r')
        state_1 = env.get_init_state()
        state_2, action_2, after_state_2 = None, None, None
        while not env.is_end_of_game(state_1):
            # Sample Environement
            action_1 = player1.get_Action(state_1, epoch=epoch)
            after_state_1 = env.get_next_state(state=state_1, action=action_1)
            reward_1, end_of_game_1 = env.reward(after_state_1)
            if state_2: # not the first action in a game
                    buffer.push(state_2, action_2, reward_1, after_state_1, end_of_game_1)
            if end_of_game_1:
                res += reward_1
                buffer.push(state_1, action_1, reward_1, after_state_1, True)
                state_1 = after_state_1
            else:
                state_2 = after_state_1
                action_2 = player2.get_Action(state=state_2)
                after_state_2 = env.get_next_state(state=state_2, action=action_2)
                reward_2, end_of_game_2 = env.reward(state=after_state_2)
                if end_of_game_2:
                    res += reward_2
                    buffer.push(state_2, action_2, reward_2, after_state_2, True)
                buffer.push(state_1, action_1, reward_2, after_state_2, end_of_game_2)
                state_1 = after_state_2

            if len(buffer) < MIN_Buffer:
                continue
            
            # Train NN
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = Q(states[0], actions)
            next_actions = player1.get_Actions(next_states, dones)
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states[0], next_actions)

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if epoch % C == 0:
                Q_hat.load_state_dict(Q.state_dict())
            scheduler.step()
            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 
            

        if (epoch+1) % 100 == 0:
            print(f'\nres= {res}')
            avgLosses.append(avgLoss)
            results.append(res)
            # if best_res < res:      
            #     best_res = res
            #     if best_res > 75 and tester_fix(1) == (1,0):
            #         player1.save_param(path_best)
            res = 0

        if (epoch+1) % 1000 == 0:
            test = tester_random(100)
            test_score = test[0]-test[1]
            if best_random < test_score:
                best_random = test_score
                player1.save_param(path_best_random)
            print(test)
            random_results.append(test_score)

        if (epoch+1) % 5000 == 0:
            torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses}, results_path)
            torch.save(buffer, buffer_path)
            player1.save_param(path_Save)
            torch.save(random_results, random_results_path)
        
        if len(buffer) > MIN_Buffer:
            print (f'epoch={epoch} loss={loss:.5f} Q_values[0]={Q_values[0].item():.3f} avgloss={avgLoss:.5f}', end=" ")
            print (f'learning rate={scheduler.get_last_lr()[0]} path={path_Save} res= {res} best_res = {best_res}')

    torch.save({'epoch': epoch, 'results': results, 'avglosses':avgLosses}, results_path)
    torch.save(buffer, buffer_path)
    torch.save(random_results, random_results_path)

if __name__ == '__main__':
    main()


