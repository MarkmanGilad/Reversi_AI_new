from Reversi import Reversi

from DQN_Agent_CNN import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
from Fix_Agent import Fix_Agent
from Action import Action
from Logger import WandB, Logger
from Constant import epsilon_start, epsilon_final, epsiln_decay
import torch
from Tester import Tester

epochs = 2000000
start_epoch = 0
C = 350
learning_rate = 0.01
batch_size = 64
env = Reversi()
MIN_Buffer = 4000

File_Num = 101
path_load= None
path_Save=f'Data/params_{File_Num}.pth'
path_best = f'Data/best_params_{File_Num}.pth'
buffer_path = f'Data/buffer_{File_Num}.pth'
path_best_random = f'Data/best_random_params_{File_Num}.pth'


def main ():
    
    player1 = DQN_Agent(player=1, env=env,parametes_path=path_load)
    player_hat = DQN_Agent(player=1, env=env, train=False)
    Q = player1.DQN
    Q_hat = Q.copy()
    Q_hat.train = False
    player_hat.DQN = Q_hat
    
    player2 = Fix_Agent(player=-1, env=env, train=True, random=0.1)   #0.1
    # player2 = Random_Agent(player=-1, env=env)   
    buffer = ReplayBuffer(path=None) # None
    
    # metrics now tracked in logger
    avgLoss = 0 #avgLosses[-1] #0
    loss = torch.Tensor([0])
    res = 0
    best_res = -200
    loss_count = 0
    tester = Tester(player1=player1, player2=Random_Agent(player=-1, env=env), env=env)
    tester_fix = Tester(player1=player1, player2=player2, env=env)
    # test scores now tracked in logger
    best_random = 0 #max(random_results)
    
    # initialize local logger and wandb
    logger = Logger(File_Num)
    config = {
        'project': 'Reversi_CNN',
        'name': f'Reversi_CNN {File_Num}',
        'file_num': File_Num,
        'checkpoint_path': path_Save,
        # Training parameters
        'epochs': epochs,
        'start_epoch': start_epoch,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'C': C,  # target network update frequency
        'MIN_Buffer': MIN_Buffer,
        # Model architecture
        'model_type': 'DQN_CNN',
        'model_architecture': str(Q),
        'input_shape': '(B,2,8,8)',  # board + action planes
        'conv_layers': '32->64->128',
        'pooling': 'AvgPool2d(8)',
        'activation': 'LeakyReLU(0.01)',
        'output_size': 1,
        # Optimizer details
        'optimizer': 'Adam',
        'scheduler_type': 'StepLR',
        'scheduler_step_size': 100000*30,
        'scheduler_gamma': 0.90,
        # Environment/Game parameters
        'board_size': '8x8',
        'reward_system': 'normalized_piece_diff + terminal_bonus',
        'terminal_reward': f'+{env.EOG_reward}/-{env.EOG_reward}',  # actual values
        'step_reward': 'piece_diff/64',
        'player1': 'DQN_Agent_CNN',
        'player2': 'Fix_Agent(random=0.1)',
        # Training setup
        'device': str(Q.device),
        'gamma': 0.99,  # from DQN
        'epsilon_start': epsilon_start,  # actual value from Constant
        'epsilon_final': epsilon_final,  # actual value from Constant 
        'epsilon_decay': epsiln_decay,   # actual value from Constant
        # File paths
        'buffer_path': buffer_path,
        'best_model_path': path_best,
        'best_random_path': path_best_random,
        # Test setup
        'test_frequency': 1000,  # epochs
        'test_games': 100,
        'tester_opponent': 'Random_Agent',
        'save_frequency': 5000,  # epochs
        'log_frequency': 100,   # epochs
    }
    wandb = WandB(project_name='Reversi_CNN', chkpt=File_Num, config=config, resume=False)
    
    
    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,100000*30, gamma=0.90)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[30*50000, 30*100000, 30*250000, 30*500000], gamma=0.5)
    
    for epoch in range(start_epoch, epochs):
        print(f'epoch = {epoch}', end='\r')
        state_1 = env.get_init_state()
        while not env.is_end_of_game(state_1):
            # Sample Environement
            action_1 = player1.get_Action(state_1, epoch=epoch)
            after_state_1 = env.get_next_state(state=state_1, action=action_1)
            reward_1, end_of_game_1 = env.reward(state_1, action_1)
            if end_of_game_1:
                # Count win/loss instead of accumulating reward
                game_outcome = env.get_game_outcome(after_state_1, player=1)
                res += game_outcome
                buffer.push(state_1, action_1, reward_1, after_state_1, True)
                break
            state_2 = after_state_1
            action_2 = player2.get_Action(state=state_2)
            after_state_2 = env.get_next_state(state=state_2, action=action_2)
            reward_2, end_of_game_2 = env.reward(state_2, action_2)
            if end_of_game_2:
                # Count win/loss instead of accumulating reward
                game_outcome = env.get_game_outcome(after_state_2, player=1)
                res += game_outcome
            buffer.push(state_1, action_1, reward_2, after_state_2, end_of_game_2)
            state_1 = after_state_2

            if len(buffer) < MIN_Buffer:
                continue
            
            # Train NN
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            # Prepare board batch: states[0] is (B,1,8,8) -> squeeze to (B,8,8)
            board_batch = states[0].squeeze(1)

            # Convert coordinate actions (B,2) to (B,8,8) one-hot planes
            action_planes = Action.coords_to_planes(actions, device=board_batch.device)
            # Zero-out planes for terminal entries
            non_terminal = (dones.view(-1) == 0)
            if not non_terminal.all():
                action_planes[~non_terminal] = 0.0

            Q_values = Q(board_batch, action_planes)

            # next actions: convert similarly
            next_actions = player_hat.get_Actions(next_states, dones) # returns (B,2) coords
            next_board_batch = next_states[0].squeeze(1)
            next_action_planes = Action.coords_to_planes(next_actions, device=next_board_batch.device)
            non_terminal_next = (dones.view(-1) == 0)
            if not non_terminal_next.all():
                next_action_planes[~non_terminal_next] = 0.0

            with torch.no_grad():
                Q_hat_Values = Q_hat(next_board_batch, next_action_planes) #todo: use the values calculated in get_Actions

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            scheduler.step()
            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 
            
        if epoch % C == 0:
                Q_hat.load_state_dict(Q.state_dict())

        if (epoch+1) % 100 == 0:
            print(f'\nres= {res}')
            # log metrics to wandb only
            try:
                wandb(res=res, avgLoss=avgLoss, best_res=best_res)
                wandb.log()
            except Exception:
                pass
            if best_res < res:      
                best_res = res
                # Save best_res to logger for resume capability
                logger.log('best_res', best_res)
                if best_res > 75 and tester_fix(1) == (1,0):
                    player1.save_param(path_best)
            res = 0

        if (epoch+1) % 1000 == 0:
            test = tester(100)
            test_score = test[0]-test[1]
            if best_random < test_score and tester_fix(1) == (1,0):
                best_random = test_score
                # Save best_random to logger for resume capability
                logger.log('best_random', best_random)
                player1.save_param(path_best_random)
            print(test)
            try:
                wandb(test_player1_win=test[0], test_player2_win=test[1], test_score=test_score)
                wandb.log()
            except Exception:
                pass

        if (epoch+1) % 5000 == 0:
            torch.save(buffer, buffer_path)
            player1.save_param(path_Save)
            # save only critical info for resume: current epoch, best scores
            try:
                logger.log('current_epoch', epoch+1)
                logger.save()
            except Exception:
                pass
        if len(buffer) > MIN_Buffer:
            try:
                wandb(train_loss=loss.item(), Q0=Q_values[0].item(), avgLoss=avgLoss)
                wandb.log()
            except Exception:
                pass
            print (f'epoch={epoch} loss={loss:.5f} Q_values[0]={Q_values[0].item():.3f} avgloss={avgLoss:.5f}', end=" ")
            print (f'learning rate={scheduler.get_last_lr()[0]} path={path_Save} res= {res} best_res = {best_res}')

    torch.save(buffer, buffer_path)
    try:
        logger.save()  # final save of all metrics
    except Exception:
        pass

if __name__ == '__main__':
    main()


