from Reversi import Reversi

from DQN_Agent_CNN import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
from Fix_Agent import Fix_Agent
from Action import Action
from Logger import WandB, Logger
from DQN_CNN import gamma
from Constant import (epsilon_start, epsilon_final, epsilon_decay, 
                      C, LEARNING_RATE, BATCH_SIZE, MIN_BUFFER,
                      SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA,
                      TEST_FREQUENCY, TEST_GAMES, SAVE_FREQUENCY, LOG_FREQUENCY,
                      ROWS, COLS)
import torch
from Tester import Tester

epochs = 2000000
start_epoch = 0
env = Reversi()

# Auto-increment file number or use specific number
File_Num = Logger.get_next_file_num(file_num=None)  # Set to None for auto-increment, or specify a number
print("Starting test number:", File_Num)
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
    
    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[30*50000, 30*100000, 30*250000, 30*500000], gamma=0.5)
    
    # initialize local logger and wandb
    logger = Logger(File_Num)
    config = {
        'project': 'Reversi_CNN',
        'file_num': File_Num,
        'checkpoint_path': path_Save,
        'buffer_path': buffer_path,
        'best_model_path': path_best,
        'best_random_path': path_best_random,
        # Training parameters
        'epochs': epochs,
        'start_epoch': start_epoch,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'target_update_freq': C,
        'min_buffer': MIN_BUFFER,
        'gamma': gamma,
        'epsilon_start': epsilon_start,
        'epsilon_final': epsilon_final,
        'epsilon_decay': epsilon_decay,
        # Model architecture (dynamic)
        'model_type': type(Q).__name__,
        'model_architecture': str(Q),
        'device': str(Q.device),
        # Optimizer details
        'optimizer': type(optim).__name__,
        'scheduler_type': type(scheduler).__name__,
        'scheduler_step_size': SCHEDULER_STEP_SIZE,
        'scheduler_gamma': SCHEDULER_GAMMA,
        # Environment/Game parameters (dynamic)
        'board_size': f'{ROWS}x{COLS}',
        'terminal_reward': env.EOG_reward,
        'player1': type(player1).__name__,
        'player2': f'{type(player2).__name__}(random={player2.random if hasattr(player2, "random") else "N/A"})',
        # Test/Log frequencies
        'test_frequency': TEST_FREQUENCY,
        'test_games': TEST_GAMES,
        'tester_opponent': type(tester.player2).__name__,
        'save_frequency': SAVE_FREQUENCY,
        'log_frequency': LOG_FREQUENCY,
    }
    wandb = WandB(project_name='Reversi_CNN', chkpt=File_Num, config=config, resume=False)
    
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

            if len(buffer) < MIN_BUFFER:
                continue
            
            # Train NN
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

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
            next_actions = player1.get_Actions(next_states, dones) # returns (B,2) coords ### DDQN
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
            
            avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
            loss_count += 1
            
            
        if epoch % C == 0:
                Q_hat.load_state_dict(Q.state_dict())

        if (epoch+1) % LOG_FREQUENCY == 0:
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

        if (epoch+1) % TEST_FREQUENCY == 0:
            test = tester(TEST_GAMES)
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

        if (epoch+1) % SAVE_FREQUENCY == 0:
            torch.save(buffer, buffer_path)
            player1.save_param(path_Save)
            # save only critical info for resume: current epoch, best scores
            try:
                logger.log('current_epoch', epoch+1)
                logger.save()
            except Exception:
                pass
        if len(buffer) > MIN_BUFFER:
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


