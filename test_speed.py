#!/usr/bin/env python3
"""
Test GPU performance of the optimized get_Actions method
"""
import torch
import time
from DQN_Agent_CNN import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Reversi import Reversi

def test_gpu_speed():
    print("=== GPU Speed Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create agent and environment
    env = Reversi()
    agent = DQN_Agent(player=1, env=env, device=device)
    agent_hat = DQN_Agent(player=1, env=env, train=False, device=device)
    
    # Create fake batch data similar to training
    batch_size = 256
    
    # Create dummy states and dones tensors
    dummy_boards = torch.randn(batch_size, 1, 8, 8, device=device)
    dummy_actions_list = []
    for i in range(batch_size):
        dummy_actions_list.append(torch.randn(4, 8, 8))  # 4 legal actions per state
    
    states_tensor = (dummy_boards, dummy_actions_list)
    dones = torch.zeros(batch_size, 1, device=device)
    
    # Warm up GPU
    for _ in range(10):
        _ = agent_hat.get_Actions(states_tensor, dones)
    
    # Time the get_Actions method
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    num_iterations = 50
    for i in range(num_iterations):
        _ = agent_hat.get_Actions(states_tensor, dones)
        
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    
    print(f"Average time per get_Actions call: {avg_time_ms:.1f}ms")
    print(f"Batch size: {batch_size}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f}MB" if device.type == 'cuda' else "CPU mode")
    
    return avg_time_ms

if __name__ == "__main__":
    test_gpu_speed()