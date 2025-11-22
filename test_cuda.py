#!/usr/bin/env python3
"""
Quick test to verify CUDA functionality before running full trainer
"""
import torch
from DQN_Agent_CNN import DQN_Agent
from Reversi import Reversi
from State import State

def test_cuda_setup():
    print("=== CUDA Test ===")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Selected device: {device}")
    
    # Test agent creation
    try:
        env = Reversi()
        agent = DQN_Agent(player=1, env=env, device=device)
        print(f"Agent created successfully on device: {agent.DQN.device}")
        
        # Test forward pass
        state = env.get_init_state()
        action = agent.get_Action(state, train=False)
        print(f"Forward pass successful, action: {action}")
        
        print("✅ All CUDA tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cuda_setup()