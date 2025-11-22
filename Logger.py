import os
import wandb
import torch
from collections import deque


class WandB:
    def __init__(self, project_name, chkpt, config, resume):
        self.wandb_dict = {}
        wandb.init(
            project=project_name,
            resume=resume,
            id=f'{project_name} {chkpt}',
            config=config,
        )

    def update_dict (self, **kwds):
        self.wandb_dict.update(kwds)

    def log (self):
        wandb.log(self.wandb_dict)
        self.wandb_dict = {}

    def __call__(self, *args, **kwds):
        self.update_dict(**kwds)
    
class Logger:
        
    def __init__(self, chkpt, maxlen = 100):
        self.chkpt = chkpt
        self.log_dict = {}
        self.maxlen = maxlen
    
    def log (self, key, value):
        if key not in self.log_dict:
            self.log_dict[key] = deque(maxlen=self.maxlen)    
        self.log_dict[key].append(value)

    def save (self):
        torch.save(self.log_dict, f'Data/logger{self.chkpt}.pth',)

    def load (self):
        self.log_dict = torch.load(f'Data/logger{self.chkpt}.pth', weights_only=False)
    
    def get_resume_info(self):
        """Get critical info for resuming training"""
        try:
            self.load()
            current_epoch = self.log_dict.get('current_epoch', [0])[-1] if self.log_dict.get('current_epoch') else 0
            best_res = self.log_dict.get('best_res', [-200])[-1] if self.log_dict.get('best_res') else -200
            best_random = self.log_dict.get('best_random', [0])[-1] if self.log_dict.get('best_random') else 0
            return current_epoch, best_res, best_random
        except:
            return 0, -200, 0  # defaults if no checkpoint
    
    def print_key(self, key =None, range=10):
        print(key)
        print(list(self.log_dict[key])[-range:])
    
    def print_all(self):
        for key, item in self.log_dict.items():
            print (key, "\t", item)
    
    def print_keys(self):
        for key in self.log_dict:
            print(key)
    
    @staticmethod
    def get_next_file_num(file_num=None, default_start=105):
        """Get the next file number for training. 
        If file_num is provided, use it. Otherwise auto-advance from saved counter."""
        file_num_path = 'Data/current_file_num.pth'
        
        if file_num is not None:
            # Use provided file_num, don't advance counter
            return file_num
            
        # Auto-advance mode
        try:
            # Try to load existing file number
            current_num = torch.load(file_num_path, weights_only=False)
        except:
            # First time or file doesn't exist, use default
            current_num = default_start
            
        # Advance counter and save for next time
        next_num = current_num + 1
        torch.save(next_num, file_num_path)
        
        return current_num
    
    @staticmethod
    def save_file_num(file_num):
        """Save current file number to Data folder"""
        file_num_path = 'Data/current_file_num.pth'
        torch.save(file_num, file_num_path)

