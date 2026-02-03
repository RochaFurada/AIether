import sys
import os

# Path configuration for package import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiether.training.trainer import train_with_hf_trainer

if __name__ == "__main__":
    train_with_hf_trainer()
