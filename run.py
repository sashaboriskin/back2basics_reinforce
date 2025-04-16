import argparse
import os
from omegaconf import OmegaConf
from reward_model import main as train_reward_model
from reinforce import main as train_reinforce

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip_reward_training", action="store_true")
    parser.add_argument("--reward_model_path", type=str)
    parser.add_argument("--only_reward_model", action="store_true")    
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
  
    reward_model_path = args.reward_model_path or cfg.reward_model.output_dir
    
    if not args.skip_reward_training or not os.path.exists(reward_model_path):
        reward_model_path = train_reward_model(args.config)

    if args.only_reward_model:
        return
    
    train_reinforce(args.config, reward_model_path)
   


if __name__ == "__main__":
    main() 