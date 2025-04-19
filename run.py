import argparse
import os
from omegaconf import OmegaConf
from reward_model import main as train_reward_model
from reinforce import main as train_reinforce
from prob_reward_model import main as train_prob_reward_model
from prob_reinforce import main as train_prob_reinforce

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1, choices=[1, 2], help="Level of the implementation: 1 or 2")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip_reward_training", action="store_true")
    parser.add_argument("--probabilistic_reward", action="store_true")
    parser.add_argument("--reward_model_path", type=str)
    parser.add_argument("--only_reward_model", action="store_true")    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
      
    if args.level == 1:
        reward_model_path = args.reward_model_path or cfg.reward_model.output_dir
        if not args.skip_reward_training or not os.path.exists(reward_model_path):
            reward_model_path = train_reward_model(args.config)

        if not args.only_reward_model:
            train_reinforce(args.config, reward_model_path)
    else: 
        reward_model_path = args.reward_model_path or cfg.prob_reward_model.output_dir
        if not args.skip_reward_training or not os.path.exists(reward_model_path):
            reward_model_path = train_prob_reward_model(args.config)
        
        if not args.only_reward_model:
            train_prob_reinforce(args.config, reward_model_path)
   

if __name__ == "__main__":
    main() 