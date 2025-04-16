import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import wandb
import os
import random
from omegaconf import OmegaConf
from typing import List, Dict, Tuple, Optional


def main(config_path="config.yaml", reward_model_path=None):
    cfg = OmegaConf.load(config_path)
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    if reward_model_path:
        cfg.reward_model.output_dir = reward_model_path

    wandb.init(
        project=cfg.project_name,
        name=cfg.reinforce.wandb_run_name,
        config={
            "sft_model": cfg.sft_model_name,
            "reward_model": cfg.reward_model.output_dir,
            "dataset": cfg.dataset_name,
            "learning_rate": cfg.reinforce.learning_rate,
            "batch_size": cfg.reinforce.batch_size,
            "iterations": cfg.reinforce.num_iterations,
            "gamma": cfg.reinforce.gamma,
            "temperature": cfg.reinforce.temperature,
        }
    )

    print("Dataset preparing...")
    
    dataset = load_dataset(cfg.dataset_name)[cfg.dataset_split]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "prompt"])
    
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=cfg.reward_model.validation_split, 
        random_state=cfg.seed
    )
    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(val_indices)
    
    sft_model = AutoModelForCausalLM.from_pretrained(cfg.sft_model_name).to(device)
    sft_tokenizer = AutoTokenizer.from_pretrained(cfg.sft_model_name)
    if not sft_tokenizer.pad_token:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(cfg.reward_model.output_dir, num_labels=1).to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model.output_dir)
    optimizer = Adam(sft_model.parameters(), lr=cfg.reinforce.learning_rate)
    
    def generate_responses(prompts: List[str], temperature: float = cfg.reinforce.temperature) -> List[str]:
        responses = []
        for prompt in prompts:
            inputs = sft_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length // 2).to(device)
            
            outputs = sft_model.generate(
                **inputs,
                max_length=cfg.max_length,
                pad_token_id=sft_tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=temperature,
                do_sample=True,
            )
            
            response_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            response = sft_tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def compute_log_probs(prompts: List[str], responses: List[str]) -> torch.Tensor:
        log_probs = []
        
        for prompt, response in zip(prompts, responses):
            input_text = prompt + response
            inputs = sft_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=cfg.max_length).to(device)
            
            outputs = sft_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            logits = outputs.logits
            
            prompt_tokens = sft_tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            
            response_logits = logits[0, prompt_tokens-1:-1, :]
            response_tokens = inputs.input_ids[0, prompt_tokens:]
            
            token_log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = torch.gather(token_log_probs, 1, response_tokens.unsqueeze(1)).squeeze(1)
            
            log_prob = token_log_probs.sum()
            log_probs.append(log_prob)
        
        return torch.stack(log_probs)
    
    def compute_rewards(prompts: List[str], responses: List[str]) -> torch.Tensor:
        rewards = []
        
        with torch.no_grad():
            for prompt, response in zip(prompts, responses):
                inputs = reward_tokenizer(
                    prompt + response,
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=cfg.max_length
                ).to(device)

                outputs = reward_model(**inputs)
                reward = outputs.logits[0].item()
                
                rewards.append(reward)

        return torch.tensor(rewards, device=device)
    
    def reinforce_step(prompts: List[str], responses: List[str], baseline: float) -> Tuple[torch.Tensor, float, float]:
        log_probs = compute_log_probs(prompts, responses)
        rewards = compute_rewards(prompts, responses)
        advantages = rewards - baseline
        policy_loss = -(log_probs * advantages.detach()).mean()
        new_baseline = cfg.reinforce.gamma * baseline + (1 - cfg.reinforce.gamma) * rewards.mean().item()
        return policy_loss, new_baseline, rewards.mean().item()
    
    baseline = 0.0
    os.makedirs(cfg.reinforce.output_dir, exist_ok=True)
    
    # print("Initial_val_reward...")
    # val_rewards = []
    # for i in tqdm(range(0, len(eval_dataset), cfg.reinforce.batch_size)):
    #     batch = eval_dataset[i:i+cfg.reinforce.batch_size]
    #     prompts = batch["prompt"]
    #     responses = generate_responses(prompts)
    #     rewards = compute_rewards(prompts, responses)
    #     val_rewards.extend(rewards.cpu().numpy())
    
    # initial_reward = np.mean(val_rewards)
    # print(f"Initial validation reward: {initial_reward:.4f}")
    # wandb.log({"initial_val_reward": initial_reward})
    initial_reward = 0.0
    
    print("Training Reinforce...")
    for iteration in tqdm(range(cfg.reinforce.num_iterations)):
        batch_indices = random.sample(range(len(train_dataset)), cfg.reinforce.batch_size)
        batch = train_dataset.select(batch_indices)
        prompts = batch["prompt"]
        
        responses = generate_responses(prompts)
        
        loss, baseline, avg_reward = reinforce_step(prompts, responses, baseline)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        wandb.log({
            "loss": loss.item(),
            "baseline": baseline,
            "average_reward": avg_reward,
            "iteration": iteration
        })
        
        if (iteration + 1) % cfg.reinforce.validation_interval == 0:
            val_rewards = []
            for i in range(0, min(len(eval_dataset), 100), cfg.reinforce.batch_size):
                batch = eval_dataset[i:i+cfg.reinforce.batch_size]
                prompts = batch["prompt"]
                responses = generate_responses(prompts)
                rewards = compute_rewards(prompts, responses)
                val_rewards.extend(rewards.cpu().numpy())
            
            val_reward = np.mean(val_rewards)
            print(f"Iteration {iteration+1}, Validation reward: {val_reward:.4f}")
            wandb.log({
                "val_reward": val_reward,
                "iteration": iteration
            })
            
            checkpoint_dir = os.path.join(cfg.reinforce.output_dir, f"checkpoint-{iteration+1}")
            sft_model.save_pretrained(checkpoint_dir)
            sft_tokenizer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved in {checkpoint_dir}")
    
    sft_model.save_pretrained(cfg.reinforce.output_dir)
    sft_tokenizer.save_pretrained(cfg.reinforce.output_dir)
    
    print("Final validation reward...")
    val_rewards = []
    for i in range(0, len(eval_dataset), cfg.reinforce.batch_size):
        batch = eval_dataset[i:i+cfg.reinforce.batch_size]
        prompts = batch["prompt"]
        responses = generate_responses(prompts)
        rewards = compute_rewards(prompts, responses)
        val_rewards.extend(rewards.cpu().numpy())
    
    final_reward = np.mean(val_rewards)
    print(f"Final validation reward: {final_reward:.4f}")
    print(f"Improvement: {final_reward - initial_reward:.4f}")
    
    wandb.log({
        "final_val_reward": final_reward,
        "improvement": final_reward - initial_reward
    })
    wandb.finish()

if __name__ == "__main__":
    main() 