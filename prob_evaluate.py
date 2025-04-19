import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from omegaconf import OmegaConf
import random
from sklearn.model_selection import train_test_split

def main(config_path="config.yaml", checkpoint_path="prob_reinforce_model"):
    cfg = OmegaConf.load(config_path)
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    
    print("Dataset preparing...")
    dataset = load_dataset(cfg.dataset_name)[cfg.dataset_split]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "prompt"])
    
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=cfg.reward_model.validation_split, 
        random_state=cfg.seed
    )
    eval_dataset = dataset.select(val_indices)
    
    sft_model = AutoModelForCausalLM.from_pretrained(cfg.sft_model_name).to(device)
    sft_tokenizer = AutoTokenizer.from_pretrained(cfg.sft_model_name)
    if not sft_tokenizer.pad_token:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token
    
    trained_model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
    trained_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.prob_reward_model.output_dir, 
        num_labels=10
    ).to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg.prob_reward_model.output_dir)
    
    def generate_response(model, tokenizer, prompt, temperature=cfg.reinforce.temperature):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length // 2).to(device)
        
        outputs = model.generate(
            **inputs,
            max_length=cfg.max_length,
            pad_token_id=tokenizer.pad_token_id,
            temperature=temperature,
            do_sample=True,
        )
        
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response
    
    def compute_reward(prompt, response):
        with torch.no_grad():
            inputs = reward_tokenizer(
                prompt + response,
                return_tensors="pt", 
                truncation=True, 
                max_length=cfg.max_length
            ).to(device)

            logits = reward_model(**inputs).logits  # [1, 10]
            probs = F.softmax(logits, dim=-1)       # [1, 10]
            scores = torch.arange(1, 11, device=device, dtype=probs.dtype)
            expected_score = (probs * scores).sum(dim=1).item()
            
        return expected_score
    
    print("Initial val reward...")
    sft_rewards = []
    for i in tqdm(range(0, len(eval_dataset), cfg.reinforce.batch_size)):
        batch = eval_dataset[i:i+cfg.reinforce.batch_size]
        prompts = batch["prompt"]
        
        for prompt in prompts:
            response = generate_response(sft_model, sft_tokenizer, prompt)
            reward = compute_reward(prompt, response)
            sft_rewards.append(reward)
    
    sft_avg_reward = np.mean(sft_rewards)
    print(f"Initial validation reward: {sft_avg_reward:.4f}")
    
    print("Final validation reward...")
    trained_rewards = []
    for i in tqdm(range(0, len(eval_dataset), cfg.reinforce.batch_size)):
        batch = eval_dataset[i:i+cfg.reinforce.batch_size]
        prompts = batch["prompt"]
        
        for prompt in prompts:
            response = generate_response(trained_model, trained_tokenizer, prompt)
            reward = compute_reward(prompt, response)
            trained_rewards.append(reward)
    
    trained_avg_reward = np.mean(trained_rewards)
    print(f"Final validation reward: {trained_avg_reward:.4f}")
    print(f"Improvement: {trained_avg_reward - sft_avg_reward:.4f}")
    
    def compute_reward_distribution(prompt, response):
        with torch.no_grad():
            inputs = reward_tokenizer(
                prompt + response,
                return_tensors="pt", 
                truncation=True, 
                max_length=cfg.max_length
            ).to(device)

            logits = reward_model(**inputs).logits  # [1, 10]
            probs = F.softmax(logits, dim=-1)[0]   # [10]
            
        return probs.cpu().numpy()
    
    print("\nExamples:")
    num_examples = 5
    sample_indices = random.sample(range(len(eval_dataset)), num_examples)
    
    for i, idx in enumerate(sample_indices, 1):
        prompt = eval_dataset[idx]["prompt"]
        
        sft_response = generate_response(sft_model, sft_tokenizer, prompt)
        sft_reward = compute_reward(prompt, sft_response)
        sft_dist = compute_reward_distribution(prompt, sft_response)
        
        trained_response = generate_response(trained_model, trained_tokenizer, prompt)
        trained_reward = compute_reward(prompt, trained_response)
        trained_dist = compute_reward_distribution(prompt, trained_response)
        
        print(f"\nExample {i}:")
        print(f"Prompt: {prompt}")
        print(f"\nOriginal model (reward: {sft_reward:.4f}):")
        print(sft_response)
        print("Reward distribution:")
        for rating, prob in enumerate(sft_dist, 1):
            print(f"  Rating {rating}: {prob:.4f}")
        print("-" * 80)
        print(f"\nTrained model (reward: {trained_reward:.4f}):")
        print(trained_response)
        print("Reward distribution:")
        for rating, prob in enumerate(trained_dist, 1):
            print(f"  Rating {rating}: {prob:.4f}")
        print("-" * 80)


if __name__ == "__main__":
    main()