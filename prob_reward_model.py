from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

class ProbRewardModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=10
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            prompts,
            completions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        logits = self.model(**inputs).logits  # [batch_size, 10]
        return logits


def compute_prob_loss(logits_w: torch.Tensor, logits_l: torch.Tensor) -> torch.Tensor:
    p_w = F.softmax(logits_w, dim=-1)  # [batch_size, 10]
    p_l = F.softmax(logits_l, dim=-1)  # [batch_size, 10]
    # Create mask matrix where i>j
    gt = torch.triu(torch.ones(10, 10, device=p_w.device), diagonal=1)
    # Compute joint probability p_w(i) * p_l(j)
    joint = p_w.unsqueeze(2) * p_l.unsqueeze(1)  # [batch_size, 10, 10]
    # Sum only over pairs where i>j
    pref_prob = (joint * gt).sum(dim=(1, 2))  # [batch_size]
    # Negative log-likelihood
    loss = -torch.log(pref_prob + 1e-8).mean()
    return loss


def main(config_path: str = "config.yaml"):
    cfg = OmegaConf.load(config_path)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    wandb.init(
        project=cfg.project_name,
        name=cfg.prob_reward_model.wandb_run_name,
        config={
            "sft_model": cfg.sft_model_name,
            "reward_model": cfg.prob_reward_model.output_dir,
            "dataset": cfg.dataset_name,
            "learning_rate": cfg.reward_model.learning_rate,
            "batch_size": cfg.reward_model.batch_size,
            "num_train_epochs": cfg.reward_model.num_train_epochs,
            "fp16": cfg.reward_model.fp16,
            "optimizer": cfg.reward_model.optimizer,
            "validation_split": cfg.reward_model.validation_split,
        }
    )

    model = ProbRewardModel(cfg.sft_model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.reward_model.learning_rate)

    print("Dataset preparing...")
    dataset = load_dataset(cfg.dataset_name)[cfg.dataset_split]
    dataset = dataset.remove_columns(["chosen_rating", "rejected_rating"])

    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=cfg.reward_model.validation_split, 
        random_state=cfg.seed
    )
    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(val_indices)

    print("Training Prob Reward Model...")
    model.train()
    total_loss = 0.0
    num_batches = len(train_dataset) // cfg.reward_model.batch_size

    for batch_idx, start in tqdm(enumerate(range(0, len(train_dataset), cfg.reward_model.batch_size), start=1), total=num_batches):
        batch = train_dataset[start:start + cfg.reward_model.batch_size]
        prompts = batch["prompt"]
        chosen = [msgs[-1]["content"] for msgs in batch["chosen"]]
        rejected = [msgs[-1]["content"] for msgs in batch["rejected"]]

        logits_w = model(prompts, chosen)
        logits_l = model(prompts, rejected)
        loss = compute_prob_loss(logits_w, logits_l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train_loss": loss.item()})
    
    model.model.save_pretrained(cfg.prob_reward_model.output_dir)
    model.tokenizer.save_pretrained(cfg.prob_reward_model.output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
