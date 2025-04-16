from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
import wandb


def main(config_path="config.yaml"):
    cfg = OmegaConf.load(config_path)
    
    wandb.init(
        project=cfg.project_name,
        name=cfg.reward_model.wandb_run_name,
        config={
            "sft_model": cfg.sft_model_name,
            "reward_model": cfg.reward_model.output_dir,
            "dataset": cfg.dataset_name,
            "learning_rate": cfg.reward_model.learning_rate,
            "batch_size": cfg.reward_model.batch_size,
            "num_train_epochs": cfg.reward_model.num_train_epochs,
            "fp16": cfg.reward_model.fp16,
            "optimizer": cfg.reward_model.optimizer,
            "validation_split": cfg.reward_model.validation_split,
        }
    )

    model = AutoModelForSequenceClassification.from_pretrained(cfg.sft_model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(cfg.dataset_name)[cfg.dataset_split]
    dataset = dataset.remove_columns(["prompt", "chosen_rating", "rejected_rating"])

    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=cfg.reward_model.validation_split, 
        random_state=cfg.seed
    )
    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(val_indices)
    
    reward_config = RewardConfig(
        output_dir=cfg.reward_model.output_dir,
        learning_rate=cfg.reward_model.learning_rate,
        per_device_train_batch_size=cfg.reward_model.batch_size,
        per_device_eval_batch_size=cfg.reward_model.batch_size,
        max_length=cfg.max_length,
        fp16=cfg.reward_model.fp16,
        num_train_epochs=cfg.reward_model.num_train_epochs,
        optim=cfg.reward_model.optimizer,
        report_to=cfg.reward_model.report_to,
    )
    
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    print("Training reward model...")
    trainer.train()

    trainer.save_model(cfg.reward_model.output_dir)
    tokenizer.save_pretrained(cfg.reward_model.output_dir)
    print(f"Reward model saved to {cfg.reward_model.output_dir}")
    wandb.finish()

    return cfg.reward_model.output_dir

if __name__ == "__main__":
    main() 