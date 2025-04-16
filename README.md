# back2basics_reinforce
Implementation of https://arxiv.org/pdf/2402.14740


Full pipeline

```bash
python run.py
```

Train only reward model

```bash
python run.py --only_reward_model
```

Train only reinforce

```bash
python run.py --skip_reward_training
```