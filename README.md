# back2basics_reinforce
Implementation of https://arxiv.org/pdf/2402.14740

## Tech instructions

### Level 1 (Regression Reward Model)

```bash
# Full pipeline
python run.py --level 1 --config config.yaml

# Train only reward model
python run.py --level 1 --config config.yaml --only_reward_model

# Train only REINFORCE
python run.py --level 1 --config config.yaml --skip_reward_training
```

### Level 2 (10 classes Reward Model)

```bash
# Full pipeline
python run.py --level 2 --config config.yaml

# Train only reward model
python run.py --level 2 --config config.yaml --only_reward_model

# Train only REINFORCE
python run.py --level 2 --config config.yaml --skip_reward_training
```

### Evaluate and compare sft and reinforce models

```bash
python evaluate.py --config config.yaml --checkpoint_path reinforce_model/checkpoint-175
```

## Results

### Reinforce baseline
 - steps: 75
 - batch_size: 32

![REINFORCE baseline wandb log ](img/level1_75_steps.png)

| Model | Eval Reward | Improvement |
|-------|-------------|-------------|
| SFT | 1.2507 | - |
| Reinforce | 1.9433 | +0.6933 |


<details>
<summary><strong>Some samples from eval dataset</strong></summary>

**Prompt: What are some current alternatives to the conventional gasoline powered vehicle? Create a list of five popular alternatives to the conventional gasoline-powered vehicle. For each item on the list explain how they align with the goal of reducing gasoline dependency. For each item list potential challenges in adopting each vehicle as an alternative to gasoline-powered vehicles. For each vehicle explain the improvements that have been made to optimize the use of these vehicles since their inception. For each explain how consumers have been responding to these alternative vehicle options.**

**SFT model:**
Based on the goal of reducing gasoline dependency, here are five alternative vehicles that could be considered:
1. Electric Car: An alternative to gasoline-powered vehicles, electric cars convert electrical energy into mechanical energy, reducing the dependence on fossil fuels. They are powered by stationary fans and wind turbines, producing zero tailpipe emissions, and generally lower operating costs compared to gasoline-powered vehicles. Electric vehicles also have lower operating costs and can be more convenient to maintain.
2. Hybrid Car: A hybrid vehicle combines a gasoline engine with an electric motor to produce energy. The vehicle combines the advantages of both and can be driven for long distances at constant speeds. Hybrid cars have lower operating costs compared to gasoline-powered vehicles, and can be more fuel-efficient. They require fewer moving parts and are more environmentally friendly, with potential improvement in noise.
3. Alternative Fuel Vehicles: Vehicles that use alternative fuels (e.g., hydrogen fuel cell vehicles, biofuels) instead of regular gasoline. These vehicles can be used for short trips or on long trips. They offer lower operational costs compared to gasoline-powered vehicles, and can be more convenient for residents in rural areas. Alternative fuel vehicles also have lower operating costs and can be more fuel-efficient compared to gasoline-powered vehicles.
4. High Performance Electric Vehicles (HEVs): Electric vehicles using electric motors to power their internal combustion engines. HEVs do not produce tailpipe emissions, are more efficient than conventional electric vehicles, and can be driven for extended distances at lower speeds. They offer a range of advantages including reduced environmental impact, lower operating costs, and potential reduction in energy consumption and greenhouse gas emissions.
5. Electric Vehicle (EV) Packages: Packaged electric vehicles contain both the car and the battery pack in a single vehicle. For example, a hybrid vehicle and a full-size electric vehicle. The car battery is used for

**Trained model**:
Here are the five suggestions for the conventional gasoline-powered vehicle:
1. **Electric Powered Vehicle (EPV)**:
        * The EPV is a zero-emission vehicle that uses electricity to run.
        * It has a range of up to 500 miles and is known for its long-range and environmentally friendly features.
        * EPVs are cost-effective, have a lower greenhouse gas emissions, and are currently a popular choice for road users.
        * They are a viable alternative to gasoline-powered vehicles, especially for urban driving.
2. **Hybrid Vehicle (HCV)**:
        * The CVR combines the advantages of a gasoline-powered vehicle with an electric motor.
        * It has a range of up to 500 miles and can be equipped with a range of four electric motors.
        * The CVR is considered to be an environmentally friendly solution, as it eliminates the need for oil changes.
        * It is a viable alternative to gasoline-powered vehicles, especially for urban driving.
3. **Fuel-Smart Car (FSC)**:
        * The FSC is a car that has been designed to be fuel-saver.
        * It has a range of up to 500 miles and can be equipped with a range of four electric motors.
        * The FSC is considered to be a more environmental friendly solution, as it uses fuel savings to reduce the number of oil changes.
        * It is a viable alternative to gasoline-powered vehicles, especially for urban driving.
4. **Vcfi Fuel Cell Car (VFC)**:
        * The VFC is a car that has been designed to be fuel-saver.
        * It has a range of up to 500 miles
</details>

### Reinforce baseline with repetition_penalty
During training, the reinforcement learning model began heavily favoring constant predictions. Starting around step 75, it began generating repetitive phrases or symbols (likely a reward-hacking behavior). By step 175, the model exclusively outputs constants while receiving extremely high rewards. This occurs because our setup lacks a connection to the original SFT model’s distribution, unlike methods like PPO that enforce this via KL divergence. To mitigate this, I attempted using `repetition_penalty=1.2` in `model.generate()` to mildly constrain deviations from the baseline policy proposed in the paper. In RLOOTrainer KL divergence is included in the loss function.

Also, I changed batch_size and anount of steps:
 - steps: 50
 - batch_size: 64

But I didn't have time to calculate the average difference of the reverts and look at the generation itself.