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
During training, the policy collapsed into “constant” outputs—by step 75 it was producing repeated phrases or symbols (a classic reward‑hacking behavior), and by step 175 it output only constants while still earning very high rewards. This degeneration happens because vanilla REINFORCE has no mechanism to tie the policy back to the original SFT distribution (unlike PPO, which uses a KL penalty). To counteract this, we introduced a mild `repetition_penalty=1.2` in `model.generate()`. In RLOOTrainer, a KL divergence term is already built into the loss.

Also, I changed batch_size and anount of steps:
 - steps: 50
 - batch_size: 64

But I didn't have time to calculate the average difference of the reverts and look at the generation itself.



### Reinforce baseline with classification reward model 

Reward model loss:

$$\mathcal{L}(x,y_w,y_l)=-\log\!\Bigl(\sum_{i=1}^{10}\sum_{j=1}^{10} p^w_i\,p^l_j\,[\,i>j\,]\Bigr)$$


This loss directly maximizes the model’s probability of assigning a higher score to the chosen completion \(y_w\) than to the rejected one \(y_l\), yielding a smooth, differentiable objective for pairwise preference learning.

 - steps: 150
 - batch_size: 16

 ![REINFORCE baseline wandb log ](img/level2_150_steps.png)

 | Model | Eval Reward | Improvement |
|-------|-------------|-------------|
| SFT | 6.5909 | - |
| Reinforce | 6.0746 | -0.5163 |

<details>
<summary><strong>Some samples from eval dataset</strong></summary>

**Prompt: please compare deadlock avoidance and deadlock prevention mechanisms**

**Original model:**

Deadlock prevention mechanisms include techniques that prevent the cycle from continuing indefinitely, such as:

1. **Managed wait**: Preventing a program from executing indefinitely by releasing a task.
2. **Exclusive locking**: Restricting access to a shared resource.
3. **Lock synchronization**: Ensuring that a thread and other tasks access shared resources.
4. **Timeouts**: Restricting access to shared resources.

In contrast, deadlock avoidance mechanisms involve techniques that intentionally cause the cycle to continue, such as:

1. **Dual locks**: Allowing two threads to execute a task simultaneously.
2. **Pending state machines**: Creating a state machine that contains two threads.
3. **Lock synchronization with a separate system call**: Prioritizing access to a shared resource.
4. **Lock management techniques**: Managed wait, exclusive lock, and lock synchronization with a system call.

When choosing a deadlock prevention mechanism, consider the following factors:

1. **Resource availability**: Consider the amount of shared resources and how they may expire.
2. **Thread characteristics**: Think about the design of the threads and the need for lock synchronization.
3. **Thread behavior**: Consider the behavior of the threads when they're trying to acquire or release locks.
4. **Error tolerance**: Consider the trade-off between deadlock prevention and overall system stability.

By understanding the techniques used in deadlock prevention and avoiding them, developers can write more reliable and fault-tolerant code.

**Reward distribution:**
[0.0948, 0.0954, 0.1315, 0.1729, 0.1343, 0.1149, 0.0776, 0.0481, 0.1110, 0.0195]


**Trained model:**

, such as:

"I can help you with your creative writing or art project, and I can help you with your family's problems or challenges."

"I can help you with your education and skills, and I can help you with your job and social network."

"I can help you with your health and well-being, and I can help you with your relationships and connections."

"I can help you with your passions and interests, and I can help you with your creativity and imagination."

"I can help you with your future, and I can help you with your limits."

"I can help you with your future, and I can help you with your love of life."

"I can help you with your love of the earth, and I can help you with your spirituality."

"I can help you with your relationships with your pets, and I can help you with your family of origin."

"I can help you with your relationship with your friends and family, and I can help you with your friends and colleagues."

"I can help you with your relationships with your artists, and I can help you with your fans."

"I can help you with your future, and I can help you with your dreams and interests."

"I can help you with your love of the sea, and I can help you with your mountains, and I can help you with your forest and tree"

"I can help you with your passions and interests, and I can help you with your determination, and I can help you with your gratitude."

"I can help you with your love of the earth, and I can help you with your mountains, and I can help you with your flowers, and your grass"

"I can help you with your love of the sea, and I can help you with your forest, and I can help you with your trees, and your sun, and the moon"

"I can help you with your love of the earth, and I can help you with your mountains, and you can be a woman, and a man, and a boy, and the sea and the mountains and the trees of our world, and the flowers and the grass"

"I can help you with your love of the city, and the nation, and the world, and the stars"

"I can help you with

**Reward distribution:**
[0.0529, 0.0491, 0.0692, 0.0707, 0.1539, 0.2066, 0.0287, 0.0269, 0.0354, 0.3066]
</details>


Judging by the train loss and the drop in average revards, the model has learned poorly. The loss is falling, but the average revards is also falling. To look closier to the generations, the higher revard is given to the repeated sentences again (Even with `repetition_penalty=1.2` applied). I think a good improvement would be to add a KL divergence to the loss with original SFT distribution to miligate this.