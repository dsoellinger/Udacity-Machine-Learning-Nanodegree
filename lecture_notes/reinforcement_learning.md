# Reinforcement Learning

## How does the reinforcement setting look like?

Reinforcement learning is all about learning from try and error. The learning framework can be described by an **agent** learning to interact with an **environment**. We assume that time involves in discrete time steps.
At the initial time step, the agent observes the environment. Then, it must select an appropriate action. In the next step (in response to the agent's action) the environment presents a new situation to the agent. It also provides the agent with some kind of reward. In response, the agent must choose an action.

The goal is to take actions that **maximize** the **expected cumulative reward**.

<img src="images/rl_setting.png" width="350px" />

## Episodic and Continuing Tasks

If a problem has a well-defined ending point we call it an **episodic tasks**. For example, a chess game has a finite number of moves. The sequence of interactions is typically called an **episode**. It's always able to start from scratch like it's being reborn in the environment, but with the added knowledge from the past life.

**Continuing tasks** are ones that never end. For example, an algorithm that buys stocks and response to the financial market. In this case, the agent lives forever and it has to choose actions while simultaneously interacting with the environment.

## Rewards hypothesis

The term "reinforcement" originally comes from behavioral science. It refers to a stimulus that is delivered immediately after a behavior to make the behavior more likely to occur in the future. 

**Reward hypothesis:**  
All goals can be framed as the maximization of **expected** cumulative reward.

Let's consider an example to understand what this means in practice. Let's say we want to teach a robot to walk.

**What are the actions?**  
They are forces that the robot applies to its joints ("Gelenke").

**What are the states?**  
- The current position and volocities of all joints  
- Measurements of the ground  
- Contact sensor data (Used to determine whether the robot is still walking or falling over)

**What are the rewards?**  
We design the reward as a feedback algorithm as a feedback mechanism that tells the agent the appropriate movement.

$r = min(v_x,v_{max}) - 0.005(v_y^2 + v_z^2) - 0.05 y^2 - 0.02 ||u||^2 + 0.02$

Let's take a closer look at the reward function to understand what the individual components are.

$min(v_x,v_{max})$ ... Reward proportional to the robot's forward velocity. If it moves faster, it gets more reward (up to a certain limit denoted by $v_{max}$)

$0.02 ||u||^2$ ... Penalized by the amount of torque applied to each joint

$0.005(v_y^2 + v_z^2)$ ... Since the agent is designed to move forward, we want to penalize vertical movements.

$0.05 y^2$ ... Tracks whether the body move away from the center of its track. We want to keep the humanoid as close to the center as possible.

$0.02$ ... At every time step the agent receives some positive award if the humanoid has not yet fallen. If the robot falls, the episode terminates meaning that the humanoid missed a opportunity to collect more award.

## Cumulative reward

The question we are going to answer in this section is whether it's enough to maximize the reward at each time step or if it's always necessary to look at the commutative sum.

Let's try to understand this using the walking robot example. If the robot only looked at the reward at a single time step, he would simply try to move as fast as possible without falling immediately. That could work well in the short term. However, it's possible that that the agent learns a movement that makes him move quickly, but forces him to fall in a short time. Hence, the individual award might be high, but the comulative award is still small meaning that the agent can't walk.  
Therefore, we always need to look at short term and long term consequences.

## Discounted reward

If we look at a time step $t$ we will notice that all the rewards in the past have already been decided. The sum of rewards is called **return**. Only future rewards are in the agents control.

At time step $t$ the agent picks $A_t$ to maximize the return (expected) $G_t$.

$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...$

In case of **discounted rewards** we want time steps that occurred earlier in time to have a much greater weight.

${G_t}_{discount} = R_{t+1} + 0.9 \cdot R_{t+2} + 0.81 \cdot R_{t+3} + ...$

We often write it like... 

${G_t}_{discount} = R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \gamma^3 R_{t+4}...$ where $\gamma \in [0,1]$

**Note:** The larger $\gamma$ gets, the more we care about the future (try it by plugging in values for $\gamma$)
