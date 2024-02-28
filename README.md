# Reinforcement Learning DQN - Fight Zombies in Minecraft

This project is written in Python using the [Malmo](https://github.com/microsoft/malmo/tree/master) platform to create 
a reinforcement learning scenario of fighting zombies in Minecraft. Additionally, this project employs the PyTorch library 
for implementing the DQN algorithm. The primary goal is to determine whether the state-of-the-art DQN algorithm can learn 
and devise strategies to defeat zombies in the Minecraft game.

## Training Scenario
The scenario in which the agent is trained involves combating zombies in Minecraft, designed using the 
[Malmo](https://github.com/microsoft/malmo/tree/master) platform. Spawned in a 20x20 enclosed arena with three zombies 
inside, the agent is equipped with iron armor and a diamond sword. Its objective is to learn appropriate actions 
(such as moving left, moving right, attacking, etc.) to survive against the zombies.

![DQN-GIF.gif](MyResults%2FDQN-GIF.gif)

## Rewards
* Damage zombie: **+ 30**
* Kill zombie: **+ 100**
* Lose health: **- 5**
* Die: **- 100**
* Every step: **- 0.1**

## DQN Algorithm
The implementation of the DQN model utilized the [PyTorch](https://pytorch.org/) library, which provides several abstractions and facilitates easy design. The fundamental concept behind the DQN algorithm involves managing a neural network to approximate the Q function, estimating the expected cumulative rewards for specific actions in a given state. For the design of this algorithm, an implementation example of the DQN algorithm by PyTorch in the gym Cart Pole environment was referenced. More information can be found at [this link](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). Below is a representation of the designed DQN architecture model and the fine-tuned hyperparameters.

![DQN.png](MyResults%2FDQN.png "DQN")

## Installation
* Install the Malmo library. The easiest method is to [install Marlo using Anaconda](https://marlo.readthedocs.io/en/latest/installation.html) (which includes Malmo).
* Ensure you have Python 3.7.12 installed.
* Install PyTorch version 1.13.0.

## Execution
To commence training the DQN model in the provided scenario, simply execute `train.py`. After training, evaluate the learned policy of the agent by executing `play.py`.

## Results
Following training, the agent demonstrates acceptable performance against the zombies, achieving:
* **92%** win rate
* **10.89** average life (with 20 being the maximum)

![Results.png](MyResults%2FResults.png "Results")
