# Reinforcement Learning DQN - Fight Zombies in Minecraft

This project is writen in Python using the [Malmo](https://github.com/microsoft/malmo/tree/master) platform, to create the 
reinforcement learning scenario of fighting zombies in Minecraft. Also, this project uses the Pytorch library for the
implementation of the DQN algorithm. The main idea, is to find out if the state of the art DQN algorithm is
able to learn and find strategies to win against the zombies in the Minecraft game.

## Training Scenario
Τhe scenario in which the agent is trained is fighting against zombies in Minecraft, which was designed using the 
[Malmo](https://github.com/microsoft/malmo/tree/master) platform. Spawned in a 20x20 close arena with 
3 zombies inside it and equipped with an iron armor and a diamond sword, agent should learn what actions to perform 
(move left, move right, attack, ...) in order to survive against the zombies.

![Scenario.png](MyResults%2FScenario.png "Scenario")

## Rewards
* Damage zombie: **+ 30**
* Kill zombie: **+ 100**
* Lose health: **- 5**
* Die: **- 100**
* Every step: **- 0.1**

## DQN Algorithm
The implementation of the DQN model utilized the [Pytorch](https://pytorch.org/) library (using the Python language), 
which provides several abstractions and easy design. The basic idea behind the DQN algorithm is managing a neural network 
to approximate the Q function, which estimates the expected cumulative rewards for taking specific actions in a given state. 
For the design of this algorithm, an implementation [example of the DQN algorithm by Pytorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) in the gym Cart Pole environment was leveraged. You can find more information at this link.

![DQN.png](MyResults%2FDQN.png "DQN")

## Installation
* Malmo library: Easiest way is to [install Marlo using Anaconda](https://marlo.readthedocs.io/en/latest/installation.html) (it will install Malmo too)
* Python 3.7.12
* Pytorch 1.13.0

## Execution
In order to start the training of the DQN model in the above scenario you only have to execute the `train.py`. After, 
the training you can execute the `play.py` to evaluate the so far learned policy of the agent.

## Results
After training, the agent is able to establish good performance against the zombies with:
* **92%** win rate
* **10.89** average life (20 is the max)

![Results.png](MyResults%2FResults.png "Results")

