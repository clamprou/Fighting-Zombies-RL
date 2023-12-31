from __future__ import print_function
from __future__ import division
from malmo_agent import *
from ai_play import *
from gym_env import FightingZombiesDisc
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Hyper Parameters:\nBATCH_SIZE: " + str(BATCH_SIZE) +"\nGAMMA: "+ str(GAMMA) +"\nEPS_START: "+ str(EPS_START)
      +"\nEPS_END: "+ str(EPS_END) +"\nEPS_DECAY: "+ str(EPS_DECAY) +"\nTAU: "+ str(TAU) +"\nLR: "+ str(LR))

NUM_EPISODES = 101
env = FightingZombiesDisc()
wins = 0

for episode in range(NUM_EPISODES):
    state, done = env.reset(), False
    print("Running mission #" + str(episode))
    t = 0
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while not done:
        action = select_action(state)
        observation, _, done, won = env.step(action.item())
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        state = next_state
        t += 1
    if won:
        wins += 1

print('Complete')
print("Wins: ", wins,"%", "| Zombies Killed: ", (sum(env.agent.kills)/300) * 100, "%") # 300 are all the zombies spawned: 3 zombies per episode x 100 episodes
plot_table(env.agent.rewards, "rewards", show_result=True)
plot_table(env.agent.kills, "kills", show_result=True)
plot_table(env.agent.player_life, "life", show_result=True)
# plot_table(env.agent.survival_time, "survival", show_result=True)
plt.ioff()
plt.show()

time.sleep(1)
