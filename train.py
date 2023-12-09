from __future__ import print_function
from __future__ import division
from malmo_agent_circle import *
from ai import *
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Hyper Parameters:\nBATCH_SIZE: " + str(BATCH_SIZE) +"\nGAMMA: "+ str(GAMMA) +"\nEPS_START: "+ str(EPS_START)
      +"\nEPS_END: "+ str(EPS_END) +"\nEPS_DECAY: "+ str(EPS_DECAY) +"\nTAU: "+ str(TAU) +"\nLR: "+ str(LR))

NUM_EPISODES = 2000

for episode in range(NUM_EPISODES):
    state, done = env.reset(), False
    t = 0
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while not done:
        action = select_action(state)
        observation, reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        t += 1
    if(episode > 100):
        values = torch.tensor(env.agent.rewards, dtype=torch.float)
        means = values.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        print("| Episode: ", episode," | Average reward: ",means.numpy()[-1], "| Episode reward: ",env.agent.episode_reward, "|")
    else:
        print("| Episode: ", episode, "| Episode reward: ",env.agent.episode_reward, "|")
    if (episode + 1) % 100 == 0:
        plot_table(env.agent.rewards, "rewards")


print('Complete')
plot_table(env.agent.rewards, "rewards", show_result=True)
plot_table(env.agent.kills, "kills", show_result=True)
plot_table(env.agent.player_life, "life", show_result=True)
# plot_table(env.agent.survival_time, "survival", show_result=True)
plt.ioff()
plt.show()

time.sleep(1)
