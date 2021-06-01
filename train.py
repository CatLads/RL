import glob
import sys
import os
import numpy as np
from temperature_observation import TemperatureObservation
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv, RailEnvActions
from dqn.agent import Agent
from flatland.envs.agent_utils import RailAgentStatus
from flatland.utils.rendertools import RenderTool
from temperature_observation.utils import normalize_tree_observation, normalize_temperature_observation
from temperature_observation.utils import format_action_prob
import wandb

seed = 69  # nice
width = 15
height = 15
num_agents = 3
tree_depth = 2
radius_observation = 10
wandb.init(project='flatlands', entity='fatlads')
config = wandb.config

random_rail_generator = complex_rail_generator(
    nr_start_goal=10, # number of start and end goals
    # connections, the higher the easier it should be for the trains
    nr_extra=10,  # extra connections
    # (useful for alternite paths), the higher the easier
    min_dist=10,
    max_dist=99999,
    seed=seed
)

env = RailEnv(
    width=width,
    height=height,
    rail_generator=random_rail_generator,
    obs_builder_object=TemperatureObservation(tree_depth),
    number_of_agents=num_agents
)

obs, info = env.reset()

normalized_temp = normalize_temperature_observation(obs[0][0]).flatten()
normalized_tree = normalize_tree_observation(obs[0][1], tree_depth, radius_observation)
state_shape = np.concatenate((normalized_temp, normalized_tree)).shape
action_shape = (5,)
method = "cdddqn"
# specify the algorithm to use and every parameter
agent007 = Agent(state_shape, 
                 action_shape[0], 
                 (width, height), 
                 gamma=0.99, 
                 replace=100, 
                 lr=0.001, 
                 epsilon_decay=1e-3, 
                 decay_type="flat", 
                 initial_epsilon=1.0, 
                 min_epsilon=0.01, 
                 batch_size=64, 
                 method=method)

# FIXME: Does this actually work?
if glob.glob(f"{method}*") != []:
    agent007.load_model()

saving_interval = 50
max_steps = env._max_episode_steps
smoothed_normalized_score = -1.0
smoothed_completion = 0.0
smoothing = 0.99

action_count = [0] * action_shape[0]
action_dict = dict()
agent_obs = [None] * num_agents
agent_prev_obs = [None] * num_agents
agent_prev_action = [2] * num_agents
update_values = [False] * num_agents

for episode in range(3000):
    try:
        # Initialize episode
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        #env_renderer = RenderTool(env)
        done = {i: False for i in range(0, num_agents)}
        done["__all__"] = False
        scores = 0
        step_counter = 0

        for agent in env.get_agent_handles():
            if obs[agent] is not None:
                norm_temp = normalize_temperature_observation(obs[agent][0]).flatten()
                norm_tree = normalize_tree_observation(obs[agent][1], tree_depth, radius_observation)
                agent_obs[agent] = np.concatenate((norm_temp, norm_tree))
                agent_prev_obs[agent] = agent_obs[agent].copy()
        for step in range(max_steps - 1):
            actions = {}
            agents_obs = {}

            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    legal_moves = np.array([1 for i in range(0, 5)])
                    for action in RailEnvActions:
                        if info["status"][agent] == RailAgentStatus.ACTIVE:
                            legal_moves[int(action)] = int(env._check_action_on_agent(action, env.agents[agent])[-1])
                    action = agent007.act(agent_obs[agent], legal_moves)

                    action_count[action] += 1
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})
            next_obs, all_rewards, done, info = env.step(action_dict)


            # env_renderer.render_env(show=True)

            # Update replay buffer and train agent
            for agent in env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    agent007.update_mem(agent_prev_obs[agent], 
                                        agent_prev_action[agent], 
                                        all_rewards[agent], 
                                        agent_obs[agent], 
                                        done[agent])
                    agent007.train()
                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                if next_obs[agent] is not None:
                    norm_temp = normalize_temperature_observation(obs[agent][0]).flatten()
                    norm_tree = normalize_tree_observation(obs[agent][1], tree_depth, radius_observation)
                    agent_obs[agent] = np.concatenate((norm_temp, norm_tree))

                scores += all_rewards[agent]

            if done['__all__']:
                break

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        if (step_counter < max_steps - 1):
            completion = tasks_finished / max(1, env.get_num_agents())
        normalized_score = scores / (max_steps * env.get_num_agents())
        smoothed_normalized_score = smoothed_normalized_score * \
            smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * \
            smoothing + completion * (1.0 - smoothing)
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_shape[0]
        step_counter += 1
        wandb.log({
            "normalized_score": normalized_score,
            "smoothed_normalized_score": smoothed_normalized_score,
            "completion": 100*completion,
            "smoothed_completion": 100*smoothed_completion
        })
        print(
            '\r🚂 Episode {}'
            '\t 🏆 Score: {:.3f}'
            ' Avg: {:.3f}'
            '\t 💯 Done: {:.2f}%'
            ' Avg: {:.2f}%'
            '\t 🔀 Action Probs: {}'
            '\n'.format(
                episode,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                format_action_prob(action_probs)
            ), end=" ")

        if (episode % saving_interval == 0):
            agent007.save_model()
        # sum_rewards += reward
    except KeyboardInterrupt:
        print('Interrupted')
        agent007.save_model()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
