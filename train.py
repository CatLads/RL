import glob
import sys
import os
import numpy as np
from temperature_observation import TemperatureObservation
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv
from dqn import DDDQN, Agent
from flatland.utils.rendertools import RenderTool
from temperature_observation.utils import normalize_observation, format_action_prob
import wandb


wandb.init(project='flatlands', entity='fatlads', tags=['dddqn_added_channels', "dddqn", "prio_exp_rpl", "temp"])
config = wandb.config
seed = 69  # nice
width = 15  # @param{type: "integer"}
height = 15  # @param{type: "integer"}
num_agents = 3  # @param{type: "integer"}
tree_depth = 2  # @param{type: "integer"}
radius_observation = 10
WINDOW_LENGTH = 22  # @param{type: "integer"}

random_rail_generator = complex_rail_generator(
    nr_start_goal=10,  # @param{type:"integer"} number of start and end goals
    # connections, the higher the easier it should be for
    # the trains
    nr_extra=10,  # @param{type:"integer"} extra connections
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

#env_renderer = RenderTool(env)
state_shape = normalize_observation(
    obs[0], tree_depth, radius_observation).shape
action_shape = (5,)
agent007 = Agent(state_shape, 5)
if (glob.glob("alternative_model.*") != []):
    agent007.load_model()
# Train for 300 episodes
saving_interval = 50
max_steps = env._max_episode_steps
smoothed_normalized_score = -1.0
smoothed_completion = 0.0
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
                agent_obs[agent] = normalize_observation(
                    obs[agent], tree_depth, radius_observation)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        for step in range(max_steps - 1):
            actions = {}
            agents_obs = {}

            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = agent007.act(agent_obs[agent])

                    action_count[action] += 1
                    # actions_taken.append(action)
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})

            next_obs, all_rewards, done, info = env.step(
                action_dict)  # base env
            # env_renderer.render_env(show=True)

            # Update replay buffer and train agent
            for agent in env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    agent007.update_mem(
                        agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])
                    agent007.train()
                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                if next_obs[agent] is not None:

                    agent_obs[agent] = normalize_observation(
                        next_obs[agent], tree_depth, radius_observation)

                scores += all_rewards[agent]

            if done['__all__']:
                break

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        if (step_counter < max_steps - 1):
            completion = tasks_finished / max(1, env.get_num_agents())
        normalized_score = scores / (max_steps * env.get_num_agents())
        smoothing = 0.99
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
            "completion": 100 * completion,
            "smoothed_completion": 100 * smoothed_completion
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
