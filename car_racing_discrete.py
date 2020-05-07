import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
from skimage import color


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LazyFrames(object):
    """
    https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class Memory:
    """
    This class is used for Experience replay.
    """
    def __init__(self, memory_size, seed=0):
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=memory_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(tuple([state, action, reward, next_state, done]))

    def sample(self, batch_size):
        """
        This method returns a list of tensors after sampling a batch from the Experience Replay memory.
        It loads the tensors on to the computing device: cpu/gpu before returning them.
        """
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack(
            [np.expand_dims(np.moveaxis(np.asarray(e[0]), [0, 1, 2], [2, 1, 0]), axis=0) for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack(
            [e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack(
            [e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [np.expand_dims(np.moveaxis(np.asarray(e[3]), [0, 1, 2], [2, 1, 0]), axis=0) for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# Discrete-to-continuous action mapping
actions = {
    0: tuple([0, 0, 0]),  # DO NOTHING
    1: tuple([0, 0.5, 0]),  # ACCELERATE
    2: tuple([0, 0, 0.5]),  # BRAKE
    3: tuple([0.5, 0, 0]),  # RIGHT
    4: tuple([-0.5, 0, 0]),  # LEFT
}


class QCNN(nn.Module):
    """
    The Deep Q-Learning Network with CNN.
    """
    def __init__(self, seed=0):
        super(QCNN, self).__init__()
        self.seed = random.seed(seed)

        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=8, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected
        self.fc1 = nn.Linear(21*21*16, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, frame):
        x = F.relu(self.conv1(frame))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Agent:
    """
    Agent for solving the CarRacing-v0 environment
    """

    def __init__(self, name, env, target_score, use_saved=False, seed=0):
        self.name = name
        self.seed = random.seed(seed)
        self.time_step = 0
        self.data_location = "data/{}/".format(self.name)

        # Environment features
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = 5
        self.target_score = target_score

        # RL parameters
        self.discount_factor = 0.95
        self.epsilon_range = [1.0, 0.01]
        self.epsilon_decay = 0.995
        self.update_every_n_steps = 10
        self.epsilon = self.epsilon_range[0]

        # Memory buffer
        self.memory_size = int(1e5)
        self.batch_size = 64
        self.memory = Memory(self.memory_size, seed=seed)

        # Q Network parameters
        self.learning_rate = 1e-3
        self.soft_update_tau = 1e-2

        # Q-Network
        self.primary_network = QCNN(seed=seed).to(device)
        self.target_network = QCNN(seed=seed).to(device)
        self.optimizer = optim.Adam(self.primary_network.parameters(), lr=self.learning_rate)

        # Training data
        self.scores = list()
        self.scores_window = deque(maxlen=100)
        self.training_start_time = None
        self.trained_duration_from_load = 0
        self.episodes_trained = 0

        # Stacked frames
        self.stack_size = 4
        self.stacked_states = deque([], maxlen=self.stack_size)

        # Load trained agent to continue training or to play the environment
        if use_saved:
            self._load_saved_agent()

    def _load_saved_agent(self):
        self._load_state()
        self._load_memory()
        self._load_network()

    def act(self, state, epsilon=0.0):
        """
        Computes the best action for the given state using the Q-Network,
        and returns an action using epsilon-greedy policy.
        """
        state = torch.from_numpy(np.asarray(state).copy()).float().unsqueeze(0).permute([0, 3, 2, 1]).to(device)

        # Eval mode and without auto-grad to save time
        self.primary_network.eval()
        with torch.no_grad():
            action_values = self.primary_network(state)
        self.primary_network.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    @staticmethod
    def get_action(action):
        return actions[action]

    def _step(self, state, action, reward, next_state, done):
        """
        The agent steps forward in time.
        This function mirrors the step function of the gym environment.

        It adds the current experience into the replay buffer,
          and trains the agent using a random sample from the memory.
        """
        self.time_step += 1
        self.memory.add(state, action, reward, next_state, done)

        if self.time_step % self.update_every_n_steps == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self._train_q_network(experiences)
                self.epsilon = max(self.epsilon_range[1], self.epsilon_decay * self.epsilon)

    def _train_q_network(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_argmax_for_next_state = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards + (self.discount_factor * q_argmax_for_next_state * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.primary_network(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update()

    def _soft_update(self):
        """
        The soft update for Double Deep Q-Learning that updates the target network using the update parameter TAU.
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.primary_network.parameters()):
            target_param.data.copy_(
                self.soft_update_tau * local_param.data + (1.0 - self.soft_update_tau) * target_param.data)

    def learn_env(self, max_episodes, max_timesteps_per_episode=1500):
        """
        This function starts the agent's training on the environment.

        It displays progress and intermittently saves the training state so that it can be recovered if
           the program shut down unexpectedly.

        :param max_episodes: Maximum episodes to train for.
        :param max_timesteps_per_episode: Maximum timesteps to train on one episode.
        :return:
        """
        self.training_start_time = time.time()
        training_start_index = self.episodes_trained + 1

        for i_episode in range(training_start_index, max_episodes+1):
            state = self.env.reset()
            state = self.preprocess(state, first=True)
            score = 0
            for t in range(max_timesteps_per_episode):
                action = self.act(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(self.get_action(action))
                next_state = self.preprocess(next_state)
                self._step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            self.scores_window.append(score)  # save most recent score
            self.scores.append(score)  # save most recent score

            self._display_progress(i_episode)

            # Save self intermittently
            if i_episode % 20 == 0:
                self.episodes_trained = i_episode
                self._save_network()
                self._save_memory()
                self._save_state()

            # Success
            if np.mean(self.scores_window) >= self.target_score:
                h, m, s = self._get_hms()
                print('\n{:02d}:{:02d}:{:02d} - Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    h, m, s, i_episode - 100, np.mean(self.scores_window)))

                self.episodes_trained = i_episode
                self._save_network()
                self._save_memory()
                self._save_state()
                break
        return self.scores

    def _display_progress(self, i_episode):
        h, m, s = self._get_hms()
        print('\r{:02d}:{:02d}:{:02d} - Episode {}\tAverage Score: {:.2f}'.format(
            h, m, s, i_episode, np.mean(self.scores_window)), end="")

        if i_episode % 20 == 0:
            print('\r{:02d}:{:02d}:{:02d} - Episode {}\tAverage Score: {:.2f}'.format(
                h, m, s, i_episode, np.mean(self.scores_window)))

    def _get_duration(self):
        return time.time() - self.training_start_time + self.trained_duration_from_load

    def _get_hms(self):
        seconds_passed = self._get_duration()
        h = int(seconds_passed / 3600)
        m = int(seconds_passed / 60) % 60
        s = int(seconds_passed) % 60
        return h, m, s

    def _load_network(self):
        self.primary_network.load_state_dict(torch.load(self.data_location + "primary_network.m"))
        self.target_network.load_state_dict(torch.load(self.data_location + "target_network.m"))

    def _save_network(self):
        import os
        if not os.path.exists(self.data_location):
            os.makedirs(self.data_location)
        torch.save(self.primary_network.state_dict(), self.data_location + "primary_network.m")
        torch.save(self.target_network.state_dict(), self.data_location + "target_network.m")

    # Not implemented because the memory buffer quickly becomes too large ( > 8GB)
    def _load_memory(self):
        pass

    # Not implemented because the memory buffer quickly becomes too large ( > 8GB)
    def _save_memory(self):
        pass

    def _load_state(self):
        with open(self.data_location+"state.pickle", 'rb') as handle:
            agent_state = pickle.load(handle)

        self.name = agent_state["name"]
        self.time_step = agent_state["time_step"]
        self.state_size = agent_state["state_size"]
        self.action_size = agent_state["action_size"]
        self.target_score = agent_state["target_score"]

        self.discount_factor = agent_state["discount_factor"]
        self.epsilon_range = agent_state["epsilon_range"]
        self.epsilon_decay = agent_state["epsilon_decay"]
        self.update_every_n_steps = agent_state["update_every_n_steps"]
        self.epsilon = agent_state["epsilon"]

        self.memory_size = agent_state["memory_size"]
        self.batch_size = agent_state["batch_size"]

        self.learning_rate = agent_state["learning_rate"]
        self.soft_update_tau = agent_state["soft_update_tau"]

        self.scores = agent_state["scores"]
        self.scores_window = agent_state["scores_window"]
        self.trained_duration_from_load = agent_state["trained_duration_from_load"]
        self.episodes_trained = agent_state["episodes_trained"]

    def _save_state(self):
        agent_state = dict(
            name=self.name,
            time_step=self.time_step,
            state_size=self.state_size,
            action_size=self.action_size,
            target_score=self.target_score,

            discount_factor=self.discount_factor,
            epsilon_range=self.epsilon_range,
            epsilon_decay=self.epsilon_decay,
            update_every_n_steps=self.update_every_n_steps,
            epsilon=self.epsilon,

            memory_size=self.memory_size,
            batch_size=self.batch_size,

            learning_rate=self.learning_rate,
            soft_update_tau=self.soft_update_tau,

            scores=self.scores,
            scores_window=self.scores_window,
            trained_duration_from_load=self._get_duration(),
            episodes_trained=self.episodes_trained
        )
        with open(self.data_location+"state.pickle", 'wb') as handle:
            pickle.dump(agent_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def display_scores(self):
        fig = plt.figure()
        fig.add_subplot(111)

        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    def play(self, n_episodes=5):
        """
        Call to see how the agent performs with its current state of training.
        """
        for i in range(n_episodes):
            state = self.env.reset()
            state = self.preprocess(state, first=True)
            done = False
            while not done:
                action = self.act(state)
                self.env.render()
                state, reward, done, _ = self.env.step(self.get_action(action))
                state = self.preprocess(state)

    def preprocess(self, state, first=False):
        """
        Given a 96x96x3 RGB state.
        First converts it to 96x96x1 grayscale.
        Then stacks k last frames.
        Returns lazy array, which is much more memory efficient.
        Lazy array has dimensions 96x96xk
        This object should only be converted to numpy array before being passed to the model.
        It should not be stored as is in the experience buffer.
        """
        state = 2 * color.rgb2gray(state) - 1.0
        state = np.expand_dims(state, axis=2)
        if first:
            for _ in range(self.stack_size):
                self.stacked_states.append(state)
        else:
            self.stacked_states.append(state)
        return LazyFrames(list(self.stacked_states))


if __name__ == '__main__':
    env = gym.make("CarRacing-v0", verbose=0)
    env.seed(0)

    agent = Agent("CarRacing-FrameStack", env, target_score=900.0)

    agent.learn_env(500)
    agent.display_scores()
