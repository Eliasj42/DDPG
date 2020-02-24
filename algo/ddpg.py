import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy

from algo.ReplayBuffer import ReplayBuffer
from algo.ActorNetwork import ActorNetwork
from algo.CriticNetwork import CriticNetwork

BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
GAMMA = 0.98                    # Discount for rewards.
TAU = 0.05                      # Target network update rate.
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.0001

class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self, mu, sigma, epsilon):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            return action + np.random.normal(self.mu, self.sigma)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        np.random.seed(1337)
        self.env = env

        self.sess = tf.Session()
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.actor = ActorNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_ACTOR)
        self.critic = CriticNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_CRITIC)
        tf.keras.backend.set_session(self.sess)

        f_mean = open('mean.txt', 'w+')
        f_mean.write('')
        f_mean.close()
        f_sd = open('sd.txt', 'w+')
        f_sd.write('')
        f_sd.close()

    def evaluate(self, num_episodes):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                a_t = self.actor.model.predict(s_t[None])[0]
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
            if i < 9:
                plt.subplot(3, 3, i+1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                 color='g' if success else 'r')
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                if i == 8:
                    # Comment out the line below to disable plotting.
                    plt.show()
        return np.mean(success_vec), np.mean(test_rewards)

    def get_progress(self):
        rewards = []
        for i in range(10):
            total_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.actor.model.predict(np.reshape(state, [1, state.shape[0]]))[0]
                new_s, reward, done, info = self.env.step(action)
                total_reward += reward
                state = new_s
            rewards.append(total_reward)
        rewards = np.array(rewards)
        mean_f = open("mean.txt", "a+")
        mean_f.write(str(np.mean(rewards)) + ",")
        mean_f.close()
        sd_f = open('sd.txt', 'a+')
        sd_f.write(str(np.std(rewards)) + ',')
        sd_f.close()
        return np.mean(rewards), np.std(rewards)






    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """

        epsilon = 1
        epsilon_decay = .99
        epsilon_min = 0.25

        for i in range(num_episodes):
            if i % 100 == 0: self.get_progress()
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            first = True
            step = 0
            loss = 0
            display_grads = 0
            store_states = []
            store_actions = []
            while not done:
                step += 1
                action = self.actor.model.predict(np.reshape(s_t, [1, s_t.shape[0]]))[0]
                action = EpsilonNormalActionNoise(0,0.15, epsilon).__call__(action)
                first = False
                new_s, reward, done, info = self.env.step(action)
                #s_t2 = np.array(new_s)
                total_reward += reward
                trained = False
                self.buffer.add(s_t, action, reward, new_s, done)

                s_t = new_s
                #state = new_s
                if self.buffer.count() > 5000:
                    trained = True
                    mini_batch = self.buffer.get_batch(BATCH_SIZE)
                    states = np.asarray([_[0] for _ in mini_batch])
                    actions = np.asarray([_[1] for _ in mini_batch])
                    rewards = np.asarray([_[2] for _ in mini_batch])
                    y = np.asarray([_[2] for _ in mini_batch])
                    new_states = np.asarray([_[3] for _ in mini_batch])
                    dones = np.asarray([_[4] for _ in mini_batch])

                    target_actions=self.actor.target_model.predict(new_states)
                    target_q_values = self.critic.target_model.predict([new_states, target_actions])
                    for j in range(len(mini_batch)):
                        assert(y[j] == rewards[j])
                        if not dones[j]:
                            y[j] = rewards[j] + GAMMA * target_q_values[j]


                    loss += self.critic.model.train_on_batch([states, actions], y)
                    grads = self.critic.gradients(states, self.actor.model.predict(states))
                    display_grads += np.mean(abs(grads))
                    self.actor.train(states, grads)
                    self.actor.update_target()
                    self.critic.update_target()

            if trained:
                epsilon = max(epsilon * epsilon_decay, epsilon_min)

            if hindsight:
                # For HER, we also want to save the final next_state.
                store_states.append(new_s)
                store_actions.append(action)
                self.add_hindsight_replay_experience(store_states,
                                                     store_actions)
            del store_states, store_actions
            store_states, store_actions = [], []

            # Logging
            print("Episode %d: Total reward = %d" % (i, total_reward))
            #print("\tTD loss = %.2f" % (loss / step,))
            print("\tSteps = %d; Info = %s" % (step, info['done']))
            print("\tMean Gradient = {}".format(display_grads/step))
            if i % 100 == 0 and False:
                successes, mean_rewards = self.evaluate(10)
                print('Evaluation: success = %.2f; return = %.2f' % (successes, mean_rewards))
                with open(self.outfile, "a") as f:
                    f.write("%.2f, %.2f,\n" % (successes, mean_rewards))

            
    def add_hindsight_replay_experience(self, states, actions):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of states.
        """
        her_states, her_rewards = self.env.apply_hindsight(states)
        for i in range(len(states)):
            self.buffer.add(states[i], actions[i], her_rewards[i], her_states[i], True)


