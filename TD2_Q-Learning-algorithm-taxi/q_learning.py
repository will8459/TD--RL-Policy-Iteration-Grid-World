import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """

    # Calculate the max Q-value for the next state (sprime)
    max_future_q = np.max(Q[sprime, :])
    
    # Update Q-value using the Bellman equation
    Q[s, a] = Q[s, a] + alpha * (r + gamma * max_future_q - Q[s, a])
    
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    
    # Explore: choose a random action with probability epsilone
    if np.random.rand() < epsilone:

        action = np.random.randint(0, Q.shape[1])
    else:
        # Exploit: choose the best known action with probability 1-epsilon
        action = np.argmax(Q[s, :])
        
    return action


if __name__ == "__main__":
    # Training Phase
    # Create environment (no render_mode to speed up training)
    env = gym.make("Taxi-v3")
    
    env.reset()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # choose your own
    alpha = 0.1      # Learning rate
    gamma = 0.99     # Discount factor
    epsilon = 0.2    # Exploration rate
    
    n_epochs = 10000        # Number of episodes
    max_itr_per_epoch = 100  # Max steps per episode (timeout)
    rewards = []             

    print("Start Training :")

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )
            
            # Update state and put a stoping criteria
            S = Sprime
            if done:
                break

        print("episode #", e, " : r = ", r)

        rewards.append(r)

    print("Average reward = ", np.mean(rewards))
    
    # Plot the rewards in function of epochs
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Training Progress")
    plt.show()

    print("Training finished.\n")

    env.close()

    # Evaluate the q-learning algorihtm
    
    print("Starting Evaluation :")
    
    # Create a new environment with rendering enabled
    env_eval = gym.make("Taxi-v3", render_mode="human")
    n_eval_episodes = 10
    eval_rewards = []

    for e in range(n_eval_episodes):
        S, _ = env_eval.reset()
        r = 0
        done = False
        
        for _ in range(max_itr_per_epoch):
            # For evaluation, we select the BEST action (Greedy), no random exploration
            A = np.argmax(Q[S, :])
            
            Sprime, R, done, _, info = env_eval.step(A)
            r += R
            S = Sprime

            env_eval.render()
            
            if done:
                print("episode #", e, " : r = ", r)
                break

        eval_rewards.append(r)
        
    print("Average reward = ", np.mean(eval_rewards))

    # Plot the evalution rewards in function of epochs
    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Evaluation Rewards")
    plt.show()

    env_eval.close()