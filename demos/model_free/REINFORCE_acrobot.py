# Implementing REINFORCE Algorithm
# Author: Parijat Parimal

import gym
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
import math

epochs = 3000
alpha = 0.0015
gamma = 0.99

# Initialize environment and weights
env = gym.make('Acrobot-v1') #-todo- Needs to change w.r.t. custom env
nA = env.action_space.n #-todo- Needs to change w.r.t. custom env
nS = 6
np.random.seed(1)
w = np.random.rand(nS, nA)

# Policy to map state to action w.r.t. weights 'w'
def policy(state,w):
    z = state.dot(w)
    exp = np.exp(z)
    return exp/np.sum(exp)

# Vectorized softmax
def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def get_reward(state):
    status = -math.cos(state[4]) - math.cos(state[5] + state[4])
    if (status > 0.):
        return (1+state[0])**2
    return 0

# Incrementatl learning rates
l_rate = [0.0025]
mean_rewards = []
mean_stddevs = []
mean_variances = []
episode_rewards = []
iter = 0

for l in l_rate:
    
    alpha = l
    for e in range(epochs):

        state = env.reset()[None,:] 
        grads = []
        rewards = []
        score = 0
        
        while True:

            # Render Animation - Also needs to change w.r.t. custom env
            if (e>1000 and e%100==0):
                env.render('rgb_array')
            #env.render()

            # Assign probabilities w.r.t. current state and weights
            probs = policy(state,w)
        
            # Choose action with non-uniform randomness w.r.t. probabilities of each action at current state
            action = np.random.choice(nA,p=probs[0])
            # Get next state, reward and game status based on the action taken 
            next_state,reward,done,_ = env.step(action) #-todo- Needs to change w.r.t. custom env

            reward += get_reward(next_state)
            next_state = next_state[None,:]

            # Compute gradient and store reward w.r.t. weight updates
            dsoftmax = softmax_grad(probs)[action,:]
            dlog = dsoftmax / probs[0,action]
            grad = state.T.dot(dlog[None,:])

            grads.append(grad)
            rewards.append(reward)
        
            # Update score
            score+=reward

            # update current state
            state = next_state
        
            # Break when game is over
            if done:
                break

        # REINFORCE weight with rewards from current episode and future rewards as per policy
        for i in range(len(grads)):
            w += alpha * grads[i] * sum([ r * (gamma ** r) for t,r in enumerate(rewards[i:])])

        # Print rewards per episode / epoch
        episode_rewards.append(score) 
        print("Episode: " + str(e) + " Score: " + str(score), end="\r", flush=False)

    iter += 1
    # Plot graph of rewards per episode
    #plt.plot(np.arange(epochs),episode_rewards[epochs*(iter-1):])
    #plt.show()
    print("Mean reward: ",stats.mean(episode_rewards[epochs*(iter-1):]))
    mean_rewards.append(stats.mean(episode_rewards[epochs*(iter-1):]))
    print("Std Deviation: ",stats.stdev(episode_rewards[epochs*(iter-1):]))
    mean_stddevs.append(stats.stdev(episode_rewards[epochs*(iter-1):]))
    print("Variance: ",stats.variance(episode_rewards[epochs*(iter-1):])/epochs)
    mean_variances.append(stats.variance(episode_rewards[epochs*(iter-1):])/epochs)
env.close()

plt.plot(np.arange(epochs*iter),episode_rewards)
plt.show()
print("Overall mean reward: ",stats.mean(episode_rewards))
print("Overall std deviation: ",stats.stdev(episode_rewards))
print("Overall variance: ",stats.variance(episode_rewards)/(epochs*iter))
#plt.figure(figsize=(10, 5))    
#plt.plot(l_rate,mean_rewards)
#plt.plot(l_rate,mean_stddevs)
#plt.plot(l_rate,mean_variances)
#plt.show()
