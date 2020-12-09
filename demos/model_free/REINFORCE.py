import gym
import numpy as np
import matplotlib.pyplot as plt
import copy 

#-todo- Need to import robot_env to replace gym

epochs = 3000
alpha = 0.0015
gamma = 0.99

# Initialize environment and weights
env = gym.make('CartPole-v1') #-todo-Needs to change w.r.t. custom env
nA = env.action_space.n #-todo-Needs to change w.r.t. custom env
np.random.seed(1)
w = np.random.rand(4, 2)
episode_rewards = []

# Policy to map state to action w.r.t. weights 'w'
def policy(state,w):
    z = state.dot(w)
    exp = np.exp(z)
    return exp/np.sum(exp)

# Vectorized softmax
def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

for e in range(epochs):

    state = env.reset()[None,:] #-todo-Needs to change w.r.t. custom env
    grads = []	
    rewards = []
    score = 0

    while True:

        # Render Animation - Also needs to change w.r.t. custom env
        #if (e%500==0):
            #env.render()
        #env.render()

        # Assign probabilities w.r.t. current state and weights
        probs = policy(state,w)
        
        # Choose action with non-uniform randomness w.r.t. probabilities of each action at current state
        action = np.random.choice(nA,p=probs[0])
        
        # Get next state, reward and game status based on the action taken 
        next_state,reward,done,_ = env.step(action) #-todo-Needs to change w.r.t. custom env
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
        
        # Break loop when game is over
        if done:
            break

    # REINFORCE weight with rewards from current episode and future rewards as per policy
    for i in range(len(grads)):
        w += alpha * grads[i] * sum([ r * (gamma ** r) for t,r in enumerate(rewards[i:])])

    # Print rewards per episode / epoch
    episode_rewards.append(score) 
    print("Episode: " + str(e) + " Score: " + str(score) + "         ",end="\r", flush=False) 

# Plot graph of  rewards per episode
plt.plot(np.arange(epochs),episode_rewards)
plt.show()
env.close()
