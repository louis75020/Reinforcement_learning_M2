# Reinforcement_learning_M2
Reinforcement learning projects


# k - armed bandits :

Strategies & modelisation of the k-armed bandits problem.

Let's try to play with k armed bandits which give random returns : how to find the best machine ?

- Greedy algorithm
- $\epsilon$-greedy algorithm
- Upper bound update
- Gradient update

# Markov decision process :

We are supposed to help a robot to walk on a grid with traps and rewards. Unfortunately some of our actions do not work well and fail with a (stationnary) probabilistic law. The goal is to maximize rewards.

- Value update (We trynna maximize the rewards, weighted by the iteration number & probabilites), we deduce then the policy to apply (in a greedy way).
- Policy update (We start from an arbitrary policy, we compute its weighted returns and we use it to update our policy).

# Markov decision process (without knowledge about the transition matrix) :

We want to learn how to play (a simplified) Blackjack - without insurance, with constant rewards & only one player.

We will:
- evaluate the performance of a given strategy
- use the "on policy Monte Carlo" algorithm to get an optimal strategy

Then we'll try to use our algorithms in another environment (Frozen lake)
