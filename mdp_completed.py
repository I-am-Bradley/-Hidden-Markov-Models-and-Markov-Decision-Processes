# CS-3083 Project 5
# Task 2
# Markov Decision Process

import numpy as np
import matplotlib.pyplot as plt


# states
S = {"Low", "Medium", "High", "Overloaded"}
# actions, applied to all states
A = {"down", "stay", "up", "max"}

# Transition models
# This is for the down action
# Each entry in the outer dictionary is indexed by the current state s.
# Each entry in the inner dictionary is indexed by the next state s'
# Each value represents one probability P(s'|s, action=down)
# You can similarly interpret other action's dictionary
P_down = {
    "Low": {
        "Low": 0.2, "Medium": 0.5, "High": 0.2, "Overloaded": 0.1
    },
    "Medium": {
        "Low": 0.1, "Medium": 0.2, "High": 0.5, "Overloaded": 0.2
    },
    "High": {
        "Low": 0.0, "Medium": 0.2, "High": 0.3, "Overloaded": 0.5
    },
    "Overloaded": {
        "Low": 0.0, "Medium": 0.1, "High": 0.2, "Overloaded": 0.7
    }
}

P_stay = {
    "Low": {
        "Low": 0.5, "Medium": 0.3, "High": 0.2, "Overloaded": 0.0
    },
    "Medium": {
        "Low": 0.2, "Medium": 0.5, "High": 0.2, "Overloaded": 0.1
    },
    "High": {
        "Low": 0.1, "Medium": 0.2, "High": 0.5, "Overloaded": 0.2
    },
    "Overloaded": {
        "Low": 0.0, "Medium": 0.2, "High": 0.3, "Overloaded": 0.5
    }
}

P_up = {
    "Low": {
        "Low": 0.8, "Medium": 0.2, "High": 0.0, "Overloaded": 0.0
    },
    "Medium": {
        "Low": 0.7, "Medium": 0.2, "High": 0.1, "Overloaded": 0.0
    },
    "High": {
        "Low": 0.1, "Medium": 0.7, "High": 0.2, "Overloaded": 0.0
    },
    "Overloaded": {
        "Low": 0.0, "Medium": 0.2, "High": 0.7, "Overloaded": 0.1
    }
}

P_max = {
    "Low": {
        "Low": 1.0, "Medium": 0.0, "High": 0.0, "Overloaded": 0.0
    },
    "Medium": {
        "Low": 0.9, "Medium": 0.1, "High": 0.0, "Overloaded": 0.0
    },
    "High": {
        "Low": 0.3, "Medium": 0.7, "High": 0.0, "Overloaded": 0.0
    },
    "Overloaded": {
        "Low": 0.2, "Medium": 0.2, "High": 0.6, "Overloaded": 0.0
    }
}

P = {"down": P_down, "stay": P_stay, "up": P_up, "max": P_max}


# reward function
Revenue = {"Low": 3.0, "Medium": 6.0, "High": 9.0, "Overloaded": 3.0}
Cost = {"down": 2.0, "stay": 1.0, "up": 3.0, "max": 5.0}

def reward(a, s_prime):
  # The reward is Revenue[s'] - Cost[a].
  # This means the reward depends on the next state reached and the action taken.
  return Revenue[s_prime] - Cost[a]


# discount factor
gamma = 0.90

# error parameter
epsilon = 0.1

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def ordered_states(S):
  # The starter code uses a set for S.
  # Sets do not keep a guaranteed order.
  # Sorting the states gives consistent output every time the program runs.
  return sorted(list(S))


def ordered_actions(A):
  # The starter code uses a set for A.
  # Sorting the actions gives consistent output every time.
  # This also makes ties easier to handle consistently.
  return sorted(list(A))


def Q(P, s, a, U, gamma):
  # General Q function for the whole program.
  #
  # Q(s,a) = sum over s' of:
  # P(s'|s,a) * [R(s,a,s') + gamma * U(s')]
  #
  # This function is used by value iteration, policy calculation,
  # and policy improvement.
  total = 0.0

  for s_prime in P[a][s]:
    transition_probability = P[a][s][s_prime]
    immediate_reward = reward(a, s_prime)
    future_value = gamma * U[s_prime]

    total = total + transition_probability * (
      immediate_reward + future_value
    )

  return total


def one_step_look_ahead(S, A, P, s, U, gamma):
  # This one function performs both jobs requested:
  #
  # 1. Utility calculation:
  #    It finds max_a Q(s,a), which is the Bellman utility update.
  #
  # 2. Policy calculation:
  #    It finds argmax_a Q(s,a), which is the best action for the state.
  #
  # The function returns both values:
  # best_action = argmax_a Q(s,a)
  # best_value = max_a Q(s,a)
  best_action = None
  best_value = -np.inf

  for a in ordered_actions(A):
    q_value = Q(P, s, a, U, gamma)

    if q_value > best_value:
      best_value = q_value
      best_action = a

  return best_action, best_value


def initialize_utilities(S):
  # At the beginning of value iteration, utilities are unknown.
  # Starting all utilities at 0 is standard and simple.
  U = dict()

  for s in ordered_states(S):
    U[s] = 0.0

  return U


def initialize_policy(S, A):
  # Policy iteration needs an initial policy.
  # We choose the first sorted action for every state.
  # The starting policy does not need to be optimal because policy iteration
  # improves it until it becomes stable.
  pi = dict()
  actions = ordered_actions(A)

  for s in ordered_states(S):
    pi[s] = actions[0]

  return pi


def policy_evaluation(S, P, gamma, pi):
  # This performs exact policy evaluation using linear algebra.
  # For a fixed policy, each state has an equation:
  # U(s) = sum over s' P(s' | s, pi(s)) * [R(s, pi(s), s') + gamma * U(s')]
  # These equations can be rearranged into matrix form:
  # A_matrix * U_vector = b_vector
  # np.linalg.solve then solves the full system at once.
  states = ordered_states(S)
  n = len(states)

  A_matrix = np.zeros((n, n))
  b_vector = np.zeros(n)

  for i in range(n):
    s = states[i]
    action = pi[s]

    # Start with U(s) on the left side of the equation.
    A_matrix[i][i] = 1.0

    for j in range(n):
      s_prime = states[j]
      probability = P[action][s][s_prime]

      # Move gamma * P(s'|s,a) * U(s') to the left side.
      A_matrix[i][j] = A_matrix[i][j] - gamma * probability

      # The expected immediate reward stays on the right side.
      b_vector[i] = b_vector[i] + probability * reward(action, s_prime)

  utility_vector = np.linalg.solve(A_matrix, b_vector)

  U = dict()
  for i in range(n):
    U[states[i]] = utility_vector[i]

  return U


def policies_are_equal(pi1, pi2):
  # This checks whether two policies choose the same action in every state.
  # Policy iteration stops when the policy no longer changes.
  for s in pi1:
    if pi1[s] != pi2[s]:
      return False

  return True


def count_policy_disagreements(pi1, pi2):
  # This counts how many states have different actions between two policies.
  # It is used for the special gamma >= 0.99 test.
  disagreements = 0

  for s in pi1:
    if pi1[s] != pi2[s]:
      disagreements = disagreements + 1

  return disagreements


# -------------------------------------------------------------------
# value iteration method
# inputs:
# (1) S: state set
# (2) A: action set
# (3) P: transition model
# (4) gamma: discount factor
# (5) epsilon: error parameter
# Outputs:
# (1) maximum utility vector, as a dictionary
# (2) optimal policy vector, as a dictionary
# -------------------------------------------------------------------

def value_iteration(S, A, P, gamma, epsilon):
  # U stores the current utility estimate for each state.
  U = initialize_utilities(S)

  # pi stores the optimal action found after utilities converge.
  pi = dict()

  # The project asks us to monitor the number of iterations needed to converge.
  iterations = 0

  # The stopping threshold comes from the standard value iteration rule.
  # Smaller epsilon means stricter convergence and usually more iterations.
  threshold = epsilon * (1.0 - gamma) / gamma

  while True:
    # Copy the previous utility values before updating.
    # This prevents one state update from affecting another state in the same pass.
    U_old = U.copy()

    # delta tracks the biggest utility change in this iteration.
    delta = 0.0
    iterations = iterations + 1

    for s in ordered_states(S):
      # Bellman update:
      # U(s) becomes the best expected value over all actions.
      best_action, best_value = one_step_look_ahead(S, A, P, s, U_old, gamma)
      U[s] = best_value

      # Track how much this state's utility changed.
      change = abs(U[s] - U_old[s])

      if change > delta:
        delta = change

    # Stop when the largest utility change is below the threshold.
    if delta < threshold:
      break

  # After utilities converge, extract the policy.
  # For each state, choose the action with the highest one-step look-ahead value.
  for s in ordered_states(S):
    best_action, best_value = one_step_look_ahead(S, A, P, s, U, gamma)
    pi[s] = best_action

  return U, pi, iterations


resU, resPi, resIterations = value_iteration(S, A, P, gamma, epsilon)
print("--- Maximum utility vector found by value iteration ---")
print(resU)
print("--- Optimal policy found by value iteration ---")
print(resPi)
print("--- Number of iterations for value iteration ---")
print(resIterations)


# -------------------------------------------------------------------
# Policy iteration method
# inputs:
# (1) S: state set
# (2) A: action set
# (3) P: transition model
# (4) gamma: discount factor
# outputs:
# optimal policy vector, as a dictionary
# -------------------------------------------------------------------

def policy_iteration(S, A, P, gamma):
  # Start with an arbitrary policy.
  pi = initialize_policy(S, A)

  # The project asks us to monitor the number of iterations needed to converge.
  iterations = 0

  while True:
    iterations = iterations + 1

    # Step 1: Policy evaluation.
    # This computes utilities for the current fixed policy.
    # This implementation uses linear algebra instead of approximation.
    U = policy_evaluation(S, P, gamma, pi)

    # Step 2: Policy improvement.
    # For each state, choose the best action based on the evaluated utilities.
    new_pi = dict()

    for s in ordered_states(S):
      best_action, best_value = one_step_look_ahead(S, A, P, s, U, gamma)
      new_pi[s] = best_action

    # If the policy did not change, it has converged.
    if policies_are_equal(pi, new_pi):
      break

    # Otherwise, keep improving.
    pi = new_pi

  return pi, iterations


resPi2, resIterations2 = policy_iteration(S, A, P, gamma)
print("--- Optimal policy found by policy iteration ---")
print(resPi2)
print("--- Number of iterations for policy iteration ---")
print(resIterations2)


# -------------------------------------------------------------------
# Correctness test
# -------------------------------------------------------------------

def correctness_test():
  # A basic correctness check is that both algorithms should usually find
  # the same optimal policy under the default gamma = 0.90.
  #
  # Value iteration approximates utilities until epsilon convergence.
  # Policy iteration evaluates policies exactly using linear algebra.
  #
  # If both policies match, that supports correctness.
  U_vi, pi_vi, iter_vi = value_iteration(S, A, P, 0.90, 0.1)
  pi_pi, iter_pi = policy_iteration(S, A, P, 0.90)

  print("\n--- Correctness test with gamma = 0.90 and epsilon = 0.1 ---")
  print("Value iteration policy:", pi_vi)
  print("Policy iteration policy:", pi_pi)
  print("Do policies agree?", policies_are_equal(pi_vi, pi_pi))
  print("Value iteration iterations:", iter_vi)
  print("Policy iteration iterations:", iter_pi)


correctness_test()


# -------------------------------------------------------------------
# Plot 1:
# How gamma affects the number of iterations needed to converge.
# -------------------------------------------------------------------

def plot_gamma_effect():
  gamma_values = [0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99]
  vi_iterations = []
  pi_iterations = []

  for g in gamma_values:
    # Epsilon is held constant so gamma is the only parameter being tested.
    U_vi, pi_vi, iter_vi = value_iteration(S, A, P, g, epsilon)
    pi_pi, iter_pi = policy_iteration(S, A, P, g)

    vi_iterations.append(iter_vi)
    pi_iterations.append(iter_pi)

  plt.figure()
  plt.plot(gamma_values, vi_iterations, marker="o",
           label="Value Iteration")
  plt.plot(gamma_values, pi_iterations, marker="x",
           label="Policy Iteration")
  plt.xlabel("Gamma")
  plt.ylabel("Iterations to Converge")
  plt.title("Effect of Gamma on Convergence")
  plt.legend()
  plt.grid(True)
  plt.show()


plot_gamma_effect()


# -------------------------------------------------------------------
# Plot 2:
# How epsilon affects the number of value iteration steps.
# -------------------------------------------------------------------

def plot_epsilon_effect():
  epsilon_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
  vi_iterations = []

  for e in epsilon_values:
    # Gamma is held constant so epsilon is the only parameter being tested.
    U_vi, pi_vi, iter_vi = value_iteration(S, A, P, gamma, e)
    vi_iterations.append(iter_vi)

  plt.figure()
  plt.plot(vi_iterations, epsilon_values, marker="o")
  plt.xlabel("Number of Value Iteration to Converge")
  plt.ylabel("Epsilon")
  plt.title("Effect of Epsilon on Value Iteration Convergence")
  plt.grid(True)
  plt.show()


plot_epsilon_effect()


# -------------------------------------------------------------------
# Special test:
# gamma >= 0.99
# -------------------------------------------------------------------

def special_gamma_test():
  # The project specifically asks us to test gamma >= 0.99.
  # A very high gamma makes the agent care heavily about far-future rewards.
  # This can make convergence slower and can sometimes make small utility
  # differences affect the final policy.
  high_gamma_values = [0.99, 0.995, 0.999]

  print("\n--- Special test for gamma >= 0.99 ---")

  for g in high_gamma_values:
    U_vi, pi_vi, iter_vi = value_iteration(S, A, P, g, epsilon)
    pi_pi, iter_pi = policy_iteration(S, A, P, g)

    disagreements = count_policy_disagreements(pi_vi, pi_pi)

    print("\nGamma:", g)
    print("Value iteration policy:", pi_vi)
    print("Policy iteration policy:", pi_pi)
    print("Disagreements:", disagreements)
    print("Value iteration iterations:", iter_vi)
    print("Policy iteration iterations:", iter_pi)


special_gamma_test()