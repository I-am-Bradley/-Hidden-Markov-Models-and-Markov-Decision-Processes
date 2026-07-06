# CS-3083 Project 5
# Task 1
# Hidden Markov Model

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


## Provided function 1: Network topology generator
## Parameters
# n: number of machines/states
# p: probability of adding extra edges between machines
def network_generator(n=10, p=0.1, show_graph = True):
  # Step 1: Create a connected backbone first.
  # This guarantees every machine can eventually be reached from every other machine.
  G = nx.random_labeled_tree(n)

  # Step 2: Add extra random edges.
  # The parameter p controls how dense the network becomes.
  # A higher p means the attacker has more possible movement choices.
  for i in range(n):
      for j in range(i + 1, n):
          if not G.has_edge(i, j) and np.random.rand() < p:
              G.add_edge(i, j)

  # Step 3: Visualize the network if it is small enough.
  # This helps us see the possible attacker movement paths.
  if n <= 10 and show_graph:
    plt.figure()
    pos = nx.spring_layout(G, k=5)
    nx.draw(G, pos, with_labels=True)
    plt.title("Connected Erdős–Rényi Random Graph")
    plt.show()

  # Step 4: Convert the NetworkX graph into a dictionary.
  # Each key is a machine.
  # Each value is the set of machines directly connected to it.
  G1 = dict()
  for u in G.nodes:
    G1[u] = set()

  for (u, v) in G.edges:
    G1[u].add(v)
    G1[v].add(u)

  return G1


## Provided function 2: Ground-truth generator
## Parameters
# graph: network topology dictionary
# p1: probability an alarm triggers when the attacker is on that machine
# p2: probability an alarm triggers when the attacker is not on that machine
# N: number of observations
def input_generator(graph, p1, p2, N):
  true_path = []
  alarm_sequence = []

  # Choose a random starting machine.
  # This is X0, which has no observation attached to it.
  true_path.append(np.random.choice(list(graph.keys())))

  # Generate N state transitions and N observations.
  for n in range(N):
    current = true_path[-1]

    # The attacker always moves to one directly connected neighbor.
    # Each neighbor has equal probability.
    next = np.random.choice(list(graph[current]))
    true_path.append(next)

    # Build one full evidence dictionary for this time step.
    # Every machine has an alarm value of either 1 or 0.
    A = dict()

    for a in graph.keys():
      if a == true_path[-1]:
        # True positive case.
        # If the attacker is on this machine, alarm probability is p1.
        A[a] = np.random.choice([1, 0], p=[p1, 1-p1])
      else:
        # False positive case.
        # If the attacker is not on this machine, alarm probability is p2.
        A[a] = np.random.choice([1, 0], p=[p2, 1-p2])

    alarm_sequence.append(A)

  return true_path, alarm_sequence


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def get_states(graph):
  # The states of the HMM are the machines in the network.
  # Sorting makes output easier to read and keeps plots consistent.
  return sorted(list(graph.keys()))


def initial_distribution(graph):
  # The attacker is equally likely to start on any machine.
  # This creates P(X0) as a dictionary.
  states = get_states(graph)
  probability = 1.0 / len(states)

  dist = dict()
  for s in states:
    dist[s] = probability

  return dist


def transition_probability(graph, old_state, new_state):
  # The attacker can only move to connected machines.
  # If new_state is connected to old_state, probability is equal among neighbors.
  # If it is not connected, probability is 0.
  if new_state in graph[old_state]:
    return 1.0 / len(graph[old_state])

  return 0.0


def evidence_probability(state, evidence, graph, p1, p2):
  # This computes P(evidence | state).
  # The evidence contains the alarm status for every machine.
  # Alarms are independent, so the total probability is a product.
  probability = 1.0
  states = get_states(graph)

  for machine in states:
    alarm_value = evidence[machine]

    if machine == state:
      # This is the machine being attacked.
      # Alarm = 1 has probability p1.
      # Alarm = 0 has probability 1 - p1.
      if alarm_value == 1:
        probability = probability * p1
      else:
        probability = probability * (1.0 - p1)

    else:
      # This is not the attacked machine.
      # Alarm = 1 is a false positive with probability p2.
      # Alarm = 0 has probability 1 - p2.
      if alarm_value == 1:
        probability = probability * p2
      else:
        probability = probability * (1.0 - p2)

  return probability


def log_evidence_probability(state, evidence, graph, p1, p2):
  # Viterbi multiplies many small probabilities.
  # Multiplying many small numbers can underflow toward 0.
  # To avoid that, Viterbi uses log probabilities.
  # Since log(a*b) = log(a) + log(b), we add logs instead of multiplying.
  probability = evidence_probability(state, evidence, graph, p1, p2)

  if probability == 0:
    return -np.inf

  return np.log(probability)


def normalize(distribution):
  # Filtering requires normalization after each evidence update.
  # This makes all probabilities add up to 1.
  total = 0.0

  for key in distribution:
    total = total + distribution[key]

  if total == 0:
    return distribution

  for key in distribution:
    distribution[key] = distribution[key] / total

  return distribution


def most_likely_state(distribution):
  # Returns the state with the highest probability in a distribution.
  return max(distribution, key=distribution.get)


def calculate_accuracy(true_path, inferred_path):
  # The true path includes X0 through XN.
  # The inferred Viterbi path corresponds to X1 through XN because evidence starts at e1.
  # Therefore, true_path[i + 1] is compared with inferred_path[i].
  correct = 0

  for i in range(len(inferred_path)):
    if true_path[i + 1] == inferred_path[i]:
      correct = correct + 1

  return correct / len(inferred_path)


# -------------------------------------------------------------------
# Inference 1: Viterbi algorithm
# Finds the most likely complete attack path.
# -------------------------------------------------------------------

def viterbi(graph, evidence_sequence, p1, p2):
  states = get_states(graph)
  initial = initial_distribution(graph)

  # viterbi_table[t][s] stores the best log probability of any path
  # that ends in state s at time t.
  viterbi_table = []

  # backpointer[t][s] stores the best previous state that led to s.
  # This allows us to reconstruct the best path after the recursion finishes.
  backpointer = []

  # -----------------------------
  # Initialization for time 1
  # -----------------------------
  first_column = dict()
  first_backpointer = dict()

  for s in states:
    # P(X0=s) is the initial probability.
    # P(e1 | X1=s) is the evidence probability.
    # We use logs to reduce underflow.
    if initial[s] == 0:
      first_column[s] = -np.inf
    else:
      first_column[s] = np.log(initial[s]) + log_evidence_probability(
        s, evidence_sequence[0], graph, p1, p2
      )

    first_backpointer[s] = None

  viterbi_table.append(first_column)
  backpointer.append(first_backpointer)

  # -----------------------------
  # Recursion for times 2 through N
  # -----------------------------
  for t in range(1, len(evidence_sequence)):
    current_column = dict()
    current_backpointer = dict()

    for current_state in states:
      best_score = -np.inf
      best_previous_state = None

      for previous_state in states:
        transition = transition_probability(graph, previous_state,
                                            current_state)

        if transition > 0:
          # Candidate path score:
          # previous best path score
          # plus log transition probability
          # We add logs instead of multiplying probabilities.
          candidate_score = (
            viterbi_table[t - 1][previous_state] + np.log(transition)
          )

          if candidate_score > best_score:
            best_score = candidate_score
            best_previous_state = previous_state

      # Add the evidence score for the current state at this time.
      current_column[current_state] = (
        best_score + log_evidence_probability(
          current_state, evidence_sequence[t], graph, p1, p2
        )
      )

      current_backpointer[current_state] = best_previous_state

    viterbi_table.append(current_column)
    backpointer.append(current_backpointer)

  # -----------------------------
  # Termination and traceback
  # -----------------------------
  final_state = max(viterbi_table[-1], key=viterbi_table[-1].get)
  inferred_path = [final_state]

  # Walk backward through the backpointers.
  for t in range(len(evidence_sequence) - 1, 0, -1):
    previous = backpointer[t][inferred_path[0]]
    inferred_path.insert(0, previous)

  return inferred_path


# -------------------------------------------------------------------
# Inference 2: Filtering
# Finds P(XN | e1:N), the current attacker location distribution.
# -------------------------------------------------------------------

def filtering(graph, evidence_sequence, p1, p2):
  states = get_states(graph)

  # Start with the uniform prior distribution.
  forward_message = initial_distribution(graph)

  # Process each evidence item one time step at a time.
  for evidence in evidence_sequence:
    new_message = dict()

    for current_state in states:
      prediction_sum = 0.0

      # Prediction step:
      # Sum over all possible previous states.
      for previous_state in states:
        prediction_sum = prediction_sum + (
          forward_message[previous_state]
          * transition_probability(graph, previous_state, current_state)
        )

      # Update step:
      # Multiply prediction by the evidence probability.
      new_message[current_state] = prediction_sum * evidence_probability(
        current_state, evidence, graph, p1, p2
      )

    # Normalize after each time step.
    forward_message = normalize(new_message)

  return forward_message


# -------------------------------------------------------------------
# Inference 3: Prediction
# Finds P(XN+d | e1:N) for future d values.
# -------------------------------------------------------------------

def prediction(graph, current_distribution, max_steps):
  states = get_states(graph)

  # Start from the filtered current distribution.
  current = current_distribution.copy()

  # Store each future distribution separately.
  predictions = []

  for d in range(1, max_steps + 1):
    next_distribution = dict()

    for future_state in states:
      total = 0.0

      # Use only the transition model.
      # No future evidence is available during prediction.
      for previous_state in states:
        total = total + (
          current[previous_state]
          * transition_probability(graph, previous_state, future_state)
        )

      next_distribution[future_state] = total

    current = normalize(next_distribution)
    predictions.append(current.copy())

  return predictions



def average_accuracy(nMachines, connectionDensity, alarmRateAttackOn,
                     alarmRateAttackOff, nObservations, runs):
  # This helper runs the full HMM experiment multiple times.
  # Multiple runs are needed because the network, path, and alarms are random.

  accuracies = []

  for r in range(runs):
    net = network_generator(nMachines, connectionDensity, False)
    path1, seq1 = input_generator(net, alarmRateAttackOn,
                                  alarmRateAttackOff, nObservations)

    inferred_path = viterbi(net, seq1, alarmRateAttackOn, alarmRateAttackOff)
    acc = calculate_accuracy(path1, inferred_path)
    accuracies.append(acc)

  return np.mean(accuracies)



# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------

def plot_true_vs_inferred(true_path, inferred_path, title):
  # This plot compares the true attack path with the Viterbi path.
  # The true path starts at X0, so the plot uses true_path[1:] for comparison.
  plt.figure()
  plt.plot(range(1, len(true_path)), true_path[1:], marker="o",
           label="Ground Truth")
  plt.plot(range(1, len(true_path)), inferred_path, marker="x",
           label="Viterbi Inference")
  plt.xlabel("Time Step")
  plt.ylabel("Machine")
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()


def plot_parameter_results(x_values, y_values, xlabel, title):
  # Generic helper for control parameter plots.
  # It shows how average Viterbi accuracy changes as one parameter changes.
  plt.figure()
  plt.plot(x_values, y_values, marker="o")
  plt.xlabel(xlabel)
  plt.ylabel("Average Inference Accuracy")
  plt.title(title)
  plt.grid(True)
  plt.show()


# -------------------------------------------------------------------
# Main testing and evaluation code
# -------------------------------------------------------------------

## These are input parameters.
nMachines = 5
connectionDensity = 0.2
alarmRateAttackOn = 0.95
alarmRateAttackOff = 0.05
nObservations = 20

## Create a network, a ground-truth attack path, and an evidence sequence.
net = network_generator(nMachines, connectionDensity, True)
path1, seq1 = input_generator(net, alarmRateAttackOn,
                              alarmRateAttackOff, nObservations)

if len(path1) <= 21:
    print("Ground-truth path:")
    print(path1)
    print("Evidence sequence:")
    print(seq1)


# -------------------------------------------------------------------
# Inference 1: most likely attacking path
# -------------------------------------------------------------------

inferred_path = viterbi(net, seq1, alarmRateAttackOn, alarmRateAttackOff)
accuracy = calculate_accuracy(path1, inferred_path)

print("\n--- Viterbi inferred attack path ---")
print(inferred_path)
print("Viterbi accuracy:", accuracy)


# -------------------------------------------------------------------
# Inference 2: filtering
# -------------------------------------------------------------------

current_distribution = filtering(net, seq1, alarmRateAttackOn,
                                 alarmRateAttackOff)

print("\n--- Filtering distribution P(XN | e1:N) ---")
print(current_distribution)
print("Most likely current attacker location:",
      most_likely_state(current_distribution))


# -------------------------------------------------------------------
# Inference 3: prediction for d = 1, 2, 3, 4, 5
# -------------------------------------------------------------------

future_predictions = prediction(net, current_distribution, 5)

print("\n--- Future prediction distributions ---")
for i in range(len(future_predictions)):
  print("d =", i + 1, future_predictions[i],
        "Most likely:", most_likely_state(future_predictions[i]))


# -------------------------------------------------------------------
# Required plot examples:
# One plot with the current run.
# Then another run is generated to show a different accuracy case.
# -------------------------------------------------------------------

plot_true_vs_inferred(path1, inferred_path,
                      "Ground Truth vs Viterbi Inference")

net2 = network_generator(nMachines, connectionDensity, True)
path2, seq2 = input_generator(net2, alarmRateAttackOn,
                              alarmRateAttackOff, nObservations)
inferred_path2 = viterbi(net2, seq2, alarmRateAttackOn,
                         alarmRateAttackOff)

plot_true_vs_inferred(path2, inferred_path2,
                      "Second Run: Ground Truth vs Viterbi Inference")


# -------------------------------------------------------------------
# Control parameter test 1:
# Number of machines
# -------------------------------------------------------------------

machine_values = [3, 5, 7, 9, 12]
machine_accuracies = []

for machines in machine_values:
  avg = average_accuracy(machines, connectionDensity,
                         alarmRateAttackOn, alarmRateAttackOff,
                         nObservations, 10)
  machine_accuracies.append(avg)

plot_parameter_results(machine_values, machine_accuracies,
                       "Number of Machines",
                       "Inference Accuracy vs Number of Machines")


# -------------------------------------------------------------------
# Control parameter test 2:
# Alarm sensitivity / true positive rate
# -------------------------------------------------------------------

sensitivity_values = [0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
sensitivity_accuracies = []

for sensitivity in sensitivity_values:
  avg = average_accuracy(nMachines, connectionDensity,
                         sensitivity, alarmRateAttackOff,
                         nObservations, 10)
  sensitivity_accuracies.append(avg)

plot_parameter_results(sensitivity_values, sensitivity_accuracies,
                       "Alarm Sensitivity / True Positive Rate",
                       "Inference Accuracy vs Alarm Sensitivity")


# -------------------------------------------------------------------
# Control parameter test 3:
# Noise / false positive rate
# -------------------------------------------------------------------

noise_values = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]
noise_accuracies = []

for noise in noise_values:
  avg = average_accuracy(nMachines, connectionDensity,
                         alarmRateAttackOn, noise,
                         nObservations, 10)
  noise_accuracies.append(avg)

plot_parameter_results(noise_values, noise_accuracies,
                       "Noise / False Positive Rate",
                       "Inference Accuracy vs Alarm Noise")


# -------------------------------------------------------------------
# Control parameter test 4:
# Length of sequence
# -------------------------------------------------------------------

sequence_values = [5, 10, 20, 40, 60]
sequence_accuracies = []

for sequence_length in sequence_values:
  avg = average_accuracy(nMachines, connectionDensity,
                         alarmRateAttackOn, alarmRateAttackOff,
                         sequence_length, 10)
  sequence_accuracies.append(avg)

plot_parameter_results(sequence_values, sequence_accuracies,
                       "Length of Observation Sequence",
                       "Inference Accuracy vs Sequence Length")