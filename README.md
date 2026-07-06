# Hidden Markov Models and Markov Decision Processes

## 🧭 Project Overview

### Title: Hidden Markov Models and Markov Decision Processes

### Purpose

This repository contains implementations of two fundamental probabilistic Artificial Intelligence techniques used for reasoning and decision-making under uncertainty.

The first project applies **Hidden Markov Models (HMMs)** to cybersecurity intrusion tracking by reconstructing attacker movement through a network using noisy intrusion detection system (IDS) alarms. The second project applies **Markov Decision Processes (MDPs)** to cloud resource management, using Value Iteration and Policy Iteration to determine optimal server scaling policies.

Together, these projects demonstrate probabilistic inference, sequential decision-making, dynamic programming, and reinforcement learning fundamentals. :contentReference[oaicite:0]{index=0}

### Audience

- Artificial Intelligence Students
- Cybersecurity Students
- Machine Learning Enthusiasts
- Software Engineers
- Researchers

---

# 🧱 Project Scope

## Task 1: Hidden Markov Models (HMM)

Implements probabilistic inference algorithms for tracking an attacker moving through a computer network where the true attacker location is hidden and only noisy IDS alarms are observable.

### Components

- Random Network Topology Generator
- Ground Truth Attack Path Generator
- Viterbi Path Reconstruction
- Bayesian Filtering
- Future State Prediction
- Performance Evaluation
- Network Visualization

### Techniques

- Hidden Markov Models (HMM)
- Dynamic Programming
- Bayesian Inference
- Probabilistic State Estimation
- Log Probability Computation
- Graph-Based Network Modeling

---

## Task 2: Markov Decision Processes (MDP)

Implements dynamic programming algorithms for optimal cloud server scaling decisions under uncertain workloads.

### Components

- Cloud Workload Environment
- Transition Probability Models
- Reward Function
- Value Iteration
- Policy Iteration
- Policy Evaluation
- Convergence Analysis

### Techniques

- Markov Decision Processes
- Bellman Optimality Equation
- Dynamic Programming
- Value Iteration
- Policy Iteration
- Policy Evaluation
- Utility Maximization

---

# 📂 Repository Structure

```text

-Hidden-Markov-Models-and-Markov-Decision-Processes/
│
├── Hidden_Markov_Model/
│   ├── Figures/
│   │   ├── Hmm network 2.png
│   │   └── Hmm network.png
│   │ 
│   ├── Results/
│   │   ├── Ground Truth vs Viterbi Inference 2.png
│   │   ├── Ground Truth vs Viterbi Inference.png
│   │   ├── Inference accuracy vs Alarm Noise.png
│   │   ├── Inference accuracy vs Alarm Sensitivity.png
│   │   ├── Inference accuracy vs Sequence length.png
│   │   └── Inference accuracy vs number of machines.png
│   │ 
│   └── hmm_completed.py
│
├── Markov_Decision_Process/
│   ├── Figures/
│   │   ├── Machine value 3.png
│   │   ├── Machine value 3b.png
│   │   └── Machine value 3c.png
│   │ 
│   ├── Results/
│   │   ├── Default test.png
│   │   ├── Effect of Epsilon on Value Iteration Convergence.png
│   │   ├── Effect of Gamma on Convergence.png
│   │   └── Special test.png
│   │ 
│   ├── MDP cheatsheet.pdf
│   └── mdp_completed.py
│
├── Report/
│   ├── CS Project 5 Report.docx
│
└── README.md
```

---

# 🤖 Algorithms Included

| Category | Algorithm | Purpose |
|-----------|-----------|----------|
| Hidden Markov Model | Viterbi Algorithm | Reconstructs the most likely attacker path from noisy observations |
| Hidden Markov Model | Filtering | Estimates the attacker's current location using Bayesian inference |
| Hidden Markov Model | Prediction | Forecasts future attacker locations using transition probabilities |
| Markov Decision Process | Value Iteration | Computes optimal utilities and policies using Bellman updates |
| Markov Decision Process | Policy Iteration | Alternates policy evaluation and policy improvement until convergence |
| Markov Decision Process | Bellman Equation | Computes expected long-term rewards for each action |

---

# 📊 Results Summary

## Hidden Markov Models

The Hidden Markov Model successfully reconstructed attacker movement through simulated computer networks using noisy IDS alarms.

Experiments evaluated:

- Viterbi path reconstruction accuracy
- Filtering for current attacker localization
- Future attack prediction
- Effect of network size
- Alarm sensitivity
- False positive rates
- Observation sequence length
- Different network topologies

Results showed that inference accuracy remained high under default conditions, improving with higher IDS sensitivity and decreasing as network size, observation length, and alarm noise increased. :contentReference[oaicite:1]{index=1}

---

## Markov Decision Processes

The Markov Decision Process modeled cloud resource allocation under varying workloads.

Experiments compared:

- Value Iteration
- Policy Iteration
- Utility convergence
- Discount factor (γ)
- Convergence threshold (ε)
- Convergence speed

Both Value Iteration and Policy Iteration consistently converged to the same optimal policy, while Policy Iteration required significantly fewer iterations than Value Iteration. Increasing the discount factor (γ) substantially increased Value Iteration convergence time, whereas Policy Iteration remained comparatively stable. :contentReference[oaicite:2]{index=2}

---

# 🚀 How to Run

## Hidden Markov Model

Run the HMM implementation:

```bash
python hmm_completed.py
```

The program automatically:

- Generates a random network topology
- Simulates attacker movement
- Produces noisy IDS observations
- Executes:
  - Viterbi Algorithm
  - Filtering
  - Prediction
- Generates performance graphs and inference visualizations

---

## Markov Decision Process

Run the MDP implementation:

```bash
python mdp_completed.py
```

The program automatically:

- Computes optimal utility values
- Computes the optimal policy
- Executes Value Iteration
- Executes Policy Iteration
- Compares convergence behavior
- Generates convergence graphs and policy evaluations

---

# 💻 Skills Demonstrated

- Artificial Intelligence
- Hidden Markov Models (HMM)
- Markov Decision Processes (MDP)
- Bayesian Inference
- Probabilistic Reasoning
- Dynamic Programming
- Viterbi Algorithm
- Filtering Algorithms
- Prediction Algorithms
- Value Iteration
- Policy Iteration
- Bellman Equations
- Reinforcement Learning Fundamentals
- Graph Algorithms
- Performance Evaluation
- Data Visualization
- Python

---

# 📈 Future Improvements

- Support weighted transition probabilities for HMMs
- Implement Particle Filtering for larger state spaces
- Add Forward-Backward smoothing algorithms
- Extend MDP implementation with Q-Learning
- Add Policy Gradient and Reinforcement Learning approaches
- Simulate larger cloud environments with real workload traces
- Compare exact inference with approximate inference methods

---

# 📄 Documentation

The repository includes:

- Complete Python implementations
- Experimental evaluation
- Performance visualizations
- Network topology diagrams
- Convergence analysis
- Project Report

The full report documents algorithm implementations, experimental design, evaluation metrics, and discussion of results. :contentReference[oaicite:3]{index=3}

---

# 📬 Author

**Bradley Titagwan**

Version: v1.0
