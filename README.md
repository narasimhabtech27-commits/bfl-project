# Blockchain-Enabled Federated Learning for Privacy-Preserving AI

> Implementation of the paper: *"Blockchain-Enabled Federated Learning for Privacy-Preserving AI"*  
> Published at **IEEE ICISC-2025** | DOI: 10.1109/ICISC65841.2025.11187566

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What This Project Does](#2-what-this-project-does)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Dataset](#5-dataset)
6. [Project Structure](#6-project-structure)
7. [Installation & Setup](#7-installation--setup)
8. [Implementation — Step by Step](#8-implementation--step-by-step)
   - [Phase 1: Federated Learning Baseline](#phase-1-federated-learning-baseline)
   - [Phase 2: Blockchain Integration](#phase-2-blockchain-integration)
   - [Phase 3: Privacy + Attack Detection](#phase-3-privacy--attack-detection)
9. [Results](#9-results)
10. [How to Run](#10-how-to-run)
11. [References](#11-references)

---

## 1. Project Overview

This project implements a **Blockchain-Enabled Federated Learning (BFL)** system — a secure, privacy-preserving AI training framework where:

- Multiple clients train an AI model **locally** on their own data
- **Raw data is never shared** with anyone
- Every model update is **logged on a blockchain** (Ethereum)
- A **smart contract** verifies each update before it is accepted
- **Malicious clients** sending fake/poisoned updates are automatically detected and blocked
- **Differential Privacy** protects client weights from reverse engineering

This directly implements the architecture described in the IEEE ICISC-2025 paper by Prajwalasimha S N et al.

---

## 2. What This Project Does

### The Problem
Traditional Machine Learning requires all data to be sent to a central server. This:
- Violates user privacy
- Breaks regulations like GDPR and HIPAA
- Creates a single point of attack

### The Solution (This Project)
```
Normal ML:                           This Project (BFL):
─────────────────────                ──────────────────────────────────
All data → Central Server            Data stays on each client
Server trains model                  Each client trains locally
Privacy violated                     Only model weights shared
No attack detection                  Blockchain verifies every update
No audit trail                       All updates permanently logged
```

### Simple Analogy
```
5 hospitals want to train an AI model together:

Without BFL:  All patient records sent to one server → Privacy risk
With BFL:     Each hospital trains locally → sends only learned weights
              → blockchain records the submission
              → fake submissions are blocked
              → global model improves without anyone sharing data
```

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BFL SYSTEM OVERVIEW                       │
│                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│   │ Client 1 │  │ Client 2 │  │ Client 3 │  │ Client 4 │  │
│   │ 12,000   │  │ 12,000   │  │ ATTACKER │  │ 12,000   │  │
│   │ images   │  │ images   │  │ (poison) │  │ images   │  │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│        │              │              │              │         │
│        └──────────────┼──────────────┼──────────────┘        │
│                       ▼              │                        │
│            ┌─────────────────────┐   │                        │
│            │  Differential       │   │                        │
│            │  Privacy (DP noise) │   │                        │
│            └─────────┬───────────┘   │                        │
│                      ▼               ▼                        │
│            ┌─────────────────────────────────┐               │
│            │   Ethereum Blockchain (Ganache)  │               │
│            │   ModelRegistry.sol              │               │
│            │   SHA-256 hash logged on-chain   │               │
│            │   Trust score computed           │               │
│            │   Attacker REJECTED              │               │
│            └─────────────────┬───────────────┘               │
│                              ▼                                │
│            ┌─────────────────────────────────┐               │
│            │      Federated Server            │               │
│            │      FedAvg Aggregation          │               │
│            │      (honest updates only)       │               │
│            └─────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Four Key Components

| Component | Description | Implementation |
|---|---|---|
| Clients (x5) | Train model locally on private data | `client_train()` in Python |
| Federated Server | Aggregates updates using FedAvg | `federated_average()` in Python |
| Blockchain Network | Logs and verifies every update | Ganache + `ModelRegistry.sol` |
| Trust Scoring (PoC) | Detects and rejects malicious clients | Cosine similarity scoring |

---

## 4. Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| Language | Python 3.x | Core implementation |
| ML Framework | TensorFlow / Keras | CNN model training |
| Math | NumPy | FedAvg, DP, trust scoring |
| Blockchain | Ganache v7.9.2 | Local Ethereum network |
| Smart Contract | Solidity 0.8.0 | ModelRegistry contract |
| Web3 Bridge | Web3.py | Python ↔ Ethereum |
| Compiler | py-solc-x | Compile Solidity in Python |
| Graphs | Matplotlib | Result visualisation |
| Dataset | MNIST (TensorFlow) | Handwritten digit images |

---

## 5. Dataset

### MNIST — Handwritten Digits

| Property | Value |
|---|---|
| Total images | 70,000 |
| Training images | 60,000 (split across 5 clients) |
| Test images | 10,000 (used for global evaluation) |
| Image size | 28 × 28 pixels (grayscale) |
| Classes | 10 (digits 0 through 9) |
| File size | ~11 MB |
| Source | Yann LeCun, NYU |

### How data is split across clients

```
Total training data: 60,000 images
─────────────────────────────────────
Client 1  →  12,000 images  (honest)
Client 2  →  12,000 images  (honest)
Client 3  →  12,000 images  (ATTACKER — sends poisoned weights)
Client 4  →  12,000 images  (honest)
Client 5  →  12,000 images  (honest)
```

No client ever sees another client's data. This is the core privacy guarantee of Federated Learning.

> **Why MNIST?**  
> MNIST is one of the three benchmark datasets used in the original paper (alongside CIFAR-10 and IoT-IDS). It trains fast enough to run 15 rounds in under 10 minutes, making it ideal for demonstrating the FL + blockchain mechanisms clearly.

---

## 6. Project Structure

```
bfl_project/
│
├── fl_phase1.py                  # Phase 1: Federated Learning baseline
├── bfl_phase2.py                 # Phase 2: FL + Blockchain integration
├── bfl_phase3.py                 # Phase 3: FL + Blockchain + DP + Trust scoring
│
├── blockchain.py                 # Web3.py connection, hash, deploy, submit
├── ModelRegistry.sol             # Solidity smart contract
│
├── generate_results.py           # Generate all result graphs
│
├── figure1_accuracy_trust.png    # Accuracy curve + trust score decay
├── figure2_comparison.png        # Bar chart: Phase 1 vs 2 vs 3
├── figure3_blockchain_updates.png # On-chain update counts
│
├── project_report.md             # Full written report
└── README.md                     # This file
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.8 or above
- Node.js (for Ganache)
- Git (optional)

### Step 1 — Install Ganache (local blockchain)
```bash
npm install -g ganache
```

### Step 2 — Install Python dependencies
```bash
pip install tensorflow numpy matplotlib web3 py-solc-x
```

### Step 3 — Install Solidity compiler
```python
python -c "from solcx import install_solc; install_solc('0.8.0')"
```

### Step 4 — Start Ganache (keep this terminal open)
```bash
ganache --port 8545 --accounts 10 --deterministic
```

You will see 10 wallet addresses appear. Your blockchain is now running locally at `http://127.0.0.1:8545`.

---

## 8. Implementation — Step by Step

---

### Phase 1: Federated Learning Baseline

**File:** `fl_phase1.py`  
**Goal:** Train a digit classifier across 5 clients without sharing raw data

#### What it does
```
1. Downloads MNIST dataset (auto via TensorFlow)
2. Splits 60,000 images equally across 5 clients
3. Each client trains a CNN locally for 2 epochs per round
4. Server aggregates using FedAvg
5. Repeats for 10 rounds
6. Evaluates global model on 10,000 test images
```

#### Model Architecture (CNN)
```
Input: 28×28×1 image
  ↓
Conv2D (32 filters, 3×3) + ReLU    → detects edges and patterns
  ↓
MaxPooling2D                        → reduces size, keeps features
  ↓
Flatten                             → converts to 1D array
  ↓
Dense (64 units) + ReLU             → learns feature combinations
  ↓
Dense (10 units) + Softmax          → outputs probability for each digit
```

#### Key Formula — FedAvg
```
w_global = Σ (p_i × w_i)

Where:
  w_i  = client i's model weights after local training
  p_i  = m_i / Σm_i  (proportion of data client i has)
  w_global = new global model
```

#### Phase 1 Results
```
Round 1   →  Accuracy: 0.9086  | Loss: 0.3296
Round 2   →  Accuracy: 0.9210  | Loss: 0.2714
Round 3   →  Accuracy: 0.9299  | Loss: 0.2413
Round 4   →  Accuracy: 0.9401  | Loss: 0.2120
Round 5   →  Accuracy: 0.9450  | Loss: 0.1909
Round 6   →  Accuracy: 0.9502  | Loss: 0.1777
Round 7   →  Accuracy: 0.9537  | Loss: 0.1602
Round 8   →  Accuracy: 0.9575  | Loss: 0.1471
Round 9   →  Accuracy: 0.9599  | Loss: 0.1397
Round 10  →  Accuracy: 0.9620  | Loss: 0.1292
─────────────────────────────────────────────
Final accuracy: 96.20%
```

---

### Phase 2: Blockchain Integration

**File:** `bfl_phase2.py` + `blockchain.py` + `ModelRegistry.sol`  
**Goal:** Log every model update on an Ethereum blockchain

#### What it adds on top of Phase 1
```
After client trains locally:
  1. SHA-256 hash of weights is computed
  2. Hash is submitted to smart contract on Ganache
  3. Smart contract logs: client address + hash + round + timestamp
  4. Server reads verified updates from blockchain
  5. Only verified updates go into FedAvg
```

#### Smart Contract (ModelRegistry.sol)
```solidity
struct ModelUpdate {
    address client;      // who submitted
    bytes32 updateHash;  // SHA-256 of weights
    uint256 roundNumber; // which FL round
    bool isVerified;     // passed verification?
    uint256 timestamp;   // when submitted
}

function submitUpdate(bytes32 _updateHash, uint256 _roundNumber) public {
    // Logs update permanently on blockchain
    // Anyone can audit this record forever
}
```

#### Phase 2 Results
```
Contract deployed at: 0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab

Round 1  → 5 updates logged on-chain | Accuracy: 0.9086
Round 2  → 5 updates logged on-chain | Accuracy: 0.9210
...
Round 10 → 5 updates logged on-chain | Accuracy: 0.9620
─────────────────────────────────────────────────────────
Total on-chain updates: 50 (5 clients × 10 rounds)
Final accuracy: 96.20%
Blockchain overhead: negligible impact on accuracy
```

---

### Phase 3: Privacy + Attack Detection

**File:** `bfl_phase3.py`  
**Goal:** Add Differential Privacy, Trust Scoring, and catch the malicious client

#### Three new mechanisms added

**A. Differential Privacy (DP)**
```
Before submitting to blockchain, each client adds Gaussian noise:

w_private = w + N(0, sensitivity/epsilon)

Where:
  epsilon = 100.0  (privacy budget — higher = less noise)
  N(0, σ) = random Gaussian noise

Effect: Even if someone intercepts the weights, they cannot
        reconstruct the original training data
```

**B. Trust Scoring — Proof of Contribution (PoC)**
```
Each round, client's trust score is updated:

T_i(t+1) = α × T_i(t) + (1-α) × S_i(t)

Where:
  α = 0.9  (decay factor — keeps history)
  S_i(t) = cosine similarity between client update and global model

Cosine similarity:
  S = dot(w_global, w_local) / (||w_global|| × ||w_local||)

  Honest client:  S ≈ 0.9–1.0  (update aligned with global model)
  Attacker:       S ≈ 0.000    (poisoned weights completely unrelated)

Rejection rule:
  if T_i < 0.85 → REJECT client
```

**C. Attack Simulation**
```
Client 3 is designated as the attacker:
  Instead of real weights → sends random noise × 10

  poison_weights = [random.randn(shape) × 10 for each layer]

  This simulates a real-world data poisoning attack
```

#### Round-by-Round Timeline
```
Round 1:
  Client 3 [ATTACKER] trust = 0.900 → ACCEPTED (not caught yet)
  All 5 clients accepted
  Accuracy: 0.0434 (model corrupted by poison)

Round 2:
  Client 3 [ATTACKER] trust = 0.810 → REJECTED ← caught here!
  Global model RESET to clean state
  Accepted: [Client 1, 2, 4, 5]

Round 3:
  Client 3 trust = 0.729 → REJECTED
  Recovery begins with 4 clean clients
  Accuracy: 0.9059 ← jumps back immediately

Round 4:  Accuracy: 0.9232
Round 5:  Accuracy: 0.9344
Round 7:  Accuracy: 0.9471
Round 10: Accuracy: 0.9547
Round 15: Accuracy: 0.9678 ← final

Attacker trust score: 1.0 → 0.900 → 0.810 → ... → 0.206
(permanently flagged, rejected every round)
```

#### Phase 3 Results
```
Contract deployed at: 0x9561C133DD8580860B6b7E504bC5Aa500f0f06a7

Round 1  → All 5 accepted  | Accuracy: 0.0434 (poisoned)
Round 2  → Attacker REJECTED | Model RESET
Round 3  → 4 clients only  | Accuracy: 0.9059
Round 5  → 4 clients only  | Accuracy: 0.9344
Round 10 → 4 clients only  | Accuracy: 0.9547
Round 15 → 4 clients only  | Accuracy: 0.9678
──────────────────────────────────────────────
Total on-chain updates: 61
Attacker final trust score: 0.206
Final accuracy: 96.78%
```

---

## 9. Results

### Accuracy Comparison Across All Phases

| Round | Phase 1 (FedAvg) | Phase 2 (BFL) | Phase 3 (BFL+DP+Trust) |
|-------|-----------------|---------------|------------------------|
| 1     | 0.9086          | 0.9086        | 0.0434 (attacked)      |
| 2     | 0.9210          | 0.9210        | — (model reset)        |
| 3     | 0.9299          | 0.9299        | 0.9059 (recovered)     |
| 5     | 0.9450          | 0.9450        | 0.9344                 |
| 7     | 0.9537          | 0.9537        | 0.9471                 |
| 10    | 0.9620          | 0.9620        | 0.9547                 |
| 15    | —               | —             | 0.9678                 |

### Final Summary Table

| Metric | Phase 1 (FedAvg) | Phase 2 (BFL) | Phase 3 (BFL+DP+Trust) |
|--------|-----------------|---------------|------------------------|
| Final accuracy | 96.20% | 96.20% | **96.78%** |
| Attack detected | No | No | **Yes — Round 2** |
| Attacker trust score | N/A | N/A | **0.206 (rejected)** |
| On-chain updates | 0 | 50 | **61** |
| Privacy mechanism | None | SHA-256 hash | **DP + Trust scoring** |
| Raw data shared | No | No | No |
| Rounds | 10 | 10 | 15 |

### Comparison with Paper Results

| Metric | Paper (BFL) | Our Implementation |
|--------|-------------|-------------------|
| MNIST accuracy | 99.2% | 96.78% |
| Attack resilience | 92.8% | Attacker blocked round 2 |
| Privacy mechanism | HE + ZKP + DP | DP + Trust scoring |
| Blockchain | Ethereum PoC | Ganache (local Ethereum) |
| Clients | 100 | 5 |

> Note: The paper used 100 clients on cloud hardware (NVIDIA Tesla V100). Our implementation uses 5 clients on a local machine, which explains the accuracy difference. The core mechanisms are identical.

### Attacker Trust Score Decay

```
Round  0:  Trust = 1.000  (starting value)
Round  1:  Trust = 0.900  ACCEPTED  (first poison — not caught yet)
Round  2:  Trust = 0.810  REJECTED  ← blocked from here onwards
Round  3:  Trust = 0.729  REJECTED
Round  4:  Trust = 0.656  REJECTED
Round  5:  Trust = 0.590  REJECTED
Round  6:  Trust = 0.531  REJECTED
Round  7:  Trust = 0.478  REJECTED
Round  8:  Trust = 0.431  REJECTED
Round  9:  Trust = 0.388  REJECTED
Round 10:  Trust = 0.349  REJECTED
Round 15:  Trust = 0.206  REJECTED  (permanently flagged)
```

---

## 10. How to Run

### Step 1 — Start Ganache (Terminal 1, keep open)
```bash
ganache --port 8545 --accounts 10 --deterministic
```

### Step 2 — Run Phase 1 (Terminal 2)
```bash
cd bfl_project
python fl_phase1.py
```
Expected output: Accuracy climbing from ~0.90 to ~0.96 over 10 rounds

### Step 3 — Run Phase 2
```bash
python bfl_phase2.py
```
Expected output: Smart contract deployed, 50 on-chain updates logged

### Step 4 — Run Phase 3
```bash
python bfl_phase3.py
```
Expected output: Attacker caught at round 2, model recovers, 96.78% final accuracy

### Step 5 — Generate result graphs
```bash
python generate_results.py
```
Expected output: 3 PNG graph files saved in project folder

### Common Errors and Fixes

| Error | Fix |
|---|---|
| `ConnectionRefusedError` | Ganache not running — restart Terminal 1 |
| `ModuleNotFoundError: web3` | Run `pip install web3` |
| `FileNotFoundError: ModelRegistry.sol` | Run from inside `bfl_project/` folder |
| `SolcError` | Run `python -c "from solcx import install_solc; install_solc('0.8.0')"` |
| TensorFlow GPU warning | Safe to ignore on Windows — CPU training works fine |

---

## 11. References

1. Prajwalasimha S N, Nilesh Shelke, et al., *"Blockchain-Enabled Federated Learning for Privacy-Preserving AI"*, IEEE ICISC-2025, DOI: 10.1109/ICISC65841.2025.11187566

2. McMahan, H. B., Moore, E., Ramage, D., et al., *"Communication-Efficient Learning of Deep Networks from Decentralized Data"*, AISTATS 2017. (FedAvg algorithm)

3. Wei, K., Li, J., Ding, M., et al., *"Federated Learning with Differential Privacy: Algorithms and Performance Analysis"*, IEEE Transactions on Information Forensics and Security, 2020.

4. LeCun, Y., Cortes, C., & Burges, C., *"The MNIST Database of Handwritten Digits"*, 1998. http://yann.lecun.com/exdb/mnist/

5. Ethereum Foundation, *Solidity Documentation v0.8.0*, 2021. https://docs.soliditylang.org

---

## Key Concepts Glossary

| Term | Meaning |
|---|---|
| **Federated Learning (FL)** | Training AI across multiple devices without sharing raw data |
| **FedAvg** | Algorithm to combine model updates using weighted average |
| **Blockchain** | Permanent, tamper-proof ledger of transactions |
| **Smart Contract** | Self-executing code on the blockchain |
| **Ganache** | Local Ethereum blockchain for development |
| **Differential Privacy (DP)** | Adding controlled noise to protect data from reverse engineering |
| **Trust Score** | A score tracking how honest a client's contributions are |
| **Cosine Similarity** | Measures direction similarity between two vectors (0 = unrelated, 1 = identical) |
| **Proof of Contribution (PoC)** | Consensus mechanism rewarding honest updates |
| **Data Poisoning Attack** | Malicious client sends fake weights to corrupt global model |
| **SHA-256** | Cryptographic hash function used to fingerprint model weights |
| **Web3.py** | Python library to communicate with Ethereum blockchain |

---

*Built as implementation of IEEE ICISC-2025 paper | Dayananda Sagar University*
