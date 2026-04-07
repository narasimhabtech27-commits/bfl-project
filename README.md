# Blockchain-Enabled Federated Learning — BFL System
### Secure, Privacy-Preserving AI Training with Decentralized Verification

> **Dayananda Sagar University | Department of Computer Science & Engineering (Cyber Security)**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Ethereum](https://img.shields.io/badge/Ethereum-Ganache-3C3C3D?style=flat&logo=ethereum&logoColor=white)
![Solidity](https://img.shields.io/badge/Solidity-0.8.0-363636?style=flat&logo=solidity&logoColor=white)
![Web3](https://img.shields.io/badge/Web3.py-Blockchain-F16822?style=flat)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

> *Based on the IEEE ICISC-2025 paper: "Blockchain-Enabled Federated Learning for Privacy-Preserving AI"*
> *by Prajwalasimha S N, Nilesh Shelke, Dilip Kumar Saini, Amit Pimpalkar, G Hemanth Kumar, Monish L*
> *DOI: 10.1109/ICISC65841.2025.11187566*

---

## Overview

**Problem:** Traditional Machine Learning requires all training data to be sent to a central server — violating user privacy, breaking regulations like GDPR and HIPAA, and creating a single point of attack.

**Why traditional FL fails:** Standard Federated Learning (FL) solves the data-sharing problem but remains vulnerable to malicious clients who submit poisoned model updates, free-riders who contribute nothing, and unverifiable contributions with no audit trail.

**Our solution:** A Blockchain-Enabled Federated Learning (BFL) system that trains a digit-recognition CNN across 5 private clients using Federated Averaging (FedAvg), logs every model update as an immutable transaction on a local Ethereum blockchain (Ganache) via a Solidity smart contract, applies Differential Privacy noise to protect client weights, and uses a Proof-of-Contribution (PoC) trust scoring mechanism to automatically detect and permanently block malicious clients — all without any raw data ever leaving a client.

**Key models used:** CNN (primary classifier), FedAvg (aggregation), Cosine Similarity Trust Scoring (PoC), Ethereum Smart Contract (ModelRegistry.sol).

**Final performance highlights:**
-  **96.78% accuracy** on MNIST test set (10,000 images)
-  **Malicious attacker detected and blocked at Round 2**
   **61 tamper-proof on-chain transactions** recorded
-  **Model recovered to 90.59% in one round** after attacker removal
-  **Zero raw data shared** across all 15 training rounds

**Keywords:** Federated Learning · Blockchain · Privacy-Preserving AI · Ethereum · Differential Privacy · Smart Contracts · Proof-of-Contribution · Data Poisoning · MNIST · Cybersecurity

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Proposed Architecture](#2-proposed-architecture)
3. [How It Works](#3-how-it-works)
4. [Dataset](#4-dataset)
5. [Results & Metrics](#5-results--metrics)
6. [Code Architecture](#6-code-architecture)
7. [Core Modules — Deep Dive](#7-core-modules--deep-dive)
8. [Setup & Usage](#8-setup--usage)
9. [Implementation — Phase by Phase](#9-implementation--phase-by-phase)
10. [Limitations & Future Work](#10-limitations--future-work)
11. [Team](#11-team)
12. [Mentor](#12-mentor)
13. [References](#13-references)

---

## 1. Problem Statement

> *"Training AI models collaboratively across distributed clients requires mechanisms that guarantee data privacy, verify honest contributions, and resist adversarial manipulation — none of which standard Federated Learning provides."*

### Why traditional methods fail

| Method | Limitation |
|---|---|
| Centralized ML | All raw data sent to one server — complete privacy violation |
| Standard FL (FedAvg) | No verification of client updates — malicious clients corrupt the global model |
| FL with encryption only | No audit trail, no accountability, free-rider problem unsolved |
| Signature-based IDS | Cannot detect novel poisoning attack patterns |

### Why the problem exists

- Federated Learning clients are untrusted — any device can submit arbitrary weight updates
- There is no central authority to verify contributions in a decentralized system
- Model poisoning attacks (sending large random noise as weights) can destroy global model accuracy within a single round
- No existing lightweight mechanism simultaneously provides: privacy + verification + accountability + attack resilience

### What is needed

- A **trustless verification layer** (blockchain) that records every update permanently
- A **privacy mechanism** (Differential Privacy) that prevents weight reconstruction
- A **trust scoring system** (Proof-of-Contribution) that identifies and blocks bad actors
- A **reproducible simulation** demonstrating all three mechanisms working together

---

## 2. Proposed Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    BFL SYSTEM ARCHITECTURE                       │
│                                                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│  │Client 1 │ │Client 2 │ │Client 3 │ │Client 4 │ │Client 5 │     │  
│  │ 12,000  │ │ 12,000  │ │ATTACKER │ │ 12,000  │ │ 12,000  │     │
│  │ images  │ │ images  │ │(poison) │ │ images  │ │ images  │     │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘     │
│       └───────────┴───────────┴───────────┴───────────┘          │
│                               │                                  │
│              ┌────────────────▼──────────────────┐               │
│              │      Differential Privacy         │               │
│              │   w_private = w + N(0, sigma)     │               │
│              └────────────────┬──────────────────┘               │
│                               │                                  │
│              ┌────────────────▼──────────────────┐               │
│              │   Ethereum Blockchain (Ganache)   │               │
│              │   ModelRegistry.sol               │               │
│              │   SHA-256 hash logged on-chain    │               │
│              │   Client address + timestamp saved│               │
│              └────────────────┬──────────────────┘               │
│                               │                                  │
│              ┌────────────────▼──────────────────┐               │
│              │   Trust Scoring — PoC             │               │
│              │   T_i = alpha*T_i + (1-alpha)*S_i │               │
│              │   if T_i < 0.85 → REJECT          │               │
│              └────────────────┬──────────────────┘               │
│                               │                                  │
│              ┌────────────────▼──────────────────┐               │
│              │   Federated Server — FedAvg       │               │
│              │   w = sum(p_i * w_i)              │               │
│              │   honest clients only             │               │
│              └───────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

### System Components

| # | Component | Role | Output |
|---|---|---|---|
| 1 | Clients (x5) | Train CNN locally on private MNIST partition | Local model weights |
| 2 | Differential Privacy | Add Gaussian noise before submission | Protected weight vectors |
| 3 | Blockchain (Ganache) | Log SHA-256 hash of every update on-chain | Immutable transaction records |
| 4 | Smart Contract | Verify and store model update metadata | On-chain audit log |
| 5 | Trust Scoring (PoC) | Cosine similarity trust score per client | Accept / Reject decision |
| 6 | Federated Server | FedAvg over verified updates only | Updated global model |
| 7 | Global Model | Evaluated on 10,000 test images each round | Accuracy + Loss metrics |

---

## 3. How It Works

### Decision Logic

```
Each FL Round:
─────────────────────────────────────────────────────────────
  For each client i:
    1. Receive global model weights from server
    2. Train locally on private dataset (2 epochs, SGD)
    3. Add differential privacy noise to weights
    4. Compute SHA-256 hash of noisy weights
    5. Submit hash to blockchain (smart contract)
    6. Compute cosine similarity with global model
    7. Update trust score: T_i = 0.9*T_i + 0.1*S_i

    If T_i >= 0.85  →  ACCEPTED  →  included in FedAvg
    If T_i <  0.85  →  REJECTED  →  excluded permanently
        (first rejection → global model RESET to clean state)

  Server:
    Collect only accepted updates
    w_global = sum(p_i * w_i)   [FedAvg]
    Evaluate on 10,000 test images
    Broadcast updated global model to all clients
─────────────────────────────────────────────────────────────
```

### Flow Diagram

```
Raw MNIST Data
      ↓
Split across 5 Clients (12,000 each)
      ↓
Local CNN Training (SGD, 2 epochs per round)
      ↓
Differential Privacy Noise Added (epsilon=100)
      ↓
SHA-256 Hash → Ethereum Smart Contract (Ganache)
      ↓
Trust Score Computed via Cosine Similarity
      ↓
Accept / Reject Decision (threshold = 0.85)
      ↓
FedAvg Aggregation (honest clients only)
      ↓
Global Model Evaluated on Test Set
      ↓
Next Round  ↺  (repeat up to 15 rounds)
```

### Key Equations

**FedAvg Aggregation:**
```
w_global(t+1) = sum over i of [ p_i * w_i(t+1) ]
p_i = m_i / sum(m_i)    (data proportion of client i)
```

**Local SGD Update:**
```
w_i(t+1) = w_i(t) - eta * gradient_L_i(w_i(t))
eta = learning rate,  L_i = cross-entropy loss
```

**Differential Privacy:**
```
w_private = w + N(0, sensitivity / epsilon)
epsilon = 100.0,  sensitivity = 1.0,  noise_scale = 0.01
```

**Trust Score — Proof of Contribution:**
```
S_i(t)   = dot(w_global, w_local) / (||w_global|| * ||w_local||)
T_i(t+1) = 0.9 * T_i(t) + 0.1 * S_i(t)
if T_i < 0.85  →  REJECTED
```

**Accuracy:**
```
A = Correct Predictions / Total Predictions
```

---

## 4. Dataset

### MNIST — Handwritten Digit Recognition

| Property | Value |
|---|---|
| Full name | Modified National Institute of Standards and Technology |
| Total images | 70,000 grayscale images |
| Training set | 60,000 images (split across 5 clients) |
| Test set | 10,000 images (global evaluation each round) |
| Image dimensions | 28 x 28 pixels (784 features per image) |
| Colour | Grayscale (0 = black, 255 = white, normalized to 0.0–1.0) |
| Classes | 10 (digits 0 through 9) |
| File size | ~11 MB |
| Source | Yann LeCun, Corinna Cortes — NYU |
| Download | Automatic via tf.keras.datasets.mnist.load_data() |

### How Data is Split Across Clients

```
Total Training Images: 60,000
──────────────────────────────────────────
Client 1  →  12,000 images  (honest)
Client 2  →  12,000 images  (honest)
Client 3  →  12,000 images  (ATTACKER — sends poisoned weights)
Client 4  →  12,000 images  (honest)
Client 5  →  12,000 images  (honest)
──────────────────────────────────────────
Test Set  →  10,000 images  (server only — never shared)
```

> No client ever sees another client's images. This is the core privacy guarantee.

### Why MNIST?

- One of three benchmark datasets in the original paper (alongside CIFAR-10 and IoT-IDS)
- Trains fast — one round completes in ~30 seconds on CPU
- Universal recognition — 96%+ accuracy is immediately meaningful to any examiner
- Simple enough to keep focus on blockchain + privacy mechanisms, not dataset complexity

---

## 5. Results & Metrics

### Accuracy Across All Phases

| Round | Phase 1 — FedAvg | Phase 2 — BFL | Phase 3 — BFL+DP+Trust |
|-------|-----------------|---------------|------------------------|
| 1     | 0.9086          | 0.9086        | 0.0434  (attacked)     |
| 2     | 0.9210          | 0.9210        | —  (model reset)       |
| 3     | 0.9299          | 0.9299        | 0.9059  (recovered)    |
| 4     | 0.9401          | 0.9401        | 0.9232                 |
| 5     | 0.9450          | 0.9450        | 0.9344                 |
| 6     | 0.9502          | 0.9502        | 0.9385                 |
| 7     | 0.9537          | 0.9537        | 0.9471                 |
| 8     | 0.9575          | 0.9575        | 0.9503                 |
| 9     | 0.9599          | 0.9599        | 0.9551                 |
| 10    | 0.9620          | 0.9620        | 0.9547                 |
| 15    | —               | —             | **0.9678**             |

### Final Summary Table

| Metric | Phase 1 — FedAvg | Phase 2 — BFL | Phase 3 — BFL+DP+Trust |
|--------|-----------------|---------------|------------------------|
| Final accuracy | 96.20% | 96.20% | **96.78%** |
| Attack detected | No | No | **Yes — Round 2** |
| Attacker final trust | N/A | N/A | **0.206 (rejected)** |
| On-chain updates | 0 | 50 | **61** |
| Privacy mechanism | None | SHA-256 hash | **DP + Trust scoring** |
| Raw data shared | Never | Never | Never |
| Rounds to converge | 10 | 10 | 15 (inc. recovery) |

### Attacker Trust Score Decay

```
Round  0:  Trust = 1.000   starting value
Round  1:  Trust = 0.900   ACCEPTED  (not caught yet)
Round  2:  Trust = 0.810   REJECTED  ← blocked permanently
Round  3:  Trust = 0.729   REJECTED
Round  4:  Trust = 0.656   REJECTED
Round  5:  Trust = 0.590   REJECTED
Round  7:  Trust = 0.478   REJECTED
Round 10:  Trust = 0.349   REJECTED
Round 15:  Trust = 0.206   REJECTED  (permanently flagged)
```

### Comparison With Paper

| Metric | Paper (Full BFL) | Our Implementation |
|--------|-----------------|-------------------|
| MNIST accuracy | 99.2% | 96.78% |
| Attack resilience | 92.8% | Attacker blocked round 2 |
| Privacy mechanism | HE + ZKP + DP | DP + Trust scoring |
| Blockchain | Ethereum PoC | Ganache (local Ethereum) |
| Number of clients | 100 | 5 |
| Hardware | NVIDIA Tesla V100 GPU | Local CPU (Windows) |

> The paper used 100 clients on cloud GPU hardware. Our implementation with 5 clients on local CPU achieves the same core results, demonstrating all mechanisms correctly.

### Performance Graphs

```
Generated by:  python generate_results.py

figure1_accuracy_trust.png      Accuracy curve Phase 2 vs Phase 3
                                 + Attacker trust score decay

figure2_comparison.png           Bar chart: final accuracy per phase

figure3_blockchain_updates.png   On-chain update counts per phase
```

---

## 6. Code Architecture

```
bfl_project/
│
├── fl_phase1.py                      Phase 1: Federated Learning baseline
├── bfl_phase2.py                     Phase 2: FL + Blockchain integration
├── bfl_phase3.py                     Phase 3: FL + DP + Trust scoring
│
├── blockchain.py                     Web3.py connection, hash, deploy, submit
├── ModelRegistry.sol                 Solidity smart contract (Ethereum)
│
├── generate_results.py               Generate all result graphs
│
├── figure1_accuracy_trust.png        Accuracy + trust score chart
├── figure2_comparison.png            Phase comparison bar chart
├── figure3_blockchain_updates.png    On-chain updates chart
│
├── project_report.md                 Full written project report
└── README.md                         This file
```

---

## 7. Core Modules — Deep Dive

### Federated Learning Core (`fl_phase1.py`)

**File:** `fl_phase1.py`

**What it does:** Simulates a complete FL training loop. 5 clients each train a CNN locally on 12,000 private MNIST images, then a central server aggregates using FedAvg for 10 rounds with no raw data exchange.

**CNN Architecture:**
```
Input:  28x28x1 image
  Conv2D (32 filters, 3x3, ReLU)   detects edges and digit patterns
  MaxPooling2D                      reduces spatial size by half
  Flatten                           converts 2D feature map to 1D
  Dense (64 units, ReLU)            learns high-level combinations
  Dense (10 units, Softmax)         outputs probability for each digit
Output: probability vector [digit 0, digit 1, ..., digit 9]
```

**Equation:**
```
w_global = sum(p_i * w_i)    where p_i = m_i / sum(m_i)
```

---

### Blockchain Layer (`blockchain.py` + `ModelRegistry.sol`)

**Files:** `blockchain.py`, `ModelRegistry.sol`

**What it does:** Connects Python to local Ethereum blockchain (Ganache) using Web3.py. Compiles and deploys the Solidity smart contract. Each client submits a SHA-256 fingerprint of their weights — permanently logged on-chain with client address, round number, and timestamp.

**Smart contract structure:**
```solidity
struct ModelUpdate {
    address client;       // Ethereum wallet of submitting client
    bytes32 updateHash;   // SHA-256 of model weights
    uint256 roundNumber;  // which FL round (1-10 or 1-15)
    bool isVerified;      // passed smart contract check
    uint256 timestamp;    // block timestamp
}
```

**Key functions:**
```solidity
submitUpdate(bytes32 hash, uint256 round)   // client calls this
getVerifiedUpdates(uint256 round)           // server calls this
getTotalUpdates()                           // audit total submissions
```

---

### Differential Privacy (`bfl_phase3.py`)

**Function:** `add_differential_privacy(weights, epsilon=100.0)`

**What it does:** Adds calibrated Gaussian noise to client weights before submission. Prevents an adversary from reconstructing original training data from intercepted weight vectors.

**Equation:**
```
w_private = w + N(0, sensitivity/epsilon)
epsilon = 100.0   (privacy budget)
noise   = 0.01 per weight value
```

---

### Trust Scoring — Proof of Contribution (`bfl_phase3.py`)

**Function:** `compute_trust_score(global_weights, local_weights, prev_score)`

**What it does:** Measures how similar a client's update direction is to the global model each round. Maintains running history with exponential decay. Attackers consistently diverge → trust falls below threshold → permanently blocked.

**Equation:**
```
S_i    = dot(w_global, w_local) / (||w_global|| * ||w_local||)
T_i    = 0.9 * T_i_prev + 0.1 * S_i
REJECT if T_i < 0.85
```

**Why it works:**
```
Honest client:   S_i ~ 0.9–1.0  (update aligned with global)  → trust stays high
Attacker:        S_i ~ 0.000    (random noise, unrelated)      → trust decays fast
```

---

### Attack Simulation (`bfl_phase3.py`)

**Function:** `poison_weights(weights)`

**What it does:** Simulates a real-world data poisoning attack. Client 3 replaces its legitimate model update with large-scale random noise to corrupt the global model.

```python
def poison_weights(weights):
    return [np.random.randn(*w.shape) * 10 for w in weights]
    # Scale x10 ensures weights are wildly different from honest updates
    # Cosine similarity with global model ≈ 0.000
```

---

### Result Generation (`generate_results.py`)

**What it does:** Reads your experimental output and produces 3 publication-quality graphs using matplotlib — accuracy comparison chart, attacker trust score decay with rejection threshold, bar charts comparing phases and on-chain update counts.

---

## 8. Setup & Usage

### Requirements

| Component | Version |
|---|---|
| Python | 3.8+ (3.11 recommended) |
| TensorFlow | 2.x |
| NumPy | Any recent |
| Web3.py | 6.x |
| py-solc-x | Any |
| Matplotlib | Any |
| Node.js | For Ganache |
| Ganache CLI | v7.9.2 |

### Installation

```bash
# Step 1: Install Ganache (local Ethereum blockchain)
npm install -g ganache

# Step 2: Install Python dependencies
pip install tensorflow numpy matplotlib web3 py-solc-x

# Step 3: Install Solidity compiler (run once only)
python -c "from solcx import install_solc; install_solc('0.8.0')"
```

### Run Project

**Terminal 1 — Start blockchain (keep this open throughout):**
```bash
ganache --port 8545 --accounts 10 --deterministic
```

**Terminal 2 — Run each phase in order:**
```bash
# Phase 1: Baseline federated learning (no blockchain)
python fl_phase1.py

# Phase 2: FL + Blockchain logging
python bfl_phase2.py

# Phase 3: FL + Blockchain + DP + Attack Detection (main result)
python bfl_phase3.py

# Generate all result graphs
python generate_results.py
```

### Common Errors and Fixes

| Error | Fix |
|---|---|
| `ConnectionRefusedError` | Ganache not running — restart Terminal 1 |
| `ModuleNotFoundError: web3` | Run `pip install web3` |
| `ModuleNotFoundError: solcx` | Run `pip install py-solc-x` |
| `FileNotFoundError: ModelRegistry.sol` | Run from inside `bfl_project/` folder |
| `SolcError` | Run `python -c "from solcx import install_solc; install_solc('0.8.0')"` |
| TensorFlow GPU warning on Windows | Safe to ignore — CPU training works correctly |

---

## 9. Implementation — Phase by Phase

### Phase 1: Federated Learning Baseline

**File:** `fl_phase1.py`
**Goal:** Prove 5 clients can collaboratively train a digit classifier without sharing raw data.

**What happens each round:**
- Global model weights sent to all 5 clients
- Each client trains local CNN for 2 epochs (SGD optimizer)
- Clients return updated weights (not data) to server
- Server computes FedAvg → new global model
- Global model evaluated on 10,000 test images

**Result:**
```
Round 1  → Accuracy: 90.86%  Loss: 0.3296
Round 5  → Accuracy: 94.50%  Loss: 0.1909
Round 10 → Accuracy: 96.20%  Loss: 0.1292   ← final baseline
On-chain updates: 0
```

---

### Phase 2: Blockchain Integration

**Files:** `bfl_phase2.py`, `blockchain.py`, `ModelRegistry.sol`
**Goal:** Log every model update permanently on Ethereum — full tamper-proof audit trail.

**What is added:**
- `ModelRegistry.sol` compiled and deployed to Ganache on startup
- Each client computes SHA-256 hash of weights → submits to smart contract
- Blockchain records: client wallet address, hash, round number, timestamp
- Server reads verified update list from blockchain before running FedAvg

**Result:**
```
Contract deployed: 0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab

Round 1  → 5 updates logged on-chain | Accuracy: 90.86%
Round 5  → 5 updates logged on-chain | Accuracy: 94.50%
Round 10 → 5 updates logged on-chain | Accuracy: 96.20%
──────────────────────────────────────────────────────────
Total on-chain updates: 50  (5 clients × 10 rounds)
Final accuracy: 96.20%
```

---

### Phase 3: Privacy + Attack Detection (Main Result)

**File:** `bfl_phase3.py`
**Goal:** Detect and block a malicious client while protecting honest clients' privacy.

**Three mechanisms added:**

1. Differential Privacy — Gaussian noise added to all client weights (epsilon=100)
2. Cosine similarity trust scoring — each client scored every round
3. Automatic rejection — trust score below 0.85 triggers permanent block
4. Model reset — global model reinitialised when attacker is first caught

**Attack timeline:**
```
Round 1:
  Client 3 submits poison | similarity=-0.002 | trust=0.900 | ACCEPTED
  Model corrupted → Accuracy drops to 4.34%

Round 2:
  Client 3 submits poison | similarity=+0.001 | trust=0.810 | REJECTED
  Global model RESET to clean state
  Accepted clients: [1, 2, 4, 5]

Round 3:
  4 clean clients train on reset model
  Accuracy: 90.59%  ← recovered in exactly 1 round

Round 5:   Accuracy: 93.44%
Round 7:   Accuracy: 94.71%
Round 10:  Accuracy: 95.47%
Round 15:  Accuracy: 96.78%  ← final result

Attacker trust at round 15: 0.206  (permanently flagged)
Total on-chain updates: 61
```

---

## 10. Limitations & Future Work

| Paper Feature | Our Implementation | How to Improve |
|---|---|---|
| 100 clients | 5 simulated clients | Scale using TensorFlow Federated (TFF) |
| Homomorphic Encryption (HE) | SHA-256 hash only | Integrate Microsoft SEAL or PySyft HE |
| Zero-Knowledge Proofs (ZKP) | Simulated via trust scoring | Implement ZoKrates or snarkjs |
| CIFAR-10 + IoT-IDS datasets | MNIST only | Add CIFAR-10 with ResNet model |
| Cloud GPU deployment | Local CPU (Windows) | Deploy to AWS/GCP with GPU instances |
| Public Ethereum testnet | Local Ganache only | Deploy to Ethereum Sepolia testnet |
| Adaptive consensus threshold | Fixed at 0.85 | Dynamic threshold per round |
| Real-time inference API | Batch simulation | Add Flask or FastAPI prediction endpoint |

---

## 11. Team

| Name | USN | Email |
|---|---|---|
| Aman Nayan | ENG23CY0004 | amannayan1905@gmail.com |
| Kushal M G | ENG23CY0022 | eng23cy0022@dsu.edu.in |
| Likith M | ENG23CY0023 | eng23cy0022@dsu.edu.in |
| Madhukar N | ENG23CY0024 | madhukarnagaraju8050@gmail.com |
| Narasimha Murthy K | ENG23CY0026 | narasimhabtech27@gmail.com |

---

## Key Concepts Glossary

| Term | Plain English Meaning |
|---|---|
| **Federated Learning** | Training AI across multiple devices without sharing raw data |
| **FedAvg** | Server combines client updates using weighted average formula |
| **Blockchain** | Permanent tamper-proof public record of all transactions |
| **Smart Contract** | Self-executing code on Ethereum — runs automatically, no human needed |
| **Ganache** | Local Ethereum blockchain for development and testing |
| **Differential Privacy** | Adding calibrated random noise to protect data from reverse engineering |
| **Trust Score** | A running score tracking how honest a client has been over time |
| **Cosine Similarity** | Measures if two vectors point in the same direction (0=unrelated, 1=identical) |
| **Proof of Contribution** | Consensus mechanism that rewards honest model updates |
| **Data Poisoning Attack** | Malicious client sends random garbage weights to corrupt global model |
| **SHA-256** | Cryptographic hash function that converts weights into a unique fingerprint |
| **Web3.py** | Python library that communicates with the Ethereum blockchain |
| **MNIST** | Dataset of 70,000 handwritten digit images, standard ML benchmark |

---

##  Mentor

**Dr. Prajwalasimha S N**

Associate Professor, Department of Computer Science and Engineering (Cyber Security)
School of Engineering, Dayananda Sagar University, Bangalore — 562112

Email: prajwasimha.sn1@gmail.com

---


