# Blockchain-Enabled Federated Learning for Privacy‑Preserving AI

High‑fidelity FL + Blockchain implementation with poisoning‑attack detection and recovery.  
TTEH Research Lab – BFL Project  

Badges:  
_Implementation of the paper “Blockchain‑Enabled Federated Learning for Privacy‑Preserving AI” (ICISC‑2025). Refer to repository artifacts for full code and figures._

---

## Overview

**Problem:** Centralized ML requires aggregating all data at a single server, which violates privacy, breaks regulations (GDPR/HIPAA), and creates a single point of failure. Federated Learning (FL) keeps data local but is still vulnerable to **poisoning attacks**, malicious clients, and unverifiable model updates.

**Why traditional methods fail:**

- Central servers see all raw data and become attractive attack targets.
- Plain FL has no immutable audit trail; malicious clients can push poisoned updates without detection.
- No built‑in notion of **trust** or proof that an update is honest.

**Our solution (very clearly):** A three‑phase **Blockchain‑Enabled Federated Learning (BFL)** pipeline over MNIST:

- **Phase 1:** Standard FedAvg CNN across 5 clients (no blockchain, no attacks).
- **Phase 2:** Federated Learning + Ethereum private blockchain (Ganache) logging SHA‑256 hashes of client model weights via a Solidity smart contract.
- **Phase 3:** BFL + **Differential Privacy** + **Trust Scoring** with an explicit poisoning attacker client, automatic attack detection, model reset, and recovery.

The code generates all accuracy/trust/summary **figures** (`figure1_accuracy_trust.png`, `phase2_vs_phase3.png`, etc.) and comparison tables under the repository root.

**Key models used:**

- CNN classifier (TensorFlow/Keras) trained via FedAvg across 5 clients.
- Trust mechanism based on **cosine similarity** between global and local weight vectors.

**Final performance highlights:**

- FedAvg / BFL without attack: ≈ **96.20%** final accuracy.
- BFL with attacker + DP + trust scoring: recovers from **4.34%** (poisoned) to **96.78%** final accuracy after model reset.
- Blockchain logs **50** on‑chain updates in Phase 2 and **61** in Phase 3 (with 1 malicious update effectively rejected).

**Keywords:** AI · Cybersecurity · Federated Learning · Blockchain · Differential Privacy · Trust Scoring · Poisoning Attack Defense

---

## Table of Contents

1. Problem Statement  
2. Proposed Architecture  
3. How It Works  
4. Results & Metrics  
5. Code Architecture  
6. Core Modules — Deep Dive  
7. Setup & Usage  
8. Implementation Results  
9. Limitations  
10. Team  
11. Mentor  

---

## 1. Problem Statement

> “Secure, privacy‑preserving federated learning must resist malicious clients, provide verifiable model updates, and maintain high accuracy under poisoning attacks.”

Traditional FL/Bare ML fails because:

- **Privacy:** Centralized training requires raw data aggregation, breaking confidentiality and compliance.
- **Security:** No tamper‑proof record of which client sent which update; attackers can inject poisoned gradients.
- **Integrity & Trust:** Server cannot easily verify if a client behaved honestly without an external trust/consensus layer.

**Why the problem exists:**

- Networked/edge data (healthcare, IoT, cyber) is **distributed, high‑dimensional and evolving**, making central collection impractical.
- Real deployments need **auditability** (who updated what, when) and **resilience** against adversarial updates.

**What is needed:**

- A **reproducible FL pipeline** that keeps data local but logs model updates immutably.
- A **trust‑aware aggregation mechanism** that down‑weights or rejects suspicious clients.
- **Privacy mechanisms** (Differential Privacy, optional ZKP) to protect client updates.

---

## 2. Proposed Architecture

**Fig. 1 — BFL System Architecture with Attack Detection**

This system trains a CNN on MNIST via FL, logs each client’s model hash on a private Ethereum blockchain, and uses a **trust score** to detect and isolate a poisoning attacker.

| # | Module                     | Role                                             | Output                                   |
|---|----------------------------|--------------------------------------------------|------------------------------------------|
| 1 | Data Loader (MNIST)       | Load and normalize handwritten digits            | Local train/test tensors                 |
| 2 | FL Core (Phase 1)         | FedAvg training across 5 clients                 | Baseline global model                    |
| 3 | Blockchain Layer (Phase 2)| Deploy contract, hash weights, log updates       | On‑chain SHA‑256 hashes, round metadata  |
| 4 | DP + Trust (Phase 3)      | Add DP noise, compute similarity and trust       | Per‑client trust scores per round        |
| 5 | Attack Simulation         | Designate one client as malicious (poison weights)| Poisoned updates in early rounds        |
| 6 | Aggregation & Reset Engine| FedAvg on trusted clients, reset on detection    | Clean global model after reset           |
| 7 | Visualization & Reporting | Plot accuracy, trust decay, summary tables       | `*.png` graphs, printed metrics          |

---

## 3. How It Works

### Decision Logic / Workflow

High‑level logic for each **training round** in Phase 3:

1. Server broadcasts current **global weights** to all 5 clients.  
2. Each client trains locally on its MNIST shard.  
3. Malicious client replaces its weights with random **poisoned weights**.  
4. Each client adds **DP noise**, computes similarity to the previous global model, and gets an updated **trust score**.  
5. If a client’s trust score `< trust_threshold (0.85)` ⇒ **REJECT** its update.  
6. Accept only trusted updates, log their SHA‑256 hash to the blockchain, aggregate via FedAvg, and evaluate accuracy.  
7. On first detection of attacker ⇒ **reset global model** to clean initial weights, then continue training only with honest clients.

### Flow Diagram (text)

Raw MNIST data  
↓  
Client‑side training (local CNN)  
↓  
DP noise + hash computation  
↓  
Blockchain submit (`submitUpdate`)  
↓  
Trust scoring & rejection  
↓  
FedAvg aggregation on honest updates  
↓  
Updated global model + metrics  

### Core Equations

**FedAvg aggregation (server):**

\[
w^{(t+1)} = \sum_{i=1}^{N} p_i w_i^{(t+1)}
\]

Where \(p_i = \frac{m_i}{\sum_j m_j}\) is the data proportion and \(w_i\) are client weights.

**Differential Privacy noise (client, simplified):**

\[
w' = w + \mathcal{N}(0, \sigma^2),\quad \sigma = \frac{\text{sensitivity}}{\epsilon}
\]

With \(\epsilon = 100.0\) in the implementation.

**Trust score update (Proof‑of‑Contribution):**

\[
T_i^{(t+1)} = \alpha T_i^{(t)} + (1-\alpha) S_i^{(t)},\quad \alpha = 0.9
\]

\(S_i^{(t)}\) is cosine similarity between flattened global and local weights.

---

## 4. Results & Metrics

This repository contains the following key artifacts:

- `phase2_vs_phase3.png` — BFL under poisoning attack (detection & recovery).
- `figure1_accuracy_trust.png` — Accuracy vs round (Phase 2 vs Phase 3) + attacker trust score decay.
- `figure2_comparison.png` — Final accuracy comparison across all three phases.
- `figure3_blockchain_updates.png` — Number of verified on‑chain updates per phase.
- `summary_table.png` — Metrics table (accuracy, updates, privacy, defense).

### Phase‑wise Model Comparison

| Model / Phase              | Accuracy (final) | Notes                                     |
|----------------------------|------------------|-------------------------------------------|
| Phase 1 – FedAvg           | 0.9620           | Baseline FL without blockchain or attacks.|
| Phase 2 – BFL (no attack)  | 0.9620           | Same accuracy, 50 on‑chain updates.       |
| Phase 3 – BFL + DP + Trust | 0.9678           | Under attack; recovers and slightly improves.|

Exact round‑wise metrics are generated at runtime by `generate_results.py` and `final_report.py` and printed to console; re‑run these scripts to regenerate numbers and plots.

---

## 5. Code Architecture

```text
project/
├── code1.py                 # Phase 1 – FedAvg baseline
├── bfl_phase2.py            # Phase 2 – BFL (hashes on blockchain)
├── bfl_phase3.py            # Phase 3 – BFL + DP + Trust + attacker
├── blockchain.py            # Web3 + Solidity helpers (deploy, submit, query)
│
├── generate_results.py      # Accuracy/trust/blockchain plots + results table
├── final_report.py          # Alternate plots and summary table
│
├── figure1_accuracy_trust.jpg
├── phase2_vs_phase3.jpg
└── Group_2_Blockchain-Enabled_Federated_Learning_for_Privacy-Preserving_AI-2.pdf
```

The repository separates **training**, **blockchain interaction**, **attack/defense logic**, and **visualization**, mirroring the structure described in the reference paper.

---

## 6. Core Modules — Deep Dive

### Federated Learning Baseline (Phase 1)

**File path:** `code1.py`

- Loads MNIST from TensorFlow, normalizes, and splits equally across 5 clients.
- Defines a shared CNN model and implements **client‑side training** and **FedAvg aggregation**.
- Runs `num_rounds=10` and prints loss/accuracy each round; final accuracy ≈ 96.2%.

### Blockchain‑Secured FL (Phase 2)

**File paths:** `bfl_phase2.py`, `blockchain.py`

- Connects to local Ganache, compiles and deploys `ModelRegistry.sol` via `deploy_contract()`.
- Each client, after training, calls `submit_update(contract, client_idx, weights, round_number)`:
  - Flattens weight tensors, computes SHA‑256 hash, and sends it to the smart contract.
- Server may call `get_verified_clients(contract, round)` to retrieve on‑chain records for that round.

### DP + Trust Scoring + Attack Simulation (Phase 3)

**File path:** `bfl_phase3.py`

- `add_differential_privacy(weights, epsilon=100.0)` adds Gaussian noise before blockchain submission.
- `compute_trust_score(global_weights, local_weights, prev_score, alpha=0.9)` returns updated trust score and cosine similarity.
- `poison_weights(weights)` replaces attacker weights with random values to simulate poisoning.
- `run_bfl_phase3(...)` orchestrates rounds, detects the malicious client once trust falls below threshold, resets model, and continues with honest clients only.

### Visualization & Summary

**File paths:** `generate_results.py`, `final_report.py`

- Reconstruct Phase 2 & 3 accuracy curves from recorded values.
- Plot attacker trust score decay vs rejection threshold.
- Generate bar charts and tables summarizing:
  - Final accuracy per phase.
  - Total on‑chain updates per phase.
  - Attack detection and privacy mechanisms.

---

## 7. Setup & Usage

### Requirements

| Component   | Value                          |
|------------|---------------------------------|
| Python     | 3.8+                           |
| Libraries  | `tensorflow`, `numpy`, `matplotlib`, `web3`, `py-solc-x` |
| Blockchain | Ganache (local Ethereum)       |
| Dataset    | MNIST (auto‑downloaded by TensorFlow) |

### Installation

```bash
git clone <REPO_URL>
cd <REPO_ROOT>

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install tensorflow numpy matplotlib web3 py-solc-x
python -c "from solcx import install_solc; install_solc('0.8.0')"
```

Start Ganache (keep this terminal open):

```bash
ganache --port 8545 --accounts 10 --deterministic
```

### Run Project

**Baseline FedAvg (Phase 1):**

```bash
python code1.py
```

**BFL with Blockchain (Phase 2):**

```bash
python bfl_phase2.py
```

**BFL + DP + Trust + Attack (Phase 3):**

```bash
python bfl_phase3.py
```

**Generate plots and summary tables:**

```bash
python generate_results.py
python final_report.py
```

All images will be saved as `*.png` / `*.jpg` in the repo root for direct embedding in your GitHub README or report.

---

## 8. Implementation Results

**Training explanation:**  
Each phase performs multiple global rounds; clients train locally for a few epochs per round, and the server aggregates via FedAvg.

**Attack behavior:**  
In Phase 3, accuracy collapses to ~4.34% in Round 1 due to the poisoned attacker update, then recovers once the attacker is detected and the model is reset.

**Confusion / performance visuals:**

- `phase2_vs_phase3.png` clearly shows detection at Round 2 and recovery in accuracy.
- `figure1_accuracy_trust.png` overlays trust decay with the attack detection threshold.

**Blockchain statistics:**

- Phase 2: 50 on‑chain updates (5 clients × 10 rounds).
- Phase 3: 61 updates with the attacker effectively rejected by trust scoring.

_Image placeholders you can keep in README:_

- BFL under poisoning attack — `phase2_vs_phase3.jpg`  
- Accuracy vs trust decay — `figure1_accuracy_trust.jpg`

---

## 9. Limitations

| Paper Concept / Claim                    | Prototype Status                                        | Possible Fix / Extension                                  |
|-----------------------------------------|---------------------------------------------------------|-----------------------------------------------------------|
| Multiple datasets (MNIST, CIFAR‑10, IoT)| Prototype uses MNIST only                               | Add CIFAR‑10 and an IoT‑IDS dataset loaders              |
| ZKP‑based verification / PoC tokens     | Trust via cosine similarity; no real ZKP or token economy| Integrate ZKP libraries and ERC‑20 rewards               |
| Adaptive consensus (PoC, PoA, etc.)     | Uses simple Ganache network with basic verification     | Implement real PoC/PoA chain or IBFT/PoS testnet         |
| Full production metrics (latency, energy)| Current focus is accuracy and trust plots               | Instrument runtime, gas usage, latency stats             |

---

## 10. Team

| Name           | USN        | Email                              |
|----------------|------------|------------------------------------|
| Vaishnavi Shri | ENG23CY0046| vaishnavi18shri@gmail.com         |
| Atif Rahim     | ENG23CY0053| atif.rahim1104@gmail.com          |
| Tilak Moger    | ENG23CY0041| tilakkm20225@gmail.com            |
| Aman Kumar     | ENG23CY0051| amangupta6299@gmail.com           |

---

## 11. Mentor

**Dr. Prajwalasimha S N**  
Associate Professor, Department of Computer Science and Engineering (Cyber Security)  
School of Engineering, Dayananda Sagar University, Bengaluru, India  
Email: [prajwalasimha.sn1@gmail.com](mailto:prajwalasimha.sn1@gmail.com)
