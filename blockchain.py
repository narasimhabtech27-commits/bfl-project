import hashlib
import numpy as np
from web3 import Web3
from solcx import compile_source, install_solc

install_solc('0.8.0')   # install Solidity compiler once

# ── Connect to your local Ganache blockchain ────────────────────────────────
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
assert w3.is_connected(), "Ganache not running! Start it first."

accounts = w3.eth.accounts
SERVER_ACCOUNT  = accounts[0]    # account 0 = federated server
CLIENT_ACCOUNTS = accounts[1:6]  # accounts 1–5 = 5 clients

# ── Compile and deploy the smart contract ──────────────────────────────────
def deploy_contract():
    with open('ModelRegistry.sol', 'r') as f:
        source = f.read()

    compiled = compile_source(source, output_values=['abi', 'bin'],
                               solc_version='0.8.0')
    contract_id  = list(compiled.keys())[0]
    contract_abi = compiled[contract_id]['abi']
    contract_bin = compiled[contract_id]['bin']

    # Deploy to Ganache
    Contract = w3.eth.contract(abi=contract_abi, bytecode=contract_bin)
    tx_hash  = Contract.constructor().transact({'from': SERVER_ACCOUNT})
    receipt  = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"Contract deployed at: {receipt.contractAddress}")
    return w3.eth.contract(address=receipt.contractAddress, abi=contract_abi)

# ── Hash a client's model weights (converts weights → a fingerprint) ────────
def hash_weights(weights):
    # Flatten all weight arrays into one byte string, then SHA-256 hash it
    flat = np.concatenate([w.flatten() for w in weights])
    raw  = flat.tobytes()
    return hashlib.sha256(raw).digest()   # returns 32 bytes

# ── Client submits their update hash to the blockchain ─────────────────────
def submit_update(contract, client_idx, weights, round_number):
    weight_hash  = hash_weights(weights)
    hash_bytes32 = weight_hash.ljust(32, b'\x00')[:32]  # ensure exactly 32 bytes

    tx = contract.functions.submitUpdate(
        hash_bytes32,
        round_number
    ).transact({'from': CLIENT_ACCOUNTS[client_idx]})

    w3.eth.wait_for_transaction_receipt(tx)
    print(f"  Client {client_idx+1} logged update on blockchain (round {round_number})")

# ── Server fetches all verified updates for the round ──────────────────────
def get_verified_clients(contract, round_number):
    clients, hashes = contract.functions.getVerifiedUpdates(round_number).call()
    print(f"  {len(clients)} verified updates found on-chain for round {round_number}")
    return clients