import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from blockchain import deploy_contract, submit_update, get_verified_clients

def add_differential_privacy(weights, epsilon=100.0, sensitivity=1.0):
    noisy_weights = []
    noise_scale = sensitivity / epsilon
    for w in weights:
        noise = np.random.normal(0, noise_scale, w.shape)
        noisy_weights.append(w + noise)
    return noisy_weights

def compute_trust_score(global_weights, local_weights, prev_score, alpha=0.9):
    global_flat = np.concatenate([w.flatten() for w in global_weights])
    local_flat  = np.concatenate([w.flatten() for w in local_weights])
    dot  = np.dot(global_flat, local_flat)
    norm = np.linalg.norm(global_flat) * np.linalg.norm(local_flat)
    sim  = dot / (norm + 1e-8)
    new_score = alpha * prev_score + (1 - alpha) * sim
    return float(new_score), float(sim)

def poison_weights(weights):
    return [np.random.randn(*w.shape) * 10 for w in weights]

def create_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_and_split_data(num_clients=5):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.0
    x_test  = x_test[..., np.newaxis]  / 255.0
    indices    = np.random.permutation(len(x_train))
    split_size = len(x_train) // num_clients
    client_data = []
    for i in range(num_clients):
        idx = indices[i * split_size:(i+1) * split_size]
        client_data.append((x_train[idx], y_train[idx]))
    return client_data, (x_test, y_test)

def client_train(global_weights, client_dataset, epochs=2):
    x, y = client_dataset
    local_model = create_model()
    local_model.set_weights(global_weights)
    local_model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
    return local_model.get_weights(), len(x)

def federated_average(client_updates):
    total_samples = sum(n for _, n in client_updates)
    avg_weights = []
    for layer_idx in range(len(client_updates[0][0])):
        layer_avg = np.sum([
            (n / total_samples) * weights[layer_idx]
            for weights, n in client_updates
        ], axis=0)
        avg_weights.append(layer_avg)
    return avg_weights

def run_bfl_phase3(num_rounds=15, num_clients=5,
                   trust_threshold=0.85,
                   epsilon=100.0,
                   malicious_client=2):

    client_datasets, (x_test, y_test) = load_and_split_data(num_clients)

    # Start with a CLEAN model (no attacker yet)
    global_model   = create_model()
    global_weights = global_model.get_weights()

    contract      = deploy_contract()
    trust_scores  = [1.0] * num_clients
    results       = []
    model_reset   = False   # track if we already reset after attacker caught

    print("\nPhase 3 BFL — DP + Trust Scoring + Attack Detection\n")

    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}/{num_rounds}")
        client_updates   = []
        accepted_clients = []
        prev_global      = [w.copy() for w in global_weights]
        attacker_caught_this_round = False

        for i, dataset in enumerate(client_datasets):
            weights, n_samples = client_train(global_weights, dataset)

            if i == malicious_client:
                weights = poison_weights(weights)
                print(f"  Client {i+1} [ATTACKER] sending poisoned weights!")

            private_weights = add_differential_privacy(weights, epsilon=epsilon)

            trust_scores[i], sim = compute_trust_score(
                prev_global, private_weights, trust_scores[i]
            )

            status = "ACCEPTED" if trust_scores[i] >= trust_threshold else "REJECTED"
            print(f"  Client {i+1} — similarity: {sim:.3f} | "
                  f"trust: {trust_scores[i]:.3f} | {status}")

            if trust_scores[i] < trust_threshold:
                if i == malicious_client:
                    attacker_caught_this_round = True
                continue

            submit_update(contract, i, private_weights, round_num)
            client_updates.append((private_weights, n_samples))
            accepted_clients.append(i + 1)

        # KEY FIX: reset global model first time attacker is caught
        if attacker_caught_this_round and not model_reset:
            print("  Attacker caught! Resetting global model to clean state...")
            global_model   = create_model()
            global_weights = global_model.get_weights()
            model_reset    = True
            print(f"  Accepted: {accepted_clients} (training on clean model)\n")
            continue   # skip this round's aggregation, start fresh next round

        print(f"  Accepted: {accepted_clients}")

        if len(client_updates) == 0:
            print("  No valid updates — skipping\n")
            continue

        global_weights = federated_average(client_updates)
        global_model.set_weights(global_weights)

        loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"  Accuracy: {acc:.4f} | Loss: {loss:.4f}\n")
        results.append((round_num, acc, len(accepted_clients)))

    print("=" * 50)
    print("PHASE 3 COMPLETE")
    print("=" * 50)
    print(f"Final accuracy : {results[-1][1]:.4f}")
    print(f"On-chain updates: {contract.functions.getTotalUpdates().call()}")
    return results

if __name__ == "__main__":
    run_bfl_phase3(
        num_rounds=15,
        num_clients=5,
        trust_threshold=0.85,
        epsilon=100.0,
        malicious_client=2
    )