import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from blockchain import deploy_contract, submit_update, get_verified_clients

# ── Paste your Phase 1 functions here ──────────────────────────────────────
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

# ── Phase 2: FL + Blockchain ────────────────────────────────────────────────
def run_bfl(num_rounds=10, num_clients=5):
    client_datasets, (x_test, y_test) = load_and_split_data(num_clients)
    global_model   = create_model()
    global_weights = global_model.get_weights()

    # Deploy smart contract once
    contract = deploy_contract()
    print("\nBlockchain ready. Starting BFL...\n")

    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}/{num_rounds}")
        client_updates = []

        for i, dataset in enumerate(client_datasets):
            weights, n_samples = client_train(global_weights, dataset)
            submit_update(contract, i, weights, round_num)   # log on blockchain
            client_updates.append((weights, n_samples))

        verified = get_verified_clients(contract, round_num)
        print(f"  {len(verified)} verified updates on blockchain")

        global_weights = federated_average(client_updates)
        global_model.set_weights(global_weights)

        loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"  Accuracy: {acc:.4f} | Loss: {loss:.4f}\n")

    total = contract.functions.getTotalUpdates().call()
    print(f"Done! Total updates recorded on blockchain: {total}")

if __name__ == "__main__":
    run_bfl()