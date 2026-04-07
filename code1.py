import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ── 1. Create a simple CNN model (shared architecture) ──────────────────────
def create_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── 2. Load and split MNIST across 5 clients ───────────────────────────────
def load_and_split_data(num_clients=5):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.0   # normalize + add channel dim
    x_test  = x_test[..., np.newaxis]  / 255.0

    # Shuffle and split training data equally among clients
    indices = np.random.permutation(len(x_train))
    client_data = []
    split_size = len(x_train) // num_clients
    for i in range(num_clients):
        idx = indices[i * split_size : (i + 1) * split_size]
        client_data.append((x_train[idx], y_train[idx]))

    return client_data, (x_test, y_test)

# ── 3. Client: train locally and return updated weights ────────────────────
def client_train(global_weights, client_dataset, epochs=2):
    x, y = client_dataset
    local_model = create_model()
    local_model.set_weights(global_weights)       # start from global model
    local_model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
    return local_model.get_weights(), len(x)      # return weights + data size

# ── 4. Server: FedAvg aggregation ──────────────────────────────────────────
def federated_average(client_updates):
    """
    client_updates: list of (weights, num_samples) tuples
    Implements: w_global = sum(p_i * w_i)  where p_i = m_i / sum(m_i)
    """
    total_samples = sum(n for _, n in client_updates)
    # Weighted average of each layer's weights
    avg_weights = []
    for layer_idx in range(len(client_updates[0][0])):
        layer_avg = np.sum([
            (n / total_samples) * weights[layer_idx]
            for weights, n in client_updates
        ], axis=0)
        avg_weights.append(layer_avg)
    return avg_weights

# ── 5. Main FL training loop ───────────────────────────────────────────────
def run_federated_learning(num_rounds=10, num_clients=5):
    client_datasets, (x_test, y_test) = load_and_split_data(num_clients)

    # Initialize global model
    global_model = create_model()
    global_weights = global_model.get_weights()

    print(f"Starting Federated Learning — {num_clients} clients, {num_rounds} rounds\n")

    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}/{num_rounds}")

        # Each client trains locally
        client_updates = []
        for i, dataset in enumerate(client_datasets):
            weights, n_samples = client_train(global_weights, dataset, epochs=2)
            client_updates.append((weights, n_samples))
            print(f"  Client {i+1} trained on {n_samples} samples")

        # Server aggregates using FedAvg
        global_weights = federated_average(client_updates)
        global_model.set_weights(global_weights)

        # Evaluate global model on test set
        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"  Global model — Loss: {loss:.4f} | Accuracy: {accuracy:.4f}\n")

    print("Training complete!")
    return global_model

# ── Run it ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    final_model = run_federated_learning(num_rounds=10, num_clients=5)