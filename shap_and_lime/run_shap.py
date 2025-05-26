# Run : pip install -r requirements.txt or pip3 install -r requirements.txt
# When all batches are collected run combine.py

import torch
import numpy as np
import shap
import os
import time
from model import RecurrentAutoencoder, LossThresholdClassifier

DEVICE = torch.device("cpu")
MODEL_PATH = "C:/Users/olleh/LIU/kandidat/ADS-B-XAI/shap_and_lime/data_real/model.pth"
BACKGROUND_PATH = "C:/Users/olleh/LIU/kandidat/ADS-B-XAI/shap_and_lime/data_real/background_data_subset.npy"
TEST_NORM_PATH = "C:/Users/olleh/LIU/kandidat/ADS-B-XAI/shap_and_lime/data_real/test_data_norm_subset.npy"
TEST_ANOM_PATH = "C:/Users/olleh/LIU/kandidat/ADS-B-XAI/shap_and_lime/data_real/test_data_anom_subset.npy"

SEQ_LEN = 7
N_FEATURES = 1
THRESHOLD = 0.7
NSAMPLES = 300
BATCH_SIZE = 50
OUTPUT_DIR = "shap_batches"

model = RecurrentAutoencoder(seq_len=SEQ_LEN, n_features=N_FEATURES, embedding_dim=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


background_data = np.load(BACKGROUND_PATH)
test_data_norm = np.load(TEST_NORM_PATH)
test_data_anom = np.load(TEST_ANOM_PATH)

test_data_norm = np.load(TEST_NORM_PATH)
test_data_anom = np.load(TEST_ANOM_PATH)

np.random.seed(42)  # för reproducerbarhet
test_data_norm = test_data_norm[np.random.choice(len(test_data_norm), 500, replace=False)]
test_data_anom = test_data_anom[np.random.choice(len(test_data_anom), 500, replace=False)]

clf = LossThresholdClassifier(
    model, threshold=THRESHOLD, device=DEVICE, seq_len=SEQ_LEN, n_features=N_FEATURES
)

explainer = shap.KernelExplainer(clf.predict_proba, background_data)


def compute_shap_in_batches(explainer, data, prefix):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(0, len(data), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE
        batch = data[i : i + BATCH_SIZE]
        out_path = os.path.join(OUTPUT_DIR, f"{prefix}_batch_{batch_idx:03d}.npy")

        if os.path.exists(out_path):
            print(f"Skipping {out_path} (already saved)")
            continue

        print(f"Shap for batch {i}-{i + len(batch) - 1}")
        start = time.time()
        try:
            shap_values = explainer.shap_values(batch, nsamples=NSAMPLES)
            np.save(out_path, shap_values)
            print(f"Saved: {out_path} ({time.time() - start:.2f}s)")
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            break


def load_all_batches(prefix):
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix)])
    loaded = [np.load(os.path.join(OUTPUT_DIR, f), allow_pickle=True) for f in files]
    return np.concatenate(loaded, axis=1 if isinstance(loaded[0], list) else 0)


compute_shap_in_batches(explainer, test_data_anom, prefix="anom")


shap_values_anom = load_all_batches("anom")
np.save("shap_values_anom_new.npy", shap_values_anom)
print(f"Saved at shap_values_anom_new.npy ({shap_values_anom.shape})")

# compute_shap_in_batches(explainer, test_data_norm, prefix="norm")
# shap_values_norm = load_all_batches("norm")
# np.save("shap_values_norm.npy", shap_values_norm)
# print(f"Saved at shap_values_norm.npy ({shap_values_norm.shape})")
