import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from model import RecurrentAutoencoder
import os

# === Konfiguration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "data/model.pth"
SEQ_LEN = 8
N_FEATURES = 1

feature_names = ['long', 'lat', 'baro_altitude', 'velocity', 'true_track',
                 'vertical_rate', 'geo_altitude', 'squawk']

# === Modell ===
model = RecurrentAutoencoder(seq_len=SEQ_LEN, n_features=N_FEATURES, embedding_dim=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Predict wrapper ===
def predict_fn(input_flat):
    input_seq = input_flat.reshape((-1, SEQ_LEN, N_FEATURES))
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()
    error = np.mean(np.abs(output - input_seq), axis=(1, 2))  # L1-fel per sekvens
    return error.reshape(-1, 1)

# === Kombinerad histogramplot för alla features ===
def save_combined_histograms(df, order, label):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f"Histogram of LIME Importances ({label})", fontsize=18)

    for ax, feat in zip(axes.flatten(), order):
        sns.histplot(df[df['Feature'] == feat]['Importance'], kde=True, ax=ax)
        ax.set_title(feat)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Count")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"lime/histograms_combined_{label}.png", dpi=300)
    plt.close()

# === Huvudfunktion för analys och plottning ===
def run_lime_analysis(data_path, label):
    # Ladda och forma data
    X_raw = np.load(data_path).astype(np.float32)
    X_flat = X_raw.reshape((X_raw.shape[0], -1))

    # LIME-explainer
    explainer = LimeTabularExplainer(
        training_data=X_flat,
        feature_names=feature_names,
        mode='regression'
    )

    # Generera förklaringar
    num_samples = 1500
    explanations = []
    for i in range(num_samples):
        exp = explainer.explain_instance(
            X_flat[i],
            predict_fn,
            num_features=len(feature_names)
        )
        explanations.append(dict(exp.as_list()))

    # DataFrame och filtrering
    df = pd.DataFrame(explanations).melt(var_name='Feature_raw', value_name='Importance')
    valid_features = set(feature_names)
    df['Feature'] = df['Feature_raw'].apply(lambda x: next((f for f in valid_features if x.startswith(f)), "Other"))
    df = df[df['Feature'] != "Other"]
    order = sorted(df['Feature'].unique())

    # === Skapa och spara varje plottyp ===
    os.makedirs("lime", exist_ok=True)

    def save_plot(plot_func, filename, **kwargs):
        plt.figure(figsize=(12, 6))
        plot_func(data=df, x='Feature', y='Importance', order=order, **kwargs)
        plt.title(f"{filename.replace('_', ' ').title()} ({label})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"lime/{filename}_{label}.png", dpi=300)
        plt.close()

    save_plot(sns.violinplot, "violinplot")
    save_plot(sns.boxplot, "boxplot")
    save_plot(sns.swarmplot, "swarmplot", size=4)
    save_plot(sns.barplot, "barplot", errorbar="sd")
    save_plot(sns.stripplot, "stripplot", jitter=True, size=3)

    # Histogram per feature
    for feat in order:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[df['Feature'] == feat]['Importance'], kde=True)
        plt.title(f"Histogram of '{feat}' LIME Importances ({label})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"lime/histogram_{feat}_{label}.png", dpi=300)
        plt.close()

    # Kombinerad histogramplot
    save_combined_histograms(df, order, label)

# === Kör för båda dataseten ===
run_lime_analysis("data/test_data_norm_subset.npy", "norm")
run_lime_analysis("data/test_data_anom_subset.npy", "anom")