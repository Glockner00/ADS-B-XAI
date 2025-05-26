import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import torch
from model import LossThresholdClassifier, RecurrentAutoencoder

# === Konfiguration ===
BACKGROUND_PATH = "data/background_data_subset.npy"
MODEL_PATH = "data/model.pth"
SEQ_LEN = 8
N_FEATURES = 1
THRESHOLD = 0.3
DEVICE = "cpu"

# === Klassificerare ===
model = RecurrentAutoencoder(seq_len=SEQ_LEN, n_features=N_FEATURES, embedding_dim=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
clf = LossThresholdClassifier(model, threshold=THRESHOLD, device=DEVICE, seq_len=SEQ_LEN, n_features=N_FEATURES)

# === Data ===
background_data = np.load(BACKGROUND_PATH)

def generate_all_shap_plots(shap_values, X, feature_names, label="anom", sample_index=0):
    os.makedirs("shap", exist_ok=True)
    shap_values = shap_values.squeeze(-1) if shap_values.ndim == 3 else shap_values
    df = pd.DataFrame(shap_values, columns=feature_names)

    # Summary plot (dot)
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="dot", show=False)
    plt.title(f"SHAP Summary Plot (dot) – {label}")
    plt.tight_layout()
    plt.savefig(f"shap/shap_summary_dot_{label}.png", dpi=300)
    plt.close()

    # Summary plot (bar)
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP Summary Plot (bar) – {label}")
    plt.tight_layout()
    plt.savefig(f"shap/shap_summary_bar_{label}.png", dpi=300)
    plt.close()

    # Dependence plot (velocity)
    shap.dependence_plot("velocity", shap_values, X, feature_names=feature_names, show=False)
    plt.title(f"SHAP Dependence Plot – velocity ({label})")
    plt.tight_layout()
    plt.savefig(f"shap/shap_dependence_velocity_{label}.png", dpi=300)
    plt.close()

    # Violin plot
    df_melt = df.melt(var_name="Feature", value_name="SHAP value")
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_melt, x="Feature", y="SHAP value")
    plt.title(f"SHAP Violin Plot – {label}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"shap/shap_violin_{label}.png", dpi=300)
    plt.close()

    # Histogram grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f"SHAP Histograms – {label}", fontsize=18)
    for ax, feat in zip(axes.flatten(), feature_names):
        sns.histplot(df[feat], kde=True, ax=ax)
        ax.set_title(feat)
        ax.set_xlabel("SHAP value")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"shap/shap_histograms_combined_{label}.png", dpi=300)
    plt.close()

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="coolwarm", center=0)
    plt.title(f"SHAP Heatmap – {label}")
    plt.xlabel("Feature")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.savefig(f"shap/shap_heatmap_{label}.png", dpi=300)
    plt.close()

    # === Force plot (HTML) ===
    expected_value = clf.predict_proba(background_data[:100]).mean()
    explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=expected_value,
        data=X[sample_index],
        feature_names=feature_names
    )
    shap.save_html(f"shap/shap_force_{label}_{sample_index}.html", shap.plots.force(explanation))

    # === Waterfall plot ===
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f"shap/shap_waterfall_{label}_{sample_index}.png", dpi=300)
    plt.close()

    # === Decision plot (global) ===
    shap.decision_plot(expected_value, shap_values[:100], feature_names=feature_names, show=False)
    plt.title(f"SHAP Decision Plot – {label}")
    plt.tight_layout()
    plt.savefig(f"shap/shap_decision_{label}.png", dpi=300)
    plt.close()

# === Kör ===
shap_values_anom = np.load("data/shap_values_anom.npy", allow_pickle=True)
X_anom = np.load("data/test_data_anom_subset.npy")
if isinstance(shap_values_anom, list):
    shap_values_anom = shap_values_anom[0]

feature_names = ['long', 'lat', 'baro_altitude', 'velocity', 'true_track',
                 'vertical_rate', 'geo_altitude', 'squawk']

generate_all_shap_plots(shap_values_anom, X_anom, feature_names, label="anom", sample_index=0)
