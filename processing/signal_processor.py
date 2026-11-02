import sys
import io

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, medfilt, welch
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis, iqr
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

NUM_SUB     = 64
WINDOW_SIZE = 30    # 3 seconds at ~10Hz
HOP_SIZE    = 5     # Fine-grained sliding step (~83.3% overlap) -> Yields 115 samples per class!
ROOT_DIR    = Path(__file__).resolve().parents[1]
RAW_DIR     = ROOT_DIR / "data" / "raw"
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# ============================================
# 1. تحميل وتجميع كل ملفات CSV
# ============================================
def load_all_files(pattern="raw_label*.csv"):
    """Load raw captures from the repository, independent of the shell CWD."""
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["capture_id"] = Path(f).stem
        print(f"  [+] Loaded: {os.path.basename(f)} -> {len(df)} samples")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n[+] Total raw packets aggregated: {len(combined)}")
    return combined

# ============================================
# 2. فلترة متقدمة للإشارة (Hampel/Median + Butterworth LPF)
# ============================================
def advanced_denoise(signal, cutoff=0.15, order=4):
    """
    مرحلتان لتنقية الإشارة:
    1. Median Filter لإزالة القمم الشاذة (Impulse Spikes)
    2. Zero-phase Butterworth Low-Pass Filter لتنعيم الضوضاء عالية التردد
    """
    # 1. Median filter (kernel size 3)
    cleaned = medfilt(signal, kernel_size=3)
    # 2. Butterworth filter
    b, a = butter(order, cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, cleaned)
    return filtered

def filter_csi_matrix(csi_matrix: np.ndarray) -> np.ndarray:
    """
    تطبيق الفلترة المزدوجة على كل subcarrier
    """
    filtered = np.zeros_like(csi_matrix)
    for i in range(csi_matrix.shape[1]):
        col = csi_matrix[:, i].astype(float)
        if np.std(col) > 1e-4:
            filtered[:, i] = advanced_denoise(col)
        else:
            filtered[:, i] = col
    return filtered

# ============================================
# 3. استخراج 35+ ميزة فائقة الجودة من كل نافذة
# ============================================
def extract_advanced_features(window: np.ndarray, rssi_window: np.ndarray) -> dict:
    """
    window shape: (WINDOW_SIZE, 64)
    استخراج ميزات تشمل:
    - Time-domain statistics
    - Subcarrier spatial band energies & ratios
    - Inter-subcarrier cross-correlation
    - Frequency domain & Doppler metrics (FFT & Spectral Centroid / Spread / Entropy)
    - Kinematics (First and Second derivative for motion speed)
    - PCA spatial variance ratios
    - RSSI statistics
    """
    f = {}
    mean_per_sub = np.mean(window, axis=0) # 64
    std_per_sub  = np.std(window, axis=0)
    flat = window.flatten()

    # ---- 1. Time-Domain Features ----
    f['mean_amp']       = float(np.mean(window))
    f['std_amp']        = float(np.std(window))
    f['max_amp']        = float(np.max(window))
    f['min_amp']        = float(np.min(window))
    f['range_amp']      = f['max_amp'] - f['min_amp']
    f['variance_amp']   = float(np.var(window))
    f['rms_amp']        = float(np.sqrt(np.mean(window ** 2)))
    f['crest_factor']   = float(f['max_amp'] / (f['rms_amp'] + 1e-6))
    f['energy_amp']     = float(np.sum(window ** 2))

    # ---- 2. Statistical Shape Features ----
    f['skewness']       = float(skew(flat))
    f['kurtosis']       = float(kurtosis(flat))
    f['iqr_amp']        = float(iqr(flat))

    # ---- 3. Spatial Band Features & Ratios ----
    f['low_band']       = float(np.mean(mean_per_sub[:21]))    # Subcarriers 0–20
    f['mid_band']       = float(np.mean(mean_per_sub[21:43]))  # Subcarriers 21–42
    f['high_band']      = float(np.mean(mean_per_sub[43:]))    # Subcarriers 43–63
    f['band_ratio_lh']  = float(f['low_band'] / (f['high_band'] + 1e-6))
    f['band_ratio_mh']  = float(f['mid_band'] / (f['high_band'] + 1e-6))

    # ---- 4. Inter-Subcarrier Correlation ----
    # الإنسان يغيّر مصفوفة الارتباط التبادلي بين الترددات الفرعية
    try:
        corr_matrix = np.corrcoef(window.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        triu_indices = np.triu_indices(NUM_SUB, k=1)
        f['subcarrier_corr_mean'] = float(np.mean(corr_matrix[triu_indices]))
        f['subcarrier_corr_std']  = float(np.std(corr_matrix[triu_indices]))
    except Exception:
        f['subcarrier_corr_mean'] = 0.0
        f['subcarrier_corr_std']  = 0.0

    # ---- 5. Kinematics & Temporal Variation (Motion Tracking) ----
    temporal_var = np.var(window, axis=0)
    f['temporal_var_mean'] = float(np.mean(temporal_var))
    f['temporal_var_max']  = float(np.max(temporal_var))
    
    # السرعة والتسارع (First & Second Differences over time)
    diff1 = np.diff(window, axis=0) # (WINDOW_SIZE-1, 64)
    f['velocity_mean']     = float(np.mean(np.abs(diff1)))
    f['velocity_max']      = float(np.max(np.abs(diff1)))
    diff2 = np.diff(diff1, axis=0)
    f['accel_mean']        = float(np.mean(np.abs(diff2)))

    # ---- 6. Frequency Domain & Doppler Spectrum (FFT) ----
    packet_mean_signal = np.mean(window, axis=1) # (WINDOW_SIZE,)
    fft_raw = np.abs(fft(packet_mean_signal))
    fft_vals = fft_raw[:WINDOW_SIZE // 2]
    fft_sum = np.sum(fft_vals) + 1e-6
    fft_norm = fft_vals / fft_sum

    f['fft_peak']    = float(np.max(fft_vals[1:]) if len(fft_vals) > 1 else fft_vals[0])
    f['fft_energy']  = float(np.sum(fft_vals ** 2))
    f['fft_entropy'] = float(-np.sum(fft_norm * np.log(fft_norm + 1e-10)))

    # Spectral Centroid & Spectral Spread
    freqs = np.linspace(0, 1, len(fft_vals))
    spectral_centroid = float(np.sum(freqs * fft_norm))
    f['spectral_centroid'] = spectral_centroid
    f['spectral_spread']   = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_norm)))

    # ---- 7. PCA Dimensionality Reduction over window ----
    try:
        pca = PCA(n_components=3)
        pca.fit(window)
        f['pca_evr_1'] = float(pca.explained_variance_ratio_[0])
        f['pca_evr_2'] = float(pca.explained_variance_ratio_[1])
        f['pca_evr_3'] = float(pca.explained_variance_ratio_[2])
    except Exception:
        f['pca_evr_1'] = 0.0
        f['pca_evr_2'] = 0.0
        f['pca_evr_3'] = 0.0

    # ---- 8. RSSI Metrics ----
    f['rssi_mean']  = float(np.mean(rssi_window))
    f['rssi_std']   = float(np.std(rssi_window))
    f['rssi_range'] = float(np.max(rssi_window) - np.min(rssi_window))

    return f

# ============================================
# 4. المعالجة الكاملة وتوليد الملفات
# ============================================
def process_all_advanced(pattern="raw_label*.csv"):
    print("=" * 60)
    print("  🚀 Wi-Fi CSI Ultra-Quality Signal Processor & Feature Pipeline")
    print("=" * 60)

    df = load_all_files(pattern)
    amp_cols = [f'amp_{i}' for i in range(NUM_SUB)]
    all_features = []

    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        count = 0
        print(f"\n[*] Processing scenario label {label} ({len(label_df)} raw packets)...")
        # Filter and window each physical capture separately. Concatenating two
        # sessions would create artificial windows across their boundary.
        for capture_id, capture_df in label_df.groupby("capture_id", sort=True):
            csi_matrix = capture_df[amp_cols].values.astype(float)
            rssi = capture_df['rssi'].values.astype(float)
            csi_filtered = filter_csi_matrix(csi_matrix)
            for start in range(0, len(csi_filtered) - WINDOW_SIZE, HOP_SIZE):
                end = start + WINDOW_SIZE
                feats = extract_advanced_features(csi_filtered[start:end], rssi[start:end])
                feats['label'] = int(label)
                feats['capture_id'] = capture_id
                all_features.append(feats)
                count += 1
        print(f"    -> Successfully extracted {count} high-resolution feature vectors")

    features_df = pd.DataFrame(all_features)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = DATA_DIR / "csi_features.csv"
    features_df.to_csv(feature_path, index=False)

    print("\n" + "=" * 60)
    print(f"✅ Feature Extraction Complete!")
    print(f"Total Feature Vectors: {len(features_df)}")
    print(f"Features per Vector:   {len(features_df.columns) - 1} (Target: label)")
    print(f"\nBalanced Class Distribution:\n{features_df['label'].value_counts().sort_index()}")
    print(f"💾 Saved Dataset: {feature_path}")
    print("=" * 60)

    # توليد الرسوم التوضيحية الأكاديمية
    generate_high_quality_visualizations(df, features_df, amp_cols)

    # تقييم نماذج ML الشاملة
    run_comprehensive_benchmark(features_df)

    return features_df

# ============================================
# 5. توليد حزمة الرسومات البيانية الاحترافية
# ============================================
def generate_high_quality_visualizations(raw_df, features_df, amp_cols):
    print("\n🎨 Generating Academic Quality Visualizations...")
    
    # 1. Filter Comparison Plot
    sample = raw_df[amp_cols].values[:250, 0].astype(float)
    filtered = advanced_denoise(sample)
    
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(sample, color='#d9534f', alpha=0.8, linewidth=1.5, label='Raw CSI Amplitude (Noisy)')
    axes[0].set_title('Raw CSI Amplitude with Impulse Noise & High-Frequency Jitter', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    
    axes[1].plot(filtered, color='#0275d8', linewidth=2.0, label='Denoised CSI (Dual Median + Butterworth LPF)')
    axes[1].set_title('Denoised CSI Signal (Clean & Preserving Human Motion Signature)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Packet Index (~10 Hz sampling)')
    axes[1].set_ylabel('Filtered Amplitude')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'filter_comparison.png', dpi=200)
    plt.close()
    print("  [+] Saved: filter_comparison.png")

    # 2. PCA 2D/3D Cluster Separation Plot
    X = features_df.drop(columns=['label', 'capture_id'])
    y = features_df['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    labels_names = {
        0: 'Empty Room (0)',
        1: 'Person Center (1)',
        2: 'Person Right (2)',
        3: 'Person Left (3)',
        4: 'Person Moving (4)'
    }
    colors = ['#6c757d', '#0275d8', '#5cb85c', '#f0ad4e', '#d9534f']
    
    plt.figure(figsize=(10, 7))
    for lbl in range(5):
        idx = (y == lbl)
        plt.scatter(
            X_pca[idx, 0], X_pca[idx, 1],
            c=colors[lbl], label=labels_names[lbl],
            alpha=0.85, edgecolors='black', linewidth=0.5, s=60
        )
    plt.title(f'PCA Feature Space Clustering (EVR: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13, fontweight='bold')
    plt.xlabel('Principal Component 1 (Primary Spatial/Motion Variance)')
    plt.ylabel('Principal Component 2 (Secondary Spectral Variance)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'pca_clusters.png', dpi=200)
    plt.close()
    print("  [+] Saved: pca_clusters.png")

    # 3. Correlation Heatmap of Selected High-Value Features
    selected_features = [
        'mean_amp', 'std_amp', 'variance_amp', 'low_band', 'high_band',
        'band_ratio_lh', 'subcarrier_corr_mean', 'temporal_var_mean',
        'velocity_mean', 'fft_peak', 'fft_entropy', 'spectral_centroid', 'rssi_mean'
    ]
    plt.figure(figsize=(11, 9))
    corr = features_df[selected_features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
    plt.title('High-Value Feature Correlation Matrix', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_correlation_heatmap.png', dpi=200)
    plt.close()
    print("  [+] Saved: feature_correlation_heatmap.png")

# ============================================
# 6. التقييم الشامل للنماذج (ML Benchmark & Confusion Matrix)
# ============================================
def run_comprehensive_benchmark(features_df):
    print("\n🧠 Running Comprehensive Multi-Model Benchmark...")
    X = features_df.drop(columns=['label', 'capture_id'])
    y = features_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=2.0, probability=True, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "Neural Net (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n" + "-" * 55)
    print(f"{'Model Architecture':<25} | {'5-Fold CV Accuracy':<25}")
    print("-" * 55)

    best_model = None
    best_acc = -1

    for name, model in models.items():
        X_tr = X_train if "Forest" in name or "Boosting" in name else X_train_scaled
        scores = cross_val_score(model, X_tr, y_train, cv=cv)
        mean_acc = scores.mean() * 100
        std_acc  = scores.std() * 100
        print(f"{name:<25} | {mean_acc:6.2f}% (+/- {std_acc:4.2f}%)")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_model = (name, model)

    print("-" * 55)

    # Fit best model and plot Confusion Matrix
    b_name, b_clf = best_model
    X_tr = X_train if "Forest" in b_name or "Boosting" in b_name else X_train_scaled
    X_te = X_test if "Forest" in b_name or "Boosting" in b_name else X_test_scaled
    b_clf.fit(X_tr, y_train)
    y_pred = b_clf.predict(X_te)
    test_acc = accuracy_score(y_test, y_pred) * 100

    print(f"\n[+] Champion Model: {b_name} with Test Accuracy: {test_acc:.2f}%")
    print("\nDetailed Test Classification Report:")
    target_names = ['Empty Room', 'Person Center', 'Person Right', 'Person Left', 'Person Moving']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix — {b_name} (Test Accuracy: {test_acc:.1f}%)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix_benchmark.png', dpi=200)
    plt.close()
    print("  [+] Saved: confusion_matrix_benchmark.png")

# ============================================
# 7. التنفيذ الرئيسي
# ============================================
if __name__ == "__main__":
    process_all_advanced("raw_label*.csv")
