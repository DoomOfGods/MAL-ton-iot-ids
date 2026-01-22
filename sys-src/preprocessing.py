"""
Author: Raphael Balzer
GitHub Copilot was used to assist with code
"""
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import isotree
from ton_iot_utils import ContextAwareImputer

class FancyIFPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.continuous_cols = [
            "duration", "src_bytes", "dst_bytes", 
            "missed_bytes", "src_pkts", "dst_pkts"
        ]

        self.freq_cols = [
            "proto", "service", "conn_state", "dns_rcode",
            "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
            "dns_qtype", "dns_qclass"
        ]
        self.binary_sparse_cols = ["dns_query"]

        self.cols_to_drop = [
            'dst_ip', 'src_ip', 'dst_port', 'src_port',
            "http_request_body_len", "http_response_body_len", "http_trans_depth",
            "http_orig_mime_types", "http_status_code", "http_resp_mime_types",
            "http_uri", "http_user_agent", "http_version", "http_method",
            "ssl_established", "ssl_resumed", "ssl_subject", "ssl_issuer",
            "ssl_cipher", "ssl_version", "weird_addl", "weird_notice", "weird_name"]
        
        self.new_feature_names = [
            'dns_pkts_ratio', 'ratio_pkts',
            'mean_bytes_src', 'dns_bytes_fraction', 'dst_pkts_per_sec'
        ]
        
        self.scaler = RobustScaler()
        self.encoder = FrequencyEncoder(cols=self.freq_cols)
        self.imputer = ContextAwareImputer(self.continuous_cols, self.freq_cols)
        self.final_columns_ = None
    
    def fit(self, X, y=None):
        """Lernt Statistiken nur auf den Trainingsdaten."""
        df = X.copy()
        df = self._basic_cleanup(df)

        df = self.imputer.transform(df)
        df = self.encoder.fit_transform(df)

        df = self._add_engineered_features(df)
        for col in self.new_feature_names:
            if col not in self.continuous_cols:
                self.continuous_cols.append(col)

        df = self._log_transform(df)
        self.scaler.fit(df[self.continuous_cols])

        self.final_columns_ = df.columns.tolist()
        return self

    def transform(self, X):
        """Wendet gelernte Statistiken auf neue Daten (Test/Prod) an."""
        df = X.copy()
        df = self._basic_cleanup(df)
        df = self.imputer.transform(df)
        df = self.encoder.transform(df)
        df = self._add_engineered_features(df)
        df = self._log_transform(df)
        df[self.continuous_cols] = self.scaler.transform(df[self.continuous_cols])
        return df.reindex(columns=self.final_columns_, fill_value=0)
    
    def _basic_cleanup(self, df):
        """Stateless Cleaning"""
        drop_list = self.cols_to_drop
        df = df.drop(columns=[c for c in drop_list if c in df.columns])

        df.replace('-', pd.NA, inplace=True)
        df = cleanup_protocol_orphans(df, 'dns_')

        for col in self.binary_sparse_cols:
            if col in df.columns:
                df[col] = df[col].notna().astype(int)

        return df
    
    def _log_transform(self, df):
        """Log-Trafo für schiefe Verteilungen"""
        for col in self.continuous_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
        return df
    
    def _add_engineered_features(self, df):
        df = df.copy()
        epsilon = 1e-3
        df['dns_pkts_ratio'] = df['dns_query'] * (df['dst_pkts'] / (df['src_pkts'] + epsilon))
        df['ratio_pkts'] = df['src_pkts'] / (df['dst_pkts'] + epsilon)
        df['mean_bytes_src'] = df['src_bytes'] / (df['src_pkts'] + epsilon)
        df['dns_bytes_fraction'] = df['dst_bytes'] / (df['src_bytes'] + df['dst_bytes'] + epsilon)
        df['dst_pkts_per_sec'] = df['dst_pkts'] / (df['duration'] + epsilon)
            
        return df

class BasicIFPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.continuous_cols = [
            "duration", "src_bytes", "dst_bytes", 
            "missed_bytes", "src_pkts", "dst_pkts"
        ]

        self.freq_cols = [
            "proto", "service", "conn_state", "dns_rcode",
            "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
            "dns_qtype", "dns_qclass"
        ]
        self.binary_sparse_cols = ["dns_query"]

        self.cols_to_drop = [
            'dst_ip', 'src_ip', 'dst_port', 'src_port',
            "http_request_body_len", "http_response_body_len", "http_trans_depth",
            "http_orig_mime_types", "http_status_code", "http_resp_mime_types",
            "http_uri", "http_user_agent", "http_version", "http_method",
            "ssl_established", "ssl_resumed", "ssl_subject", "ssl_issuer",
            "ssl_cipher", "ssl_version", "weird_addl", "weird_notice", "weird_name"]

        # 2. Initialisierung der State-Objekte (noch nicht gefittet)
        self.scaler = RobustScaler()
        self.encoder = FrequencyEncoder(cols=self.freq_cols)
        self.imputer = ContextAwareImputer(self.continuous_cols, self.freq_cols)
        self.final_columns_ = None # Speichert die Spaltenreihenfolge

    def fit(self, X, y=None):
        """Lernt Statistiken nur auf den Trainingsdaten."""
        df = X.copy()
        # Vorverarbeitung für das Lernen vorbereiten
        df = self._basic_cleanup(df)
        
        df = self.imputer.transform(df)
        df = self.encoder.fit_transform(df)
        # Scaler lernen (auf Log-transformierten Daten)
        df = self._log_transform(df)
        self.scaler.fit(df[self.continuous_cols])
        # Spaltenreihenfolge merken, damit Test-Set identisch aussieht
        self.final_columns_ = df.columns.tolist()
        return self

    def transform(self, X):
        """Wendet gelernte Statistiken auf neue Daten (Test/Prod) an."""
        df = X.copy()
        # 1. Identische Basis-Reinigung
        df = self._basic_cleanup(df)
        # 3. Imputer anwenden
        df = self.imputer.transform(df)
        df = self.encoder.transform(df)
        # 5. Log & Scaling
        df = self._log_transform(df)
        df[self.continuous_cols] = self.scaler.transform(df[self.continuous_cols])
        # 6. Sicherstellen, dass Spalten exakt wie im Training sind
        # (Fängt fehlende Spalten ab oder entfernt neue, unbekannte Spalten)
        return df.reindex(columns=self.final_columns_, fill_value=0)

    def _basic_cleanup(self, df):
        """Stateless Cleaning"""
        drop_list = self.cols_to_drop
        df = df.drop(columns=[c for c in drop_list if c in df.columns])
        
        # 2. Values säubern
        df.replace('-', pd.NA, inplace=True)
        df = cleanup_protocol_orphans(df, 'dns_')
        #df = cleanup_protocol_orphans(df, 'http_')
        #df = cleanup_protocol_orphans(df, 'ssl_')

        # 3. Binary Features erzeugen (Existenz -> 0/1)
        for col in self.binary_sparse_cols:
            if col in df.columns:
                df[col] = df[col].notna().astype(int)
        
        return df

    def _log_transform(self, df):
        """Log-Trafo für schiefe Verteilungen"""
        for col in self.continuous_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
        return df

class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.continuous_cols = [
            "duration", "src_bytes", "dst_bytes", "missed_bytes",
            "src_pkts", "dst_pkts",  
            "http_request_body_len", "http_response_body_len", "http_trans_depth"
        ]
        # Wir behalten die Kategorien als Liste, führen aber kein OHE mehr durch
        self.categorical_cols = [
            "proto", "service", "conn_state", "dns_rcode",
            "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
            "ssl_established", "ssl_resumed", "weird_addl",
            "http_orig_mime_types", "weird_notice", "http_version", "dns_query",
            "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent",
            "ssl_cipher", "ssl_version", "http_method",
            "http_status_code", "http_resp_mime_types", "weird_name"
        ]

        self.cols_to_drop = ['dst_ip', 'src_ip', 'dst_port', 'src_port']

        self.imputer = ContextAwareImputer(self.continuous_cols, self.categorical_cols)
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = self.preprocess(df)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=[c for c in self.cols_to_drop if c in df.columns])

        df = self.imputer.transform(df)
        df.replace('-', np.nan, inplace=True)

        df = cleanup_protocol_orphans(df, 'dns_')
        df = cleanup_protocol_orphans(df, 'http_')

        df[self.categorical_cols] = df[self.categorical_cols].astype(str)

        return df

class IsoTreeTuner:
    def __init__(self, X_train, y_train):
        """
        Initialisiert den Tuner mit den (bereits frequency-encoded) Daten.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.best_params = []
        self.seed = 42
        
        # Optuna Logging etwas leiser stellen
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _get_params_for_architecture(self, trial, architecture_type):
        """
        Definiert den Suchraum basierend auf der Architektur.
        """
        # Gemeinsame Basis-Parameter
        base_params = {
            "ntrees": 100,            # Fixiert für Vergleichbarkeit
            "missing_action": "fail", # Fail, da Frequency Encoding genutzt wird
            "scoring_metric": "depth",
            "random_state": 42,
            "sample_size": 256,        # Wird ggf. überschrieben
        }

        # Standard Isolation Forest
        if architecture_type == "IF":
            sample_size = trial.suggest_categorical("sample_size", [256, 512, 1024, 2048])
            max_depth = trial.suggest_int("max_depth", 4, 15) # None = auto
            
            spec_params = {
                "ndim": 1, 
                "ntry": 1, # FIX: iForest würfelt nur, optimiert nicht!
                "coefs": "uniform",
                "prob_pick_avg_gain": 0.0, 
                "prob_pick_pooled_gain": 0.0,
                "penalize_range": False,
                "max_depth": max_depth,
                "sample_size": sample_size
            }

        # Standard Extended Isolation Forest    
        elif architecture_type == "EIF":
            sample_size = trial.suggest_categorical("sample_size", [256, 512, 1024, 2048])
            max_depth = trial.suggest_int("max_depth", 4, 15)
            
            spec_params = {
                "ndim": 2, 
                "ntry": 1,             # FIX: EIF würfelt nur, optimiert nicht!
                "coefs": "uniform",    # FIX: EIF Standard (keine Normalverteilung)
                "prob_pick_avg_gain": 0.0, 
                "prob_pick_pooled_gain": 0.0,
                "penalize_range": False,
                "max_depth": max_depth,
                "sample_size": sample_size
            }

        # SCiForest
        elif architecture_type == "SCIF":
            sample_size = trial.suggest_categorical("sample_size", [256, 512, 1024, 2048])
            max_depth = trial.suggest_int("max_depth", 4, 15)
            ntry = trial.suggest_int("ntry", 2, 10) 
            prob_pick = trial.suggest_float("prob_pick_avg_gain", 0.5, 1.0)
            
            spec_params = {
                "ndim": 2, 
                "ntry": ntry, 
                "coefs": "normal",     # FIX: SCiForest nutzt Normalverteilung
                "prob_pick_avg_gain": prob_pick, 
                "prob_pick_pooled_gain": 0.0,
                "penalize_range": True, # FIX: Teil der SCiForest Logik
                "max_depth": max_depth#,
                #"sample_size": sample_size
            }
            
        # Fair-Cut Forest
        elif architecture_type == "FCF":
            sample_size = trial.suggest_categorical("sample_size", [256, 512, 1024, 2048])
            max_depth = trial.suggest_int("max_depth", 4, 15)
            ntry = trial.suggest_int("ntry", 2, 10)
            prob_pick = trial.suggest_float("prob_pick_pooled_gain", 0.5, 1.0)
            # Hier Penalize tunen, da es massiven Einfluss hat
            penalize_range = trial.suggest_categorical("penalize_range", [True, False])
            
            spec_params = {
                "ndim": 2, 
                "ntry": ntry, 
                "coefs": "normal",
                "prob_pick_avg_gain": 0.0, 
                "prob_pick_pooled_gain": prob_pick,
                "penalize_range": penalize_range,
                "max_depth": max_depth,
                "sample_size": sample_size
            }
        else:
            raise ValueError(f"Unbekannte Architektur: {architecture_type}")
            
        return {**base_params, **spec_params}

    def _objective(self, trial, architecture_type, n_splits=3):
        params = self._get_params_for_architecture(trial, architecture_type)
        
        # 1. Wir brauchen Indizes sauber resetet
        X = self.X_train.reset_index(drop=True)
        y = self.y_train.reset_index(drop=True)
        
        # 2. StratifiedKFold sorgt dafür, dass überall Angriffe drin sind
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        fold_separations = []
        
        for train_idx, val_idx in skf.split(X, y):
            # 1. Daten vorbereiten (Clean Train, Mixed Val)
            X_tr_raw = X.iloc[train_idx]
            y_tr_raw = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Training nur auf Normaldaten (y=0)
            X_tr_clean = X_tr_raw[y_tr_raw == 0]
            
            model = isotree.IsolationForest(**params)
            model.fit(X_tr_clean)
            
            # 2. Scores holen (Hier nehmen wir an: Je höher, desto anomaler)
            scores = model.decision_function(X_val) 
            
            # 3. Separation berechnen
            # Wir trennen die Scores anhand der Wahren Labels
            scores_normal = scores[y_val == 0]
            scores_attack = scores[y_val == 1]
            
            if len(scores_attack) == 0: 
                return 0.0 # Sollte dank StratifiedKFold nicht passieren
            
            # Kennzahl: Distanz der Mediane / Summe der Streuungen
            # Das ist ähnlich zum "Fisher Discriminant Ratio"
            median_diff = np.median(scores_attack) - np.median(scores_normal)
            spread = (scores_normal.std() + scores_attack.std()) + 1e-9
            
            separation_index = median_diff / spread
            fold_separations.append(separation_index)

        # Optuna soll diesen Index maximieren
        return np.mean(fold_separations)

    def tune(self, variants=["IF", "EIF", "SCIF", "FCF"], n_trials=20):
        """
        Führt Tuning durch.
        """
        print(f"Starte Benchmark für {len(variants)} Varianten mit je {n_trials} Trials...")
        
        for arch in variants:
            print(f"\n--- Processing: {arch.upper()} ---")
            # 1. Tuning (Optuna)
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(lambda t: self._objective(t, arch), n_trials=n_trials)
            
            best_params = self._get_params_for_architecture(study.best_trial, arch)
            self.best_params.append({"architecture": arch, "params": best_params})

        return self.best_params
    
    def build_model(self, architecture):
        """
        Erstellt ein Modell mit den besten Parametern für die gegebene Architektur.
        
        Args:
            architecture: Name der Architektur (z.B. "iforest", "eif", "sciforest", "fcf")
        
        Returns:
            Trainiertes isotree.IsolationForest Modell
        
        Raises:
            ValueError: Wenn die Architektur nicht in best_params gefunden wird
        """
        # Finde die Parameter für die gewünschte Architektur
        best_param_entry = None
        for entry in self.best_params:
            if entry["architecture"] == architecture:
                best_param_entry = entry
                break
        
        if best_param_entry is None:
            available = [entry["architecture"] for entry in self.best_params]
            raise ValueError(
                f"Architektur '{architecture}' nicht gefunden. "
                f"Verfügbar: {available}. Bitte zuerst 'tune()' ausführen."
            )
        
        params = best_param_entry["params"]
        print(f"\nBaue Modell für '{architecture}' mit besten Parametern...")
        print(f"Parameter: {params}")
        
        # Modell mit besten Parametern trainieren
        model = isotree.IsolationForest(**params)
        model.fit(self.X_train)
        
        print(f"Modell erfolgreich trainiert.")
        return model
    
class PredictorWrapper:
    def __init__(self, model, model_name, X, y_val):
        self.model = model
        self.model_name = model_name
        self.X = X
        self.y_val = y_val
        self.y_pred = None
        self.best_threshold = None

    def predict(self):
        scores = self.model.predict(self.X)
        # --- 4. Optimalen Threshold finden (statt fixer Perzentile) ---
        # Wir suchen den Threshold, der den F2-Score maximiert (Recall stärker gewichtet)
        precisions, recalls, thresholds = [], [], np.linspace(scores.min(), scores.max(), 100)
        best_f2 = 0
        best_threshold = 0

        for th in thresholds:
            y_pred_temp = (scores > th).astype(int)
            f2 = fbeta_score(self.y_val, y_pred_temp, beta=2)
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = th

        print(f"Optimaler Threshold gefunden: {best_threshold:.4f} (Max F2: {best_f2:.4f})")
        self.best_threshold = best_threshold
        # Finale Predictions mit optimalem Threshold
        self.y_pred = (scores > best_threshold).astype(int)
        return self.y_pred
    
    def evaluate(self):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        import seaborn as sns
        
        # Classification Report ausgeben
        report = classification_report(self.y_val, self.y_pred, target_names=['Normal', 'Attack'])
        print(report)
        
        # Erstelle Figure mit 3 Subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Model Evaluation: {self.model_name}', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_val, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                    xticklabels=['Normal', 'Attack'], 
                    yticklabels=['Normal', 'Attack'])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # 2. ROC-AUC Curve
        scores = self.model.predict(self.X)
        fpr, tpr, _ = roc_curve(self.y_val, scores)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC-AUC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        # 3. Score Distribution
        scores_normal = scores[self.y_val == 0]
        scores_attack = scores[self.y_val == 1]
        axes[2].hist(scores_normal, bins=50, alpha=0.6, label='Normal', color='blue')
        axes[2].hist(scores_attack, bins=50, alpha=0.6, label='Attack', color='red')
        axes[2].axvline(self.best_threshold, color='green', linestyle='--', 
                        linewidth=2, label=f'Threshold = {self.best_threshold:.3f}')
        axes[2].set_xlabel('Anomaly Score')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Score Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    
class FrequencyEncoder:
    def __init__(self, cols=None):
        self.cols = cols
        self.mappings = {}

    def fit(self, X, y=None):
        # Falls keine Spalten angegeben, nimm alle kategorialen (object/category)
        if self.cols is None:
            self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.cols:
            # Berechne relative Häufigkeiten auf dem Trainingsset
            self.mappings[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mappings.items():
            # Wende Mapping an. fillna(0) behandelt Werte, die im Train-Set nie vorkamen.
            X_copy[col] = X_copy[col].map(mapping).fillna(0)
        return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

def cleanup_protocol_orphans(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    # Trenne in Metadaten (Strings/Objekte) und Messwerte (Numerisch)
    meta_cols = df[cols].select_dtypes(exclude='number').columns
    num_cols = df[cols].select_dtypes(include='number').columns

    # Maske für Zeilen, in denen ALLE Header-Informationen fehlen
    header_missing = df[meta_cols].isna().all(axis=1)

    # In diesen Zeilen: Alle numerischen Spalten auf NaN setzen
    # (Egal ob sie 0 oder Restwerte enthielten)
    df.loc[header_missing, num_cols] = pd.NA
    return df

def plot_score_distribution(isoforest, X_test, Y_test, threshold, title=""):
    scores = isoforest.decision_function(X_test)
    plt.figure(figsize=(10, 6))
    plt.hist(scores[Y_test == 0], bins=50, label='Normal', alpha=0.5, color='blue', density=True)
    plt.hist(scores[Y_test == 1], bins=50, label='Angriff', alpha=0.5, color='red', density=True)
    plt.axvline(x=threshold, color='black', linestyle='--', label='Threshold')
    plt.legend()
    plt.title(f"Verteilung der Anomaly-Scores - {title}")
    plt.show()


def plot_score_distributions_grid(models_input, X_data, y_data, percentile=20, bins=40, figsize=None):
    """Plot multiple anomaly-score distributions in a shared subplot grid.

    Accepts a dict ``{name: model}``, a list of ``(name, model)`` tuples,
    or a plain list of models. Threshold per model is set via the given
    percentile on the decision_function scores.
    """

    # Normalize input to a dict with readable names
    if isinstance(models_input, dict):
        models_dict = models_input
    elif isinstance(models_input, list):
        if len(models_input) == 0:
            raise ValueError("models_input must not be empty")
        if isinstance(models_input[0], (tuple, list)) and len(models_input[0]) == 2:
            models_dict = {name: model for name, model in models_input}
        else:
            models_dict = {f"Model_{i+1}": model for i, model in enumerate(models_input)}
    else:
        models_dict = {"Model_1": models_input}

    model_names = list(models_dict.keys())
    num_models = len(model_names)

    # Grid sizing: up to 2 columns for readability
    cols = min(2, num_models)
    rows = math.ceil(num_models / cols)
    if figsize is None:
        figsize = (6 * cols, 4 * rows)

    y_array = np.asarray(y_data)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    thresholds = {}

    for idx, name in enumerate(model_names):
        model = models_dict[name]
        scores = model.decision_function(X_data)
        threshold = np.percentile(scores, percentile)
        thresholds[name] = threshold

        ax = axes_flat[idx]
        ax.hist(scores[y_array == 0], bins=bins, label='Normal', alpha=0.55, color='#1f77b4', density=True)
        ax.hist(scores[y_array == 1], bins=bins, label='Angriff', alpha=0.55, color='#d62728', density=True)
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=1.2, label=f'Schwelle p{percentile}')
        ax.set_title(f"{name} | Schwelle={threshold:.3f}")
        ax.set_xlabel('Score')
        ax.set_ylabel('Dichte')
        if idx == 0:
            ax.legend()

    # Hide any unused axes
    for j in range(num_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    return thresholds

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix using seaborn heatmap"""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()

def plot_feature_importances(models_input, X_data, y_data=None, top_n=15, figsize=None):
    """
    Berechnet und visualisiert Feature-Importances für Isolation Forest Modelle.
    
    Verbesserungen:
    - Zeigt nur Top-N Features für Lesbarkeit
    - Akzeptiert Liste oder Dict
    - Korrigierte Metrik für Unsupervised-Fall
    """
    
    # 1. Input-Handling: Liste in Dict umwandeln, falls nötig
    if isinstance(models_input, list):
        models_dict = {f"Model_{i+1}": m for i, m in enumerate(models_input)}
    elif isinstance(models_input, dict):
        models_dict = models_input
    else:
        models_dict = {"Model_1": models_input}

    num_models = len(models_dict)
    feature_names = X_data.columns
    
    # 2. Automatische Größe berechnen (optimiert für Lesbarkeit)
    # Breite: Genug Platz pro Subplot
    # Höhe: Abhängig von top_n, damit Balken nicht zu dünn werden
    if figsize is None:
        width = 6 * num_models 
        height = max(6, top_n * 0.4) 
        figsize = (width, height)
    
    importances_dict = {}
    
    # Subplots erstellen (sharey=False, da unterschiedliche Features wichtig sein können)
    fig, axes = plt.subplots(1, num_models, figsize=figsize, sharey=False)
    
    if num_models == 1:
        axes = [axes]
    
    print(f"Berechne Importances für {num_models} Modelle (Top {top_n} Features)...")

    for idx, (model_name, model) in enumerate(models_dict.items()):
        # Baseline berechnen
        baseline_scores = model.decision_function(X_data)
        
        if y_data is not None:
            baseline_metric = roc_auc_score(y_data, baseline_scores)
        
        importances = np.zeros(len(feature_names))
        X_permuted = X_data.copy()
        
        # Permutation Loop
        for feature_idx, feature in enumerate(feature_names):
            # Feature mischen
            X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
            permuted_scores = model.decision_function(X_permuted)
            
            if y_data is not None:
                # Supervised: Wie viel AUC verlieren wir?
                permuted_metric = roc_auc_score(y_data, permuted_scores)
                importances[feature_idx] = max(0, baseline_metric - permuted_metric)
            else:
                # Unsupervised: Wie stark ändert sich der Score absolut? (Impact)
                # Wir nutzen Mean Absolute Difference
                score_diff = np.abs(baseline_scores - permuted_scores)
                importances[feature_idx] = np.mean(score_diff)
            
            # Reset
            X_permuted[feature] = X_data[feature].values
        
        # Normalisieren
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        importances_dict[model_name] = importances
        
        # 3. Sortieren und auf Top-N beschneiden
        sorted_idx = np.argsort(importances)[::-1]
        top_idx = sorted_idx[:top_n]  # Nur die besten N Indizes
        
        top_features = feature_names[top_idx]
        top_importances = importances[top_idx]
        
        # 4. Plotting (Lesbarkeit!)
        ax = axes[idx]
        # y-Positionen für Balken
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_importances, color='#4c72b0', edgecolor='white', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=11) # Größere Schrift
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=15)
        ax.invert_yaxis() # Wichtigstes oben
        
        # Grid nur vertikal, hinter den Balken
        ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    
    return importances_dict