"""
Author: Raphael Balzer
GitHub Copilot was used to assist with code
"""
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, fbeta_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import isotree
from ton_iot_utils import ContextAwareImputer

class FancyIFPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor with feature engineering for Isolation Forest models."""
    
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
        """Apply learned statistics to new data (test/production)."""
        df = X.copy()
        df = self._basic_cleanup(df)
        df = self.imputer.transform(df)
        df = self.encoder.transform(df)
        df = self._add_engineered_features(df)
        df = self._log_transform(df)
        df[self.continuous_cols] = self.scaler.transform(df[self.continuous_cols])
        return df.reindex(columns=self.final_columns_, fill_value=0)
    
    def _basic_cleanup(self, df):
        """Perform stateless data cleaning."""
        drop_list = self.cols_to_drop
        df = df.drop(columns=[c for c in drop_list if c in df.columns])

        df.replace('-', pd.NA, inplace=True)
        df = cleanup_protocol_orphans(df, 'dns_')

        for col in self.binary_sparse_cols:
            if col in df.columns:
                df[col] = df[col].notna().astype(int)

        return df
    
    def _log_transform(self, df):
        """Apply log transformation for skewed distributions."""
        for col in self.continuous_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
        return df
    
    def _add_engineered_features(self, df):
        """Add engineered features based on domain knowledge."""
        df = df.copy()
        epsilon = 1e-3
        df['dns_pkts_ratio'] = df['dns_query'] * (df['dst_pkts'] / (df['src_pkts'] + epsilon))
        df['ratio_pkts'] = df['src_pkts'] / (df['dst_pkts'] + epsilon)
        df['mean_bytes_src'] = df['src_bytes'] / (df['src_pkts'] + epsilon)
        df['dns_bytes_fraction'] = df['dst_bytes'] / (df['src_bytes'] + df['dst_bytes'] + epsilon)
        df['dst_pkts_per_sec'] = df['dst_pkts'] / (df['duration'] + epsilon)
            
        return df

class BasicIFPreprocessor(BaseEstimator, TransformerMixin):
    """Basic preprocessor for Isolation Forest models without feature engineering."""
    
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

        # Initialize state objects (not yet fitted)
        self.scaler = RobustScaler()
        self.encoder = FrequencyEncoder(cols=self.freq_cols)
        self.imputer = ContextAwareImputer(self.continuous_cols, self.freq_cols)
        self.final_columns_ = None  # Store column order

    def fit(self, X, y=None):
        """Learn statistics from training data only."""
        df = X.copy()
        # Prepare preprocessing for learning
        df = self._basic_cleanup(df)
        
        df = self.imputer.transform(df)
        df = self.encoder.fit_transform(df)
        # Learn scaler on log-transformed data
        df = self._log_transform(df)
        self.scaler.fit(df[self.continuous_cols])
        # Remember column order so test set looks identical
        self.final_columns_ = df.columns.tolist()
        return self

    def transform(self, X):
        """Apply learned statistics to new data (test/production)."""
        df = X.copy()
        # Identical basic cleaning
        df = self._basic_cleanup(df)
        # Apply imputer
        df = self.imputer.transform(df)
        df = self.encoder.transform(df)
        # Log transformation and scaling
        df = self._log_transform(df)
        df[self.continuous_cols] = self.scaler.transform(df[self.continuous_cols])
        # Ensure columns match training set exactly
        # (handles missing columns or removes new, unknown columns)
        return df.reindex(columns=self.final_columns_, fill_value=0)

    def _basic_cleanup(self, df):
        """Perform stateless data cleaning."""
        drop_list = self.cols_to_drop
        df = df.drop(columns=[c for c in drop_list if c in df.columns])
        
        # Clean values
        df.replace('-', pd.NA, inplace=True)
        df = cleanup_protocol_orphans(df, 'dns_')
        #df = cleanup_protocol_orphans(df, 'http_')
        #df = cleanup_protocol_orphans(df, 'ssl_')

        # Generate binary features (existence -> 0/1)
        for col in self.binary_sparse_cols:
            if col in df.columns:
                df[col] = df[col].notna().astype(int)
        
        return df

    def _log_transform(self, df):
        """Apply log transformation for skewed distributions."""
        for col in self.continuous_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
        return df

class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for CatBoost models that preserves categorical features."""
    
    def __init__(self):
        self.continuous_cols = [
            "duration", "src_bytes", "dst_bytes", "missed_bytes",
            "src_pkts", "dst_pkts",  
            "http_request_body_len", "http_response_body_len", "http_trans_depth"
        ]
        # Keep categories as list but don't perform one-hot encoding
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
        """No-op fit method as this preprocessor is stateless."""
        return self

    def transform(self, X):
        """Apply preprocessing transformations."""
        df = X.copy()
        df = self.preprocess(df)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform full preprocessing pipeline."""
        df = df.drop(columns=[c for c in self.cols_to_drop if c in df.columns])

        df = self.imputer.transform(df)
        df.replace('-', np.nan, inplace=True)

        df = cleanup_protocol_orphans(df, 'dns_')
        df = cleanup_protocol_orphans(df, 'http_')

        df[self.categorical_cols] = df[self.categorical_cols].astype(str)

        return df

class IsoTreeTuner:
    """Hyperparameter tuning for Isolation Tree models using Optuna."""
    
    def __init__(self, X_train, y_train):
        """Initialize tuner with pre-encoded training data."""
        self.X_train = X_train
        self.y_train = y_train
        self.best_params = []
        self.seed = 42
        
        # Reduce Optuna logging verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _get_params_for_architecture(self, trial, architecture_type):
        """Define search space based on architecture type."""
        # Gemeinsame Basis-Parameter
        base_params = {
            "ntrees": 100,            # Fixed for comparability
            "missing_action": "fail", # Fail because frequency encoding is used
            "scoring_metric": "depth",
            "random_state": 42,
            "sample_size": 256,        # May be overridden
        }

        # Standard Isolation Forest
        if architecture_type == "IF":
            sample_size = trial.suggest_categorical("sample_size", [256, 512, 1024, 2048])
            max_depth = trial.suggest_int("max_depth", 4, 15) # None = auto
            
            spec_params = {
                "ndim": 1, 
                "ntry": 1, # FIX: iForest only samples, doesn't optimize
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
                "ntry": 1,             # FIX: EIF only samples, doesn't optimize
                "coefs": "uniform",    # FIX: EIF standard (no normal distribution)
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
                "coefs": "normal",     # FIX: SCiForest uses normal distribution
                "prob_pick_avg_gain": prob_pick, 
                "prob_pick_pooled_gain": 0.0,
                "penalize_range": True, # FIX: Part of SCiForest logic
                "max_depth": max_depth#,
                #"sample_size": sample_size
            }
            
        # Fair-Cut Forest
        elif architecture_type == "FCF":
            sample_size = trial.suggest_categorical("sample_size", [256, 512, 1024, 2048])
            max_depth = trial.suggest_int("max_depth", 4, 15)
            ntry = trial.suggest_int("ntry", 2, 10)
            prob_pick = trial.suggest_float("prob_pick_pooled_gain", 0.5, 1.0)
            # Tune penalize_range as it has massive impact
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
        """Objective function for Optuna optimization."""
        params = self._get_params_for_architecture(trial, architecture_type)
        
        # Clean reset of indices
        X = self.X_train.reset_index(drop=True)
        y = self.y_train.reset_index(drop=True)
        
        # StratifiedKFold ensures attacks are present in all folds
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        fold_separations = []
        
        for train_idx, val_idx in skf.split(X, y):
            # Prepare data (clean train, mixed validation)
            X_tr_raw = X.iloc[train_idx]
            y_tr_raw = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Train only on normal data (y=0)
            X_tr_clean = X_tr_raw[y_tr_raw == 0]
            
            model = isotree.IsolationForest(**params)
            model.fit(X_tr_clean)
            
            # Get scores (assumption: higher = more anomalous)
            scores = model.decision_function(X_val) 
            
            # Calculate separation
            # Separate scores by true labels
            scores_normal = scores[y_val == 0]
            scores_attack = scores[y_val == 1]
            
            if len(scores_attack) == 0: 
                return 0.0  # Should not happen with StratifiedKFold
            
            # Metric: distance of medians / sum of spreads
            # Similar to Fisher Discriminant Ratio
            median_diff = np.median(scores_attack) - np.median(scores_normal)
            spread = (scores_normal.std() + scores_attack.std()) + 1e-9
            
            separation_index = median_diff / spread
            fold_separations.append(separation_index)

        # Optuna maximizes this index
        return np.mean(fold_separations)

    def tune(self, variants=["IF", "EIF", "SCIF", "FCF"], n_trials=20):
        """Perform hyperparameter tuning for specified model variants."""
        print(f"Starte Benchmark für {len(variants)} Varianten mit je {n_trials} Trials...")
        
        for arch in variants:
            print(f"\n--- Processing: {arch.upper()} ---")
            # Tuning with Optuna
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(lambda t: self._objective(t, arch), n_trials=n_trials)
            
            best_params = self._get_params_for_architecture(study.best_trial, arch)
            self.best_params.append({"architecture": arch, "params": best_params})

        return self.best_params
    
    def build_model(self, architecture):
        """Build model with best parameters for given architecture.
        
        Args:
            architecture: Architecture name (e.g., "iforest", "eif", "sciforest", "fcf")
        
        Returns:
            Trained isotree.IsolationForest model
        
        Raises:
            ValueError: If architecture not found in best_params
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
        print(f"\nBuilding model for '{architecture}' with best parameters...")
        print(f"Parameters: {params}")
        
        # Train model with best parameters
        model = isotree.IsolationForest(**params)
        model.fit(self.X_train)
        
        print(f"Model trained successfully.")
        return model
    
class PredictorWrapper:
    """Wrapper for model prediction and evaluation with optimal threshold finding."""
    
    def __init__(self, model, model_name, X, y_val):
        self.model = model
        self.model_name = model_name
        self.X = X
        self.y_val = y_val
        self.y_pred = None
        self.best_threshold = None

    def predict(self):
        """Find optimal threshold and generate predictions."""
        scores = self.model.predict(self.X)
        # Find optimal threshold (instead of fixed percentile)
        # Search for threshold that maximizes F2-score (recall weighted higher)
        precisions, recalls, thresholds = [], [], np.linspace(scores.min(), scores.max(), 100)
        best_f2 = 0
        best_threshold = 0

        for th in thresholds:
            y_pred_temp = (scores > th).astype(int)
            f2 = fbeta_score(self.y_val, y_pred_temp, beta=2)
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = th

        print(f"Optimal threshold found: {best_threshold:.4f} (Max F2: {best_f2:.4f})")
        self.best_threshold = best_threshold
        # Final predictions with optimal threshold
        self.y_pred = (scores > best_threshold).astype(int)
        return self.y_pred
    
    def evaluate(self):
        """Evaluate model and visualize results with confusion matrix, ROC curve, and score distribution."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        import seaborn as sns
        
        # Print classification report
        report = classification_report(self.y_val, self.y_pred, target_names=['Normal', 'Attack'])
        print(report)
        
        # Create figure with 3 subplots
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
    """Encoder that transforms categorical features to their relative frequencies."""
    
    def __init__(self, cols=None):
        self.cols = cols
        self.mappings = {}

    def fit(self, X, y=None):
        """Learn frequency mappings from training data."""
        # If no columns specified, use all categorical (object/category)
        if self.cols is None:
            self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.cols:
            # Calculate relative frequencies on training set
            self.mappings[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        """Apply frequency encoding to data."""
        X_copy = X.copy()
        for col, mapping in self.mappings.items():
            # Apply mapping. fillna(0) handles values never seen in training
            X_copy[col] = X_copy[col].map(mapping).fillna(0)
        return X_copy

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

def cleanup_protocol_orphans(df, prefix):
    """Clean orphaned numeric values for protocol-specific columns when metadata is missing."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    # Separate metadata (strings/objects) and measurements (numeric)
    meta_cols = df[cols].select_dtypes(exclude='number').columns
    num_cols = df[cols].select_dtypes(include='number').columns

    # Mask for rows where ALL header information is missing
    header_missing = df[meta_cols].isna().all(axis=1)

    # In these rows: set all numeric columns to NaN
    # (regardless of whether they contained 0 or residual values)
    df.loc[header_missing, num_cols] = pd.NA
    return df

def plot_score_distribution(isoforest, X_test, Y_test, threshold, title=""):
    """Plot anomaly score distribution for normal vs attack samples."""
    scores = isoforest.decision_function(X_test)
    plt.figure(figsize=(10, 6))
    plt.hist(scores[Y_test == 0], bins=50, label='Normal', alpha=0.5, color='blue', density=True)
    plt.hist(scores[Y_test == 1], bins=50, label='Attack', alpha=0.5, color='red', density=True)
    plt.axvline(x=threshold, color='black', linestyle='--', label='Threshold')
    plt.legend()
    plt.title(f"Anomaly Score Distribution - {title}")
    plt.show()


def plot_score_distributions_grid(models_input, X_data, y_data, percentile=20, bins=40, figsize=None):
    """Plot multiple anomaly score distributions in a shared subplot grid.

    Accepts a dict {name: model}, a list of (name, model) tuples,
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
        ax.hist(scores[y_array == 1], bins=bins, label='Attack', alpha=0.55, color='#d62728', density=True)
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=1.2, label=f'Threshold p{percentile}')
        ax.set_title(f"{name} | Threshold={threshold:.3f}")
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        if idx == 0:
            ax.legend()

    # Hide any unused axes
    for j in range(num_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    return thresholds

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix using seaborn heatmap."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()

def plot_feature_importances(models_input, X_data, y_data=None, top_n=15, figsize=None):
    """Calculate and visualize feature importances for Isolation Forest models.
    
    Features:
    - Shows only top-N features for readability
    - Accepts list or dict
    - Corrected metric for unsupervised case
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
    
    # Calculate automatic size (optimized for readability)
    # Width: enough space per subplot
    # Height: depends on top_n so bars aren't too thin
    if figsize is None:
        width = 6 * num_models 
        height = max(6, top_n * 0.4) 
        figsize = (width, height)
    
    importances_dict = {}
    
    # Create subplots (sharey=False, as different features may be important)
    fig, axes = plt.subplots(1, num_models, figsize=figsize, sharey=False)
    
    if num_models == 1:
        axes = [axes]
    
    print(f"Calculating importances for {num_models} models (Top {top_n} features)...")

    for idx, (model_name, model) in enumerate(models_dict.items()):
        # Calculate baseline
        baseline_scores = model.decision_function(X_data)
        
        if y_data is not None:
            baseline_metric = roc_auc_score(y_data, baseline_scores)
        
        importances = np.zeros(len(feature_names))
        X_permuted = X_data.copy()
        
        # Permutation loop
        for feature_idx, feature in enumerate(feature_names):
            # Shuffle feature
            X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
            permuted_scores = model.decision_function(X_permuted)
            
            if y_data is not None:
                # Supervised: how much AUC do we lose?
                permuted_metric = roc_auc_score(y_data, permuted_scores)
                importances[feature_idx] = max(0, baseline_metric - permuted_metric)
            else:
                # Unsupervised: how much does the score change absolutely? (impact)
                # Use mean absolute difference
                score_diff = np.abs(baseline_scores - permuted_scores)
                importances[feature_idx] = np.mean(score_diff)
            
            # Reset
            X_permuted[feature] = X_data[feature].values
        
        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        importances_dict[model_name] = importances
        
        # Sort and trim to top-N
        sorted_idx = np.argsort(importances)[::-1]
        top_idx = sorted_idx[:top_n]  # Only the best N indices
        
        top_features = feature_names[top_idx]
        top_importances = importances[top_idx]
        
        # Plotting (for readability)
        ax = axes[idx]
        # Y-positions for bars
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_importances, color='#4c72b0', edgecolor='white', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=11)  # Larger font
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=15)
        ax.invert_yaxis()  # Most important on top
        
        # Grid only vertical, behind bars
        ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    
    return importances_dict