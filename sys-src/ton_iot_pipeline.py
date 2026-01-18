"""
ton_iot_pipeline.py
Pipeline functions for data loading, preprocessing, training, and evaluation
Claude Sonnet 4.5 was used for refactoring, to add comments and docstrings and for the implementation of OCSVM mode in train_svm() and evaluate_model()
GitHub Copilot was used to assist with code
"""

import numpy as np
import pandas as pd
from isotree import IsolationForest
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                             precision_score, recall_score, accuracy_score, fbeta_score, make_scorer)
import threading

from ton_iot_utils import (
    FeatureEliminator, ContextAwareImputer, BinaryPresenceEncoder,
    ResourceMonitor, sample_cpu_periodically
)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_split_data(filepath, test_size_1=0.30, test_size_2=0.50, random_state=42):
    """
    Load TON-IOT dataset and perform stratified split
    
    Args:
        filepath: Path to CSV file
        test_size_1: First split ratio (for val+test)
        test_size_2: Second split ratio (to separate val and test)
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, strat_train, strat_val, strat_test
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    X = df.drop(columns=['label', 'type'])
    y = df['label'] # Binary: 0=normal, 1=attack
    stratify_col = df['type']
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
        X, y, stratify_col,
        test_size=test_size_1,
        stratify=stratify_col,
        random_state=random_state
    )
    
    # Second split: 15% val, 15% test
    X_val, X_test, y_val, y_test, strat_val, strat_test = train_test_split(
        X_temp, y_temp, strat_temp,
        test_size=test_size_2,
        stratify=strat_temp,
        random_state=random_state
    )
    
    print(f"\nData split (stratified by attack type):")
    print(f"  Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Val:   {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    print(f"\nLabel distribution (0=normal, 1=attack):")
    print(f"  Train: {y_train.mean()*100:.2f}% attacks")
    print(f"  Val:   {y_val.mean()*100:.2f}% attacks")
    print(f"  Test:  {y_test.mean()*100:.2f}% attacks")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, strat_train, strat_val, strat_test


# ============================================================================
# PREPROCESSING
# ============================================================================

def create_preprocessing_pipeline():
    """Create sklearn preprocessing pipeline"""
    
    continuous_cols = [
        "duration", "src_bytes", "dst_bytes", "missed_bytes",
        "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes", 
        "http_request_body_len", "http_response_body_len", "http_trans_depth"
    ]
    
    ohe = [
        "proto", "service", "conn_state", "dns_qtype", "dns_rcode",
        "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
        "ssl_established", "ssl_resumed", "weird_addl", 
        "http_orig_mime_types", "dns_qclass"
    ]
    
    binary_sparse = [
        "weird_notice", "http_version", "dns_query",
        "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent",
        "ssl_cipher", "ssl_version", "http_method", 
        "http_status_code", "http_resp_mime_types", "weird_name"
    ]
    
    categorical_cols = ohe + binary_sparse
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', RobustScaler(), continuous_cols),
            ('ohe', OneHotEncoder(
                drop='first',
                handle_unknown='ignore',
                sparse_output=False
            ), ohe),
            ('binary_sparse', BinaryPresenceEncoder(), binary_sparse)
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('eliminate', FeatureEliminator(['src_ip', 'src_port', 'dst_ip', 'dst_port'])),
        ('impute', ContextAwareImputer(continuous_cols, categorical_cols)),
        ('preprocess', preprocessor)
    ])
    
    return pipeline

def preprocess_data(X_train, X_val, X_test=None):
    """
    Fit preprocessing pipeline on train and transform all sets
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features (optional, can remain None)
        monitor: Enable resource monitoring
    
    Returns:
        pipeline, X_train_processed, X_val_processed, X_test_processed (or None), stats
    """
    print("\n" + "="*70)
    print("PREPROCESSING")
    print("="*70)
    
    pipeline = create_preprocessing_pipeline()
      
    print("\nFitting pipeline on training data...")
    X_train_processed = pipeline.fit_transform(X_train)
    
    print("Transforming validation data...")
    X_val_processed = pipeline.transform(X_val)
    
    X_test_processed = None
    if X_test is not None:
        print("Transforming test data...")
        X_test_processed = pipeline.transform(X_test)
    
    print(f"\nProcessed dimensions:")
    print(f"  Training:   {X_train_processed.shape}")
    print(f"  Validation: {X_val_processed.shape}")
    if X_test_processed is not None:
        print(f"  Test:       {X_test_processed.shape}")
    
    feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
    print(f"  Total features: {len(feature_names)}")
    
    return pipeline, X_train_processed, X_val_processed, X_test_processed

# ============================================================================
# TRAINING
# ============================================================================

def train_svm(X_train, y_train, mode='linearsvc', monitor=True, n_trials=200, timeout=3600):
    """
    Train classifier with different modes using Optuna for hyperparameter optimization
    Uses F2-score (prioritizes recall for security applications)
    
    Args:
        X_train: Preprocessed training features
        y_train: Training labels
        mode: Training mode - one of:
            - 'svc': SVC with RBF kernel
            - 'linearsvc': LinearSVC
            - 'sgd': SGDClassifier
            - 'ocsvm': OneClassSVM (unsupervised)
        monitor: Enable resource monitoring
        n_trials: Number of Optuna trials (default: 200, timeout will stop early if needed)
        timeout: Timeout in seconds for Optuna optimization (default: 3600 = 1 hour)
    
    Returns:
        best_model, training_stats
    """
    print("\n" + "="*70)
    print("TRAINING SVM")
    print("="*70)
        
    # F2-score: Weighs recall higher than precision
    # Alternative: 'recall' for maximum attack detection, but risks high FP rate
    f2_scorer = make_scorer(fbeta_score, beta=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    # Start monitoring
    training_monitor = ResourceMonitor("SVM Training") if monitor else None
    if training_monitor:
        training_monitor.start()
    
    # CPU sampling in background
    stop_sampling = threading.Event()
    sampling_thread = None
    if training_monitor:
        sampling_thread = threading.Thread(
            target=sample_cpu_periodically, 
            args=(training_monitor, stop_sampling, 2),
            daemon=True
        )
        sampling_thread.start()
    
    # Configure based on mode
    if mode == 'svc':
        print("\n Mode: SVC with Optuna (RBF Kernel)")
        print("   - Non-linear kernel (can capture complex patterns)")
        print("   - SLOW on large datasets but potentially better accuracy")
        print(f"   - Optuna optimization: {n_trials} trials, 5-fold CV")
        print(f"   - Timeout: {timeout}s" if timeout else "   - No timeout set")
        print("   - Intelligent hyperparameter search (TPE sampler)")
        
        def objective(trial):
            C = trial.suggest_float('C', 0.1, 100, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            
            model = SVC(
                C=C,
                gamma=gamma,
                kernel='rbf',
                class_weight='balanced',
                random_state=42
            )
            
            # Cross-validation with F2-Score
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=f2_scorer, n_jobs=-1)
            return scores.mean()
        
        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"\n Starting Optuna optimization...")
        print(f"   WARNING: This may take a while (up to {timeout}s if timeout set)...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"\n   Completed {len(study.trials)} trials")
        
        best_model = SVC(
            C=best_params['C'],
            gamma=best_params['gamma'],
            kernel='rbf',
            class_weight='balanced',
            random_state=42
        )
        best_model.fit(X_train, y_train)
        
    elif mode == 'linearsvc':
        print("\n Mode: LinearSVC with Optuna")
        print("   - Linear kernel only (faster than RBF)")
        print("   - Scales well to large datasets")
        print(f"   - Optuna optimization: {n_trials} trials, 5-fold CV")
        print("   - Intelligent hyperparameter search (TPE sampler)")
        
        def objective(trial):
            C = trial.suggest_float('C', 0.001, 100, log=True)
            loss = trial.suggest_categorical('loss', ['hinge', 'squared_hinge'])
            
            model = LinearSVC(
                C=C,
                loss=loss,
                class_weight='balanced',
                max_iter=2000,
                dual='auto',
                random_state=42
            )
            
            # Cross-validation with F2-Score
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=f2_scorer, n_jobs=-1)
            return scores.mean()
        
        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"\n Starting Optuna optimization...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        best_model = LinearSVC(
            C=best_params['C'],
            loss=best_params['loss'],
            class_weight='balanced',
            max_iter=2000,
            dual='auto',
            random_state=42
        )
        best_model.fit(X_train, y_train)
        
    elif mode == 'sgd':
        print("\n Mode: SGDClassifier with Optuna")
        print("   - Stochastic Gradient Descent (fast, requires more tuning)")
        print("   - Scales well to large datasets")
        print(f"   - Optuna optimization: {n_trials} trials, 5-fold CV")
        print("   - Intelligent hyperparameter search (TPE sampler)")
        
        def objective(trial):
            loss = trial.suggest_categorical('loss', ['hinge', 'log_loss'])
            alpha = trial.suggest_float('alpha', 0.00001, 0.1, log=True)
            penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
            
            # l1 and elasticnet penalties require specific solver
            if penalty in ['l1', 'elasticnet']:
                l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9) if penalty == 'elasticnet' else 0.15
            else:
                l1_ratio = 0.15
            
            model = SGDClassifier(
                loss=loss,
                alpha=alpha,
                penalty=penalty,
                l1_ratio=l1_ratio,
                class_weight='balanced',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation with F2-Score
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=f2_scorer, n_jobs=-1)
            return scores.mean()
        
        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"\n Starting Optuna optimization...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Handle l1_ratio for final model
        l1_ratio = best_params.get('l1_ratio', 0.15)
        
        best_model = SGDClassifier(
            loss=best_params['loss'],
            alpha=best_params['alpha'],
            penalty=best_params['penalty'],
            l1_ratio=l1_ratio,
            class_weight='balanced',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            n_jobs=-1
        )
        best_model.fit(X_train, y_train)

    elif mode == 'ocsvm':
        print("\n Mode: OneClassSVM with Optuna (Novelty Detection)")
        print("   - Unsupervised anomaly detection")
        print("   - Trains ONLY on normal data (y==0)")
        print("   - Learns boundary of normal behavior")
        print(f"   - Optuna optimization: {n_trials} trials, custom validation")
        print("   - Intelligent hyperparameter search (TPE sampler)")
        
        # Filter only normal data for training
        normal_mask = (y_train == 0)
        X_train_normal = X_train[normal_mask]
        
        print(f"\n   Using {len(X_train_normal):,} normal samples (filtered from {len(X_train):,} total)")
        print(f"   Attack ratio in full training set: {(~normal_mask).sum() / len(y_train) * 100:.1f}%")
        
        def objective(trial):
            nu = trial.suggest_float('nu', 0.01, 1)  # upper bound on the fraction of training errors and a lower bound of the fraction of support vectors
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            
            model = OneClassSVM(
                nu=nu,
                gamma=gamma,
                kernel='rbf',
            )
            
            n_val = min(5000, int(len(X_train_normal) * 0.2))
            indices = np.random.RandomState(42).permutation(len(X_train_normal))
            
            train_idx = indices[n_val:]
            val_idx = indices[:n_val]
            
            X_train_cv = X_train_normal[train_idx]
            X_val_normal = X_train_normal[val_idx]
            
            # Add attack samples to validation (simulate real scenario)
            X_train_attacks = X_train[~normal_mask]
            n_attacks_val = min(len(X_val_normal), len(X_train_attacks))
            attack_idx = np.random.RandomState(42).choice(len(X_train_attacks), n_attacks_val, replace=False)
            X_val_attacks = X_train_attacks[attack_idx]
            
            # Combine for validation
            X_val_mixed = np.vstack([X_val_normal, X_val_attacks])
            y_val_mixed = np.hstack([np.zeros(len(X_val_normal)), np.ones(len(X_val_attacks))])
            
            # Train on normal data only
            model.fit(X_train_cv)
            
            # Predict on mixed validation set
            y_pred_raw = model.predict(X_val_mixed)
            # Convert: -1 (anomaly) → 1 (attack), +1 (normal) → 0 (normal)
            y_pred = (y_pred_raw == -1).astype(int)
            
            # Calculate F2-score
            f2 = fbeta_score(y_val_mixed, y_pred, beta=2, zero_division=0)
            return f2
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"\n Starting Optuna optimization...")
        print(f"   Note: Each trial trains on normal data, validates on mixed data")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"\n   Completed {len(study.trials)} trials")
        
        # Train final model on ALL normal data
        best_model = OneClassSVM(
            nu=best_params['nu'],
            gamma=best_params['gamma'],
            kernel='rbf',
        )
        best_model.fit(X_train_normal)
        
        print(f"   Final model trained on {len(X_train_normal):,} normal samples")
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: 'svc_simple', 'linearsvc', 'sgd', 'ocsvm")
    
    # Stop sampling
    if sampling_thread:
        stop_sampling.set()
        sampling_thread.join(timeout=1)
    
    # Stop monitoring
    training_stats = None
    if training_monitor:
        training_stats = training_monitor.stop()
    
    print("\n" + "-"*70)
    if mode not in ['svc_simple']:
        print(f" Best parameters: {best_params}")
        print(f" Best CV F2-score: {best_score:.4f}")
        print(f" Trials completed: {len(study.trials)}/{n_trials}")
        if len(study.trials) < n_trials:
            print(f" (Stopped early due to timeout)")
    else:
        print(f" Model trained with parameters: {best_params}")
        print(f"  (No optimization - model not tuned)")
    print("-"*70)
    
    return best_model, training_stats

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_val, y_val, monitor=True, dataset_name="Validation",  isoforest_threshold=70):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_val: Preprocessed features
        y_val: True labels
        monitor: Enable resource monitoring
        dataset_name: Name for display
    
    Returns:
        metrics: Dictionary with performance metrics
    """
    print("\n" + "="*70)
    print(f"EVALUATION ON {dataset_name.upper()} SET")
    print("="*70)

    # Check if OneClassSVM
    is_one_class = isinstance(model, OneClassSVM)
    if is_one_class:
        print("\n Model type: OneClassSVM (unsupervised)")
        print("   Predictions: +1 (normal) → 0, -1 (anomaly) → 1")

    # Check if Isolation Forest (Raphael Balzer)
    is_isoforest = isinstance(model, IsolationForest)
    if is_isoforest:
        print("\n Model type: Isolation Forest (unsupervised)")
    
    # Monitor inference
    inference_monitor = ResourceMonitor(f"{dataset_name} Inference") if monitor else None
    if inference_monitor:
        inference_monitor.start()
    
    # CPU sampling in background
    stop_sampling = threading.Event()
    sampling_thread = None
    if inference_monitor:
        sampling_thread = threading.Thread(
            target=sample_cpu_periodically, 
            args=(inference_monitor, stop_sampling, 2),
            daemon=True
        )
        sampling_thread.start()

    
    y_pred_raw = model.predict(X_val)
    
    # Convert predictions for OneClassSVM
    if is_one_class:
        # OneClassSVM returns: +1 (normal), -1 (anomaly)
        # Convert to: 0 (normal), 1 (attack)
        y_pred = (y_pred_raw == -1).astype(int)
    # Convert predictions for Isolation Forest (Raphael Balzer)
    elif is_isoforest:
        scores = model.predict(X_val)
        threshold = np.percentile(scores, isoforest_threshold) 
        y_pred = (scores > threshold).astype(int)

    else:
        y_pred = y_pred_raw
    
    # Stop sampling
    if sampling_thread:
        stop_sampling.set()
        sampling_thread.join(timeout=1)

    # Stop monitoring
    inference_stats = None
    if inference_monitor:
        inference_stats = inference_monitor.stop()
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Normal', 'Attack']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    print(f"\n  [[TN={cm[0,0]:5}  FP={cm[0,1]:5}]   TN: Normal correctly classified")
    print(f"   [FN={cm[1,0]:5}  TP={cm[1,1]:5}]]   FP: False alarm")
    print(f"                         FN: Missed attack!")
    print(f"                         TP: Attack detected")
    
    f1 = f1_score(y_val, y_pred)
    f2 = fbeta_score(y_val, y_pred, beta=2)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    false_negatives = cm[1, 0]
    false_negative_rate = false_negatives / (cm[1, 0] + cm[1, 1])
    
    print("\n" + "-"*70)
    print("SUMMARY METRICS:")
    print(f"F1-Score:    {f1:.4f}")
    print(f"F2-Score:    {f2:.4f} (prioritizes recall)")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f} <- KEY METRIC")
    print(f"\nSecurity Analysis:")
    print(f"  Detected attacks: {cm[1,1]} / {cm[1,0] + cm[1,1]} ({recall*100:.1f}%)")
    print(f"  Missed attacks:   {false_negatives} ({false_negative_rate*100:.1f}%)")
    print(f"  False alarms:     {cm[0,1]}")
    if inference_stats:
        print(f"\nInference throughput: {len(X_val)/inference_stats['wall_time']:.0f} samples/second")
    print("-"*70)
    
    return {
        'f1': f1,
        'f2': f2,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'false_negative_rate': false_negative_rate,
        'confusion_matrix': cm,
        'inference_stats': inference_stats
    }
