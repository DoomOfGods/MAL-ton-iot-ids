"""
ton_iot_utils.py
Utility classes and functions for TON-IOT anomaly detection pipeline
Claude Sonnet 4.5 was used for refactoring and to add comments and docstrings
GitHub Copilot was used to assist with code
"""

import os
import numpy as np
import pandas as pd
import time
import psutil
import platform
import tracemalloc
import json
from codecarbon import EmissionsTracker
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
from pathlib import Path

# ============================================================================
# RESOURCE MONITORING
# ============================================================================

class ResourceMonitor:
    """Monitor CPU, Memory, and Energy consumption during training"""
    
    def __init__(self, label="Process"):
        self.label = label
        self.cpu_samples = []

    def process_memory(self):
        """Return current total process memory in bytes"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def sample_cpu(self):
        """Sample current CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_samples.append(cpu_percent)
        return cpu_percent
        
    def start(self):
        """Start monitoring resources"""
        # Python memory
        tracemalloc.start()
        self.start_memory_python, _ = tracemalloc.get_traced_memory()

        # Total process memory
        self.start_memory_process = self.process_memory()

        # Time
        self.start_wall_time = time.time()
        self.start_process_time = time.process_time()

        # Energy
        self.emissions_tracker = EmissionsTracker(
            project_name="TON-IOT-SVM",
            measure_power_secs=1,
            save_to_file=False,
            log_level='error'
        )
        self.emissions_tracker.start()
        
        print(f"Resource monitoring started for: {self.label}")
        
    def stop(self):
        """Stop monitoring and return statistics"""
        # Stop energy
        emissions = self.emissions_tracker.stop() if self.emissions_tracker else 0
        
        # Time
        wall_time = time.time() - self.start_wall_time
        process_time = time.process_time() - self.start_process_time

        # Memory
        end_memory_python, peak_memory_python = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_python_memory = end_memory_python - self.start_memory_python

        end_memory_process = self.process_memory()
        total_process_memory = end_memory_process - self.start_memory_process
        peak_process_memory = max(self.start_memory_process, end_memory_process)

        # CPU
        avg_cpu = np.mean(self.cpu_samples) if self.cpu_samples else 0
        max_cpu = np.max(self.cpu_samples) if self.cpu_samples else 0
        
        stats = {
            'wall_time': wall_time,
            'process_time': process_time,
            'python_peak_memory_mb': peak_memory_python / (1024*1024),
            'python_total_memory_mb': total_python_memory / (1024*1024),
            'process_peak_memory_mb': peak_process_memory / (1024*1024),
            'process_total_memory_mb': total_process_memory / (1024*1024),
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max_cpu,
            'cpu_samples': len(self.cpu_samples),
            'energy_kwh': emissions if isinstance(emissions, float) else 0,
            'co2_kg': emissions if isinstance(emissions, float) else 0
        }
        
        print(f"Resource monitoring stopped for: {self.label}")
        self._print_stats(stats)
        
        return stats
    
    def _print_stats(self, stats):
        """Print resource statistics"""
        print("\n" + "="*70)
        print(f"RESOURCE USAGE - {self.label}")
        print("="*70)
        print(f"Time:")
        print(f"  Wall time:     {stats['wall_time']:.2f} seconds")
        print(f"  Process time:  {stats['process_time']:.2f} seconds (isolated CPU time)")
        print(f"  I/O wait:      {stats['wall_time'] - stats['process_time']:.2f} seconds")
        print(f"\nMemory (Python objects):")
        print(f"  Peak:  {stats['python_peak_memory_mb']:.2f} MB")
        print(f"  Total increase:  {stats['python_total_memory_mb']:.2f} MB")
        print(f"\nMemory (Total process):")
        print(f"  Peak:  {stats['process_peak_memory_mb']:.2f} MB")
        print(f"  Total increase:  {stats['process_total_memory_mb']:.2f} MB")
        print(f"\nCPU:")
        print(f"  Average: {stats['avg_cpu_percent']:.1f}%")
        print(f"  Peak:    {stats['max_cpu_percent']:.1f}%")
        print(f"  Samples: {stats['cpu_samples']}")
        print(f"\nEnergy:")
        print(f"  Consumption: {stats['energy_kwh']:.6f} kWh")
        print(f"  CO2 emissions: {stats['co2_kg']:.6f} kg")
        print("="*70 + "\n")

# ============================================================================
# CUSTOM TRANSFORMERS
# ============================================================================

class FeatureEliminator(BaseEstimator, TransformerMixin):
    """Remove specified features (e.g., IPs and ports for generalization)"""
    
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        return X_copy.drop(columns=self.features_to_drop)

class ContextAwareImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputation:
    1. Fill NaN with "-" for categorical features
    2. Set dns_rcode to "-" where service != "dns"
    3. Replace "-" with 0 for continuous features
    """
    
    def __init__(self, continuous_cols, categorical_cols):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Fill NaN with "-" for categorical
        for col in self.categorical_cols:
            X_copy[col] = X_copy[col].fillna('-').astype(str)
        
        # Set dns_rcode to "-" where service is not DNS
        if 'dns_rcode' in X_copy.columns and 'service' in X_copy.columns:
            X_copy.loc[X["service"] != "dns", "dns_rcode"] = "-"
        
        # Replace "-" with 0 for continuous features
        for col in self.continuous_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace('-', 0).pipe(pd.to_numeric).fillna(0)
        
        return X_copy

class BinaryPresenceEncoder(BaseEstimator, TransformerMixin):
    """Encode features as binary: 0 if missing, 1 if present"""
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_list()
        return self
    
    def transform(self, X):
        return ((X.astype(str) != '-') & (X != '0')).astype(int).to_numpy()
    
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(self.feature_names_in_, dtype=object)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sample_cpu_periodically(monitor, stop_event, interval=2):
    """Sample CPU usage periodically in a separate thread"""
    while not stop_event.is_set():
        if monitor:
            monitor.sample_cpu()
        time.sleep(interval)

def print_system_info():
    """Print current system information"""
    print(f"\nOperating System:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Version:  {platform.version()}")
    print(f"  Machine:  {platform.machine()}")
    print(f"  Processor: {platform.processor()}")

    print(f"\nCPU:")
    print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical cores:  {psutil.cpu_count(logical=True)}")
    print(f"  CPU frequency:  {psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "  CPU frequency: N/A")

    print(f"\nMemory:")
    print(f"  Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    print(f"\nPython:")
    print(f"  Version: {platform.python_version()}")

def get_system_info_dict():
    """Get system information as dictionary"""
    return {
        'timestamp': datetime.now().isoformat(),
        'operating_system': {
            'platform': f"{platform.system()} {platform.release()}",
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'cpu': {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'cpu_frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
        },
        'memory': {
            'total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'available_gb': round(psutil.virtual_memory().available / (1024**3), 1)
        },
        'python': {
            'version': platform.python_version()
        }
    }

def save_inference_results(
    model_name,
    model_type,
    pipeline_file,
    model_file,
    test_size,
    metrics,
    confusion_matrix,
    inference_stats=None,
    output_dir='inference_results',
    config=None):
    """
    Save inference results to JSON file for later comparison
    
    Args:
        model_name: Name/identifier for the model (e.g., 'linearsvc', 'sgd')
        model_type: Type of the model (e.g., 'LinearSVC', 'SGDClassifier')
        pipeline_file: Path to preprocessing pipeline
        model_file: Path to model file
        test_size: Number of test samples
        metrics: Dict with f1, f2, precision, recall, accuracy
        confusion_matrix: 2x2 numpy array [[TN, FP], [FN, TP]]
        inference_stats: Dict with timing/resource usage (optional)
        output_dir: Directory to save results
        config: Additional config information (optional)
    
    Returns:
        Path to saved file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate derived metrics
    cm = confusion_matrix
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Build results dictionary
    results = {
        'metadata': {
            'model_name': model_name,
            'model_type': model_type,
            'pipeline_file': str(pipeline_file),
            'model_file': str(model_file),
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_set_size': test_size
        },
        'system_info': get_system_info_dict(),
        'metrics': {
            'accuracy': round(metrics['accuracy'], 6),
            'f1_score': round(metrics['f1'], 6),
            'f2_score': round(metrics['f2'], 6),
            'precision': round(metrics['precision'], 6),
            'recall': round(metrics['recall'], 6)
        },
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'false_negative_rate': round(false_negative_rate, 6),
            'false_positive_rate': round(false_positive_rate, 6)
        },
        'security_analysis': {
            'total_attacks': int(fn + tp),
            'detected_attacks': int(tp),
            'missed_attacks': int(fn),
            'detection_rate_percent': round(metrics['recall'] * 100, 2),
            'miss_rate_percent': round(false_negative_rate * 100, 2),
            'false_alarms': int(fp)
        }
    }
    
    # Add inference stats if available
    if inference_stats:
        results['performance'] = {
            'timing': {
                'wall_time_seconds': round(inference_stats['wall_time'], 6),
                'process_time_seconds': round(inference_stats['process_time'], 6),
                'throughput_samples_per_second': round(test_size / inference_stats['wall_time'], 2),
                'per_sample_ms': round(inference_stats['wall_time'] / test_size * 1000, 6)
            },
            'memory': {
                'python_peak_mb': round(inference_stats['python_peak_memory_mb'], 2),
                'python_total_mb': round(inference_stats['python_total_memory_mb'], 2),
                'process_peak_mb': round(inference_stats['process_peak_memory_mb'], 2),
                'process_total_mb': round(inference_stats['process_total_memory_mb'], 2),
            },
            'cpu': {
                'average_percent': round(inference_stats['avg_cpu_percent'], 2),
                'peak_percent': round(inference_stats['max_cpu_percent'], 2)
            },
            'energy': {
                'consumption_kwh': inference_stats['energy_kwh'],
                'co2_kg': inference_stats['co2_kg']
            }
        }

    # Add config if provided
    if config:
        results['config'] = config
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_inference_{timestamp}.json"
    filepath = output_path / filename
    
    # Save to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"INFERENCE RESULTS SAVED")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    print(f"Model: {model_name} ({model_type})")
    print(f"F2-Score: {metrics['f2']:.4f}")
    print(f"Detection Rate: {metrics['recall']*100:.2f}%")
    print(f"{'='*70}\n")
    
    return filepath

def load_and_compare_results(result_files):
    """
    Load multiple inference result files and create comparison
    
    Args:
        result_files: List of paths to JSON result files
    
    Returns:
        DataFrame with comparison
    """
    import pandas as pd
    
    data = []
    for filepath in result_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # Extract key metrics for comparison
        row = {
            'Model': result['metadata']['model_name'],
            'Type': result['metadata']['model_type'],
            'Accuracy': result['metrics']['accuracy'],
            'F1': result['metrics']['f1_score'],
            'F2': result['metrics']['f2_score'],
            'Precision': result['metrics']['precision'],
            'Recall': result['metrics']['recall'],
            'Missed_Attacks': result['security_analysis']['missed_attacks'],
            'False_Alarms': result['security_analysis']['false_alarms'],
            'Timestamp': result['metadata']['evaluation_timestamp']
        }
        
        # Add performance metrics if available
        if 'performance' in result:
            row['Inference_Time_s'] = result['performance']['timing']['wall_time_seconds']
            row['Throughput_samples/s'] = result['performance']['timing']['throughput_samples_per_second']
            row['Python_Peak_Memory_MB'] = result['performance']['memory']['python_peak_mb']
            row['Python_Total_Memory_MB'] = result['performance']['memory']['python_total_mb']
            row['Process_Peak_Memory_MB'] = result['performance']['memory']['process_peak_mb']
            row['Process_Total_Memory_MB'] = result['performance']['memory']['process_total_mb']
            row['CPU_Peak'] = result['performance']['cpu']['peak_percent']
            row['Energy_kWh'] = result['performance']['energy']['consumption_kwh']
    
        data.append(row)

    return pd.DataFrame(data)

def print_comparison_summary(comparison_df):
    """Print formatted comparison of multiple models"""
    print("\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)
    
    # Sort by F2-score (descending)
    comparison_df = comparison_df.sort_values('F2', ascending=False)
    
    print("\n## Performance Metrics ##")
    print(comparison_df[['Model', 'Accuracy', 'F1', 'F2', 'Precision', 'Recall']].to_string(index=False))
    
    print("\n## Security Analysis ##")
    print(comparison_df[['Model', 'Missed_Attacks', 'False_Alarms']].to_string(index=False))
    
    if 'Inference_Time_s' in comparison_df.columns:
        print("\n## Efficiency Metrics ##")
        print(comparison_df[['Model', 'Inference_Time_s', 'Throughput_samples/s', 'Process_Peak_Memory_MB', 'CPU_Peak', 'Energy_kWh']].to_string(index=False))
    
    print("\n" + "="*100)
    
    # Highlight best model
    best_model = comparison_df.iloc[0]
    print(f"\nBEST MODEL (by F2-Score): {best_model['Model']}")
    print(f"  F2-Score: {best_model['F2']:.4f}")
    print(f"  Recall: {best_model['Recall']:.4f}")
    print(f"  Missed Attacks: {int(best_model['Missed_Attacks'])}")
    print("="*100 + "\n")