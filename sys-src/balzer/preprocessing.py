from sklearn.preprocessing import RobustScaler, OneHotEncoder
from ton_iot_utils import ContextAwareImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class IsoForestPreprocessor:
    def __init__(self):
        self.continuous_cols = [
            "duration", "src_bytes", "dst_bytes", "missed_bytes",
            "src_pkts", "dst_pkts",  
            "http_request_body_len", "http_response_body_len", "http_trans_depth"
        ]
        self.ohe_cols = [
            "proto", "service", "conn_state", "dns_rcode",
            "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
            "ssl_established", "ssl_resumed", "weird_addl",
            "http_orig_mime_types", "dns_qtype", "dns_qclass"
        ]
        self.binary_sparse_cols = [
            "weird_notice", "http_version", "dns_query",
            "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent",
            "ssl_cipher", "ssl_version", "http_method",
            "http_status_code", "http_resp_mime_types", "weird_name"
        ]
        self.categorical_cols = self.ohe_cols + self.binary_sparse_cols
        #self.imputer = ContextAwareImputer(self.continuous_cols, self.categorical_cols)
        self.scaler = RobustScaler()
        self.ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        self.conn_state_freq_map = {}

    def preprocess(self, df: pd.DataFrame, train=True) -> pd.DataFrame:
        def get_col_group(starting_str, df):
            return [col for col in df.columns if col.startswith(starting_str)]

        ssl_cols = get_col_group('ssl_', df)
        http_cols = get_col_group('http_', df)
        weird_cols = get_col_group('weird_', df)
        df.drop(columns=ssl_cols + http_cols + weird_cols, inplace=True)
        df.replace('-', pd.NA, inplace=True)
        df = cleanup_protocol_orphans(df, 'dns_')
        #df = cleanup_protocol_orphans(df, 'http_')
        df = df.drop(columns=['dst_ip', 'src_ip', 'dst_port', 'src_port'])
        df['has_dns'] = ~df['dns_query'].isnull()
        self.binary_sparse_cols = [col for col in self.binary_sparse_cols if col in df.columns]
        self.continuous_cols = [col for col in self.continuous_cols if col in df.columns]
        self.ohe_cols = [col for col in self.ohe_cols if col in df.columns]
        self.categorical_cols = self.ohe_cols + self.binary_sparse_cols
        for col in self.binary_sparse_cols:
            if col in df.columns:
                df[col] = df[col].notna().astype(int)
        imputer = ContextAwareImputer(self.continuous_cols, self.categorical_cols)
        df = imputer.fit_transform(df)
        freq_dns = df['dns_rcode'].value_counts(normalize=True)
        df['dns_rcode_freq'] = df['dns_rcode'].map(freq_dns).fillna(0)
        freq_service = df['service'].value_counts(normalize=True)
        df['service_freq'] = df['service'].map(freq_service).fillna(0)
        df, new_features = self._add_engineered_features(df)
        self.continuous_cols += new_features
        self.continuous_cols = [col for col in self.continuous_cols if col in df.columns]
        #log transform skewed continuous features
        for col in self.continuous_cols:
            df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
        df[self.continuous_cols] = self.scaler.fit_transform(df[self.continuous_cols])
        # drop categorical columns
        if train:
            counts = df['conn_state'].value_counts(normalize=True) # normalize=True gibt % (0 bis 1)
            freq_map = counts.to_dict()
            freq_map['default'] = 0.0 # for unknown categories
            self.conn_state_freq_map = freq_map
        else:
            freq_map = self.conn_state_freq_map
        default_val = freq_map.get('default', 0.0)
        df['conn_state_freq'] = df['conn_state'].map(freq_map).fillna(default_val)
        df = df.drop(columns=self.ohe_cols)
        return df
    
    def _add_engineered_features(self, df):
        df = df.copy()
        # 1. Packet Ratio (Paketverhältnis)
        # df['ratio_bytes'] = df['src_bytes'] / (df['dst_bytes'] + 1)
        df['dns_pkts_ratio'] = df['has_dns'] * (df['dst_pkts'] / (df['src_pkts'] + 1))
        df['ratio_pkts'] = df['src_pkts'] / (df['dst_pkts'] + 1)
        # 2. Payload Density (Inhaltsschwere)
        # Hier sehen wir, ob Pakete leer sind (Scanning) oder voll (Exfil)
        df['mean_bytes_src'] = df['src_bytes'] / (df['src_pkts'] + 1)
        df['dns_bytes_fraction'] = df['dst_bytes'] / (df['src_bytes'] + df['dst_bytes'] + 1)
        # 3. Velocity (Aggressivität)
        # Wie schnell passiert das Ganze?
        df['dst_pkts_per_sec'] = df['dst_pkts'] / (df['duration'] + 1e-3)
        
        new_features = [
            'dns_pkts_ratio', 'ratio_pkts',
            'mean_bytes_src', 'dns_bytes_fraction',
            'dst_pkts_per_sec'
        ]
            
        return df, new_features
    
class CatBoostPreprocessor:
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

    def preprocess(self, df: pd.DataFrame, train=True) -> pd.DataFrame:
        df = df.copy()
        df.replace('-', "na", inplace=True)
        
        # Grundreinigung bleibt gleich
        df = cleanup_protocol_orphans(df, 'dns_')
        df = cleanup_protocol_orphans(df, 'http_')
        df = df.drop(columns=['dst_ip', 'src_ip', 'dst_port', 'src_port', 
                              "src_ip_bytes", "dst_ip_bytes", "dns_qclass", "dns_qtype"])

        # WICHTIG: Kategorische Spalten explizit als String/Kategorie markieren
        # CatBoost mag keine gemischten Typen oder echte NaNs in Kategorien während des Fits
        # for col in self.categorical_cols:
        #     if col in df.columns:
        #         df[col] = df[col].astype(str).fillna('missing')

        return df

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

def plot_score_distribution(isoforest, X_test, Y_test, title):
    scores = isoforest.decision_function(X_test)
    plt.figure(figsize=(10, 6))
    plt.hist(scores[Y_test == 0], bins=50, label='Normal', alpha=0.5, color='blue', density=True)
    plt.hist(scores[Y_test == 1], bins=50, label='Angriff', alpha=0.5, color='red', density=True)
    plt.axvline(x=0, color='black', linestyle='--', label='Standard-Threshold')
    plt.legend()
    plt.title(f"Verteilung der Anomaly-Scores - {title}")
    plt.show()

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix using seaborn heatmap"""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()