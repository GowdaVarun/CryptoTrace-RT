#!/usr/bin/env python3
"""
Extract features from ELF binaries for crypto detection.

Based on EMBER feature engineering (Anderson & Roth 2018) adapted for ELF:
1. Byte histogram (256-dim → compressed to statistical features)
2. Byte-entropy histogram (16x16=256-dim → compressed)
3. Section-level entropy & size features
4. Import/library-based crypto signals
5. Structural ELF header features
6. String-based features
"""

import os
import json
import math
import struct
import numpy as np
import pandas as pd
from collections import Counter
import lief
import warnings
warnings.filterwarnings('ignore')

def shannon_entropy(data: bytes) -> float:
    """Shannon entropy in bits/byte"""
    if not data or len(data) == 0:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def byte_histogram(data: bytes) -> np.ndarray:
    """256-bin normalized byte value histogram"""
    hist = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist

def byte_entropy_histogram(data: bytes, n_entropy_bins=16, n_byte_bins=16, 
                            window=2048, step=1024) -> np.ndarray:
    """
    Joint p(H, X) histogram — EMBER's key feature.
    Computes entropy of sliding windows and bins (entropy, byte_value) pairs.
    Returns flattened n_entropy_bins x n_byte_bins array.
    """
    hist = np.zeros((n_entropy_bins, n_byte_bins), dtype=np.float64)
    data_array = np.frombuffer(data, dtype=np.uint8)
    
    for i in range(0, max(1, len(data) - window), step):
        chunk = data_array[i:i+window]
        h = shannon_entropy(bytes(chunk))
        h_bin = min(int(h / 8.0 * n_entropy_bins), n_entropy_bins - 1)
        
        # Bin the byte values in this window
        byte_bins = chunk // (256 // n_byte_bins)
        byte_bins = np.minimum(byte_bins, n_byte_bins - 1)
        for bb in byte_bins:
            hist[h_bin, bb] += 1
    
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.flatten()

def extract_strings(data: bytes, min_len=4) -> list:
    """Extract printable ASCII strings from binary"""
    strings = []
    current = []
    for byte in data:
        if 32 <= byte <= 126:
            current.append(chr(byte))
        else:
            if len(current) >= min_len:
                strings.append(''.join(current))
            current = []
    if len(current) >= min_len:
        strings.append(''.join(current))
    return strings

def compression_ratio(data: bytes) -> float:
    """Estimate compression ratio using simple byte counting"""
    import zlib
    if len(data) == 0:
        return 1.0
    compressed = zlib.compress(data, 6)
    return len(data) / max(len(compressed), 1)

# Crypto-related import prefixes and library names
CRYPTO_IMPORT_PREFIXES = [
    'EVP_', 'AES_', 'RSA_', 'SHA', 'HMAC', 'BN_', 'EC_', 'DES_',
    'MD5', 'MD4', 'SHA1', 'SHA256', 'SHA512', 'PKCS',
    'gcrypt_', 'nettle_', 'mbedtls_',
    'ssl_', 'SSL_', 'TLS_',
    'RAND_', 'OPENSSL_', 'PEM_',
    'X509_', 'CRYPTO_', 'ERR_',
    'DSA_', 'DH_', 'ECDSA_', 'ECDH_',
    'aes_', 'sha_', 'rsa_', 'des_',
    'CMAC_', 'HKDF',
]

CRYPTO_LIBRARIES = [
    'libcrypto', 'libssl', 'libgcrypt', 'libmbedcrypto', 'libmbedtls',
    'libnettle', 'libgnutls', 'libsodium', 'libnss', 'libwolfssl',
]

# AES S-box magic bytes for YARA-like scanning
AES_SBOX_START = bytes([0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5])
SHA256_INIT = bytes([0x67, 0xe6, 0x09, 0x6a])  # little-endian H0
SHA1_INIT = bytes([0x67, 0x45, 0x23, 0x01])
DES_SBOX_START = bytes([0x0e, 0x04, 0x0d, 0x01])
MD5_INIT_A = bytes([0x01, 0x23, 0x45, 0x67])

CRYPTO_CONSTANTS = [AES_SBOX_START, SHA256_INIT, SHA1_INIT, DES_SBOX_START, MD5_INIT_A]


def extract_features(binary_path: str) -> dict:
    """Extract all features from a single ELF binary."""
    
    feats = {}
    
    # Read raw bytes
    with open(binary_path, 'rb') as f:
        raw = f.read()
    
    # Parse with LIEF
    binary = lief.parse(binary_path)
    if binary is None:
        return None
    
    # ================================================================
    # 1. GLOBAL FILE FEATURES
    # ================================================================
    feats['file_size'] = len(raw)
    feats['file_entropy'] = shannon_entropy(raw)
    feats['compression_ratio'] = compression_ratio(raw)
    
    # ================================================================
    # 2. BYTE HISTOGRAM STATISTICS (derived from 256-dim histogram)
    # ================================================================
    bhist = byte_histogram(raw)
    feats['byte_hist_mean'] = np.mean(bhist)
    feats['byte_hist_std'] = np.std(bhist)
    feats['byte_hist_max'] = np.max(bhist)
    feats['byte_hist_min'] = np.min(bhist)
    feats['byte_hist_skew'] = float(pd.Series(bhist).skew())
    feats['byte_hist_kurtosis'] = float(pd.Series(bhist).kurtosis())
    # Uniformity score (how close to uniform distribution — crypto tends toward uniform)
    feats['byte_hist_uniformity'] = 1.0 - np.std(bhist) / (1.0/256 + 1e-9)
    # Number of zero-frequency bytes
    feats['byte_hist_zero_count'] = int(np.sum(bhist == 0))
    # Top-10 byte concentration
    sorted_hist = np.sort(bhist)[::-1]
    feats['byte_hist_top10_mass'] = float(np.sum(sorted_hist[:10]))
    feats['byte_hist_top50_mass'] = float(np.sum(sorted_hist[:50]))
    
    # ================================================================
    # 3. BYTE-ENTROPY HISTOGRAM STATISTICS (16x16 joint distribution)
    # ================================================================
    beh = byte_entropy_histogram(raw)
    feats['beh_mean'] = np.mean(beh)
    feats['beh_std'] = np.std(beh)
    feats['beh_max'] = np.max(beh)
    feats['beh_high_entropy_mass'] = float(np.sum(beh[192:]))  # top quarter = high entropy windows
    feats['beh_low_entropy_mass'] = float(np.sum(beh[:64]))    # bottom quarter = low entropy
    feats['beh_nonzero_bins'] = int(np.sum(beh > 0))
    
    # ================================================================
    # 4. SECTION-LEVEL FEATURES
    # ================================================================
    section_names_of_interest = ['.text', '.data', '.rodata', '.bss', '.plt', 
                                  '.got', '.init', '.fini', '.plt.got', '.dynamic']
    
    for sname in section_names_of_interest:
        safe_name = sname.replace('.', '_').lstrip('_')
        section = None
        for s in binary.sections:
            if s.name == sname:
                section = s
                break
        
        if section is not None:
            sec_bytes = bytes(section.content)
            feats[f'sec_{safe_name}_size'] = section.size
            feats[f'sec_{safe_name}_entropy'] = shannon_entropy(sec_bytes) if len(sec_bytes) > 0 else 0.0
            feats[f'sec_{safe_name}_exists'] = 1
        else:
            feats[f'sec_{safe_name}_size'] = 0
            feats[f'sec_{safe_name}_entropy'] = 0.0
            feats[f'sec_{safe_name}_exists'] = 0
    
    feats['num_sections'] = len(list(binary.sections))
    
    # Section entropy statistics
    sec_entropies = []
    sec_sizes = []
    for s in binary.sections:
        if s.size > 0:
            sec_bytes = bytes(s.content)
            if len(sec_bytes) > 0:
                sec_entropies.append(shannon_entropy(sec_bytes))
                sec_sizes.append(s.size)
    
    if sec_entropies:
        feats['sec_entropy_mean'] = np.mean(sec_entropies)
        feats['sec_entropy_max'] = np.max(sec_entropies)
        feats['sec_entropy_std'] = np.std(sec_entropies)
        feats['sec_high_entropy_count'] = sum(1 for e in sec_entropies if e > 7.0)
    else:
        feats['sec_entropy_mean'] = 0
        feats['sec_entropy_max'] = 0
        feats['sec_entropy_std'] = 0
        feats['sec_high_entropy_count'] = 0
    
    if sec_sizes:
        feats['sec_size_mean'] = np.mean(sec_sizes)
        feats['sec_size_max'] = np.max(sec_sizes)
        feats['sec_size_ratio_text_total'] = feats.get('sec_text_size', 0) / max(sum(sec_sizes), 1)
    else:
        feats['sec_size_mean'] = 0
        feats['sec_size_max'] = 0
        feats['sec_size_ratio_text_total'] = 0
    
    # ================================================================
    # 5. IMPORT & LIBRARY FEATURES (KEY for crypto detection)
    # ================================================================
    imported_funcs = [f.name for f in binary.imported_functions]
    libraries = list(binary.libraries)
    
    feats['num_imports'] = len(imported_funcs)
    feats['num_libraries'] = len(libraries)
    
    # Crypto import counting
    crypto_import_count = 0
    crypto_import_categories = set()
    for fname in imported_funcs:
        for prefix in CRYPTO_IMPORT_PREFIXES:
            if fname.startswith(prefix) or fname.lower().startswith(prefix.lower()):
                crypto_import_count += 1
                crypto_import_categories.add(prefix.rstrip('_'))
                break
    
    feats['n_crypto_imports'] = crypto_import_count
    feats['n_crypto_import_categories'] = len(crypto_import_categories)
    feats['crypto_import_ratio'] = crypto_import_count / max(len(imported_funcs), 1)
    
    # Library-based signals
    has_crypto_lib = 0
    crypto_lib_count = 0
    for lib in libraries:
        for clib in CRYPTO_LIBRARIES:
            if clib in lib.lower():
                has_crypto_lib = 1
                crypto_lib_count += 1
                break
    
    feats['has_crypto_library'] = has_crypto_lib
    feats['n_crypto_libraries'] = crypto_lib_count
    
    # ================================================================
    # 6. CRYPTO CONSTANT SCANNING (YARA-like)
    # ================================================================
    crypto_const_hits = 0
    for const in CRYPTO_CONSTANTS:
        if const in raw:
            crypto_const_hits += 1
    feats['crypto_constant_hits'] = crypto_const_hits
    
    # Scan .rodata specifically
    rodata_crypto_hits = 0
    for s in binary.sections:
        if s.name == '.rodata' and s.size > 0:
            rodata_bytes = bytes(s.content)
            for const in CRYPTO_CONSTANTS:
                if const in rodata_bytes:
                    rodata_crypto_hits += 1
    feats['rodata_crypto_hits'] = rodata_crypto_hits
    
    # ================================================================
    # 7. STRUCTURAL / HEADER FEATURES
    # ================================================================
    feats['is_pie'] = int(binary.is_pie)
    feats['has_nx'] = int(binary.has_nx)
    
    # Check if stripped
    has_symtab = any(s.name == '.symtab' for s in binary.sections)
    feats['is_stripped'] = 0 if has_symtab else 1
    
    # Number of exported functions
    feats['num_exports'] = len(list(binary.exported_functions))
    
    # Segments
    feats['num_segments'] = len(list(binary.segments))
    
    # ================================================================
    # 8. STRING FEATURES
    # ================================================================
    strings = extract_strings(raw, min_len=4)
    feats['n_strings'] = len(strings)
    
    # Count crypto-related strings
    crypto_string_keywords = [
        'aes', 'sha', 'rsa', 'encrypt', 'decrypt', 'cipher', 'hash',
        'hmac', 'digest', 'openssl', 'crypto', 'ssl', 'tls', 'certificate',
        'key', 'pkcs', 'x509', 'pem', 'des', 'blowfish', 'chacha',
        'md5', 'signature', 'verify', 'sign', 'nonce', 'iv',
    ]
    crypto_str_count = 0
    for s in strings:
        sl = s.lower()
        if any(kw in sl for kw in crypto_string_keywords):
            crypto_str_count += 1
    
    feats['n_crypto_strings'] = crypto_str_count
    feats['crypto_string_ratio'] = crypto_str_count / max(len(strings), 1)
    
    # String entropy
    if strings:
        str_lens = [len(s) for s in strings]
        feats['avg_string_len'] = np.mean(str_lens)
        feats['max_string_len'] = np.max(str_lens)
    else:
        feats['avg_string_len'] = 0
        feats['max_string_len'] = 0
    
    # ================================================================
    # 9. CODE PATTERN FEATURES  
    # ================================================================
    # XOR instruction density (in .text section) — crypto code is XOR-heavy
    text_section = None
    for s in binary.sections:
        if s.name == '.text':
            text_section = s
            break
    
    if text_section and text_section.size > 0:
        text_bytes = bytes(text_section.content)
        # x86 XOR opcodes: 0x31, 0x33, 0x35 (common XOR variants)
        xor_opcodes = [0x31, 0x33, 0x35, 0x30, 0x32, 0x34]
        xor_count = sum(text_bytes.count(bytes([op])) for op in xor_opcodes)
        feats['text_xor_density'] = xor_count / max(len(text_bytes), 1)
        
        # ROL/ROR opcodes (0xC0, 0xC1 with mod bits)
        rot_opcodes = [0xC0, 0xC1, 0xD0, 0xD1, 0xD2, 0xD3]
        rot_count = sum(text_bytes.count(bytes([op])) for op in rot_opcodes)
        feats['text_rotate_density'] = rot_count / max(len(text_bytes), 1)
        
        feats['text_size'] = len(text_bytes)
        feats['text_entropy'] = shannon_entropy(text_bytes)
    else:
        feats['text_xor_density'] = 0
        feats['text_rotate_density'] = 0
        feats['text_size'] = 0
        feats['text_entropy'] = 0
    
    return feats


def main():
    # Load metadata
    with open("/app/binary_metadata.json") as f:
        metadata = json.load(f)
    
    print(f"Extracting features from {len(metadata)} binaries...")
    
    rows = []
    for i, meta in enumerate(metadata):
        path = meta['binary_path']
        if not os.path.exists(path):
            continue
        
        feats = extract_features(path)
        if feats is None:
            continue
        
        # Add metadata
        feats['binary_name'] = meta['binary_name']
        feats['source_file'] = meta['source']
        feats['label'] = meta['label']
        feats['label_name'] = meta['label_name']
        feats['opt_level'] = meta['opt_level']
        
        rows.append(feats)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(metadata)}")
    
    df = pd.DataFrame(rows)
    
    # Save
    df.to_csv("/app/binary_features.csv", index=False)
    print(f"\nFeature extraction complete:")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {len([c for c in df.columns if c not in ['binary_name','source_file','label','label_name','opt_level']])}")
    print(f"  Crypto:     {(df['label']==1).sum()}")
    print(f"  Non-crypto: {(df['label']==0).sum()}")
    print(f"\nClass distribution:")
    print(df['label_name'].value_counts())
    
    # Quick data audit
    print(f"\n--- Data Audit ---")
    feature_cols = [c for c in df.columns if c not in ['binary_name','source_file','label','label_name','opt_level']]
    print(f"Missing values: {df[feature_cols].isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()}")
    print(f"\nFeature stats (selected):")
    key_feats = ['file_entropy', 'n_crypto_imports', 'has_crypto_library', 
                 'crypto_constant_hits', 'n_crypto_strings', 'sec_rodata_entropy',
                 'text_xor_density', 'compression_ratio']
    for feat in key_feats:
        if feat in df.columns:
            by_label = df.groupby('label')[feat].agg(['mean','std'])
            print(f"  {feat}:")
            print(f"    Non-crypto: mean={by_label.loc[0,'mean']:.4f}, std={by_label.loc[0,'std']:.4f}")
            print(f"    Crypto:     mean={by_label.loc[1,'mean']:.4f}, std={by_label.loc[1,'std']:.4f}")

if __name__ == '__main__':
    main()
