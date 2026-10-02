"""
=============================================================
circ_preprocess.py — circRNA-RBP 交互数据预处理
功能：加载 CircInteractome 数据集（37 RBP），生成训练/测试数据
=============================================================
"""
import os, json, re
import numpy as np
from pathlib import Path
from collections import Counter

BASE_DIR = Path(os.environ.get("RBP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CIRC_DIR = BASE_DIR / "data" / "circinteractome" / "circRNA-RBP"
PROTEIN_JSON = BASE_DIR / "data" / "rbp_proteins" / "rbp_sequences.json"
OUT_DIR = BASE_DIR / "data" / "circ_processed"
OUT_DIR.mkdir(exist_ok=True)

# RBP name mapping (circRNA dataset name → protein JSON key)
NAME_MAP = {
    'HUR': 'ELAVL1', 'SFRS1': 'SFRS1', 'AGO2': 'AGO2',
    'EWSR1': 'EWSR1', 'FUS': 'FUS', 'IGF2BP1': 'IGF2BP1',
    'IGF2BP2': 'IGF2BP2', 'IGF2BP3': 'IGF2BP3', 'MOV10': 'MOV10',
    'PUM2': 'PUM2', 'QKI': 'QKI', 'TAF15': 'TAF15',
    'TDP43': 'TDP43', 'U2AF65': 'U2AF65', 'HNRNPC': 'HNRNPC',
}

# UniProt-fetched sequence gene-name mapping
UNIPROT_MAP = {
    'AGO1': 'AGO1', 'AGO3': 'AGO3', 'ALKBH5': 'ALKBH5',
    'AUF1': 'HNRNPD', 'C17ORF85': 'NCBP3', 'C22ORF28': 'RTCB',
    'CAPRIN1': 'CAPRIN1', 'DGCR8': 'DGCR8', 'EIF4A3': 'EIF4A3',
    'FMRP': 'FMR1', 'FOX2': 'RBFOX2', 'FXR1': 'FXR1', 'FXR2': 'FXR2',
    'LIN28A': 'LIN28A', 'LIN28B': 'LIN28B', 'METTL3': 'METTL3',
    'PTB': 'PTBP1', 'TIA1': 'TIA1', 'TIAL1': 'TIAL1',
    'TNRC6': 'TNRC6A', 'WTAP': 'WTAP', 'ZC3H7B': 'ZC3H7B',
}

# Nucleotide vocab
NUC_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
MAX_SEQ_LEN = 501  # As used in iCircRBP-DHN paper


def one_hot_encode(seq, max_len=MAX_SEQ_LEN):
    """One-hot encode nucleotide sequence (A,C,G,T) → (max_len, 4)"""
    arr = np.zeros((max_len, 4), dtype=np.float32)
    for i, nt in enumerate(seq[:max_len].upper()):
        idx = NUC_VOCAB.get(nt, 4)
        if idx < 4:
            arr[i, idx] = 1.0
    return arr


def load_protein_sequences():
    """Load all protein sequences from JSON + FASTA"""
    proteins = {}
    # Load existing JSON
    if PROTEIN_JSON.exists():
        with open(PROTEIN_JSON) as f:
            proteins = json.load(f)
    # Load FASTA from UniProt
    fasta_path = BASE_DIR / "data" / "rbp_proteins" / "missing_rbp_sequences.fasta"
    if fasta_path.exists():
        fasta_text = fasta_path.read_text()
        entries = fasta_text.strip().split('>sp|')
        for entry in entries:
            if not entry.strip():
                continue
            lines = entry.strip().split('\n')
            header = lines[0]
            seq = ''.join(lines[1:])
            # Extract gene name
            gn_match = re.search(r'GN=(\w+)', header)
            if gn_match:
                gn = gn_match.group(1)
                proteins[gn] = seq

    return proteins


def load_circ_data():
    """Load all circRNA-RBP datasets and generate labeled samples"""
    circ_rbps = sorted([d.name for d in CIRC_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(circ_rbps)} RBP directories")

    proteins = load_protein_sequences()
    print(f"Loaded {len(proteins)} protein sequences")

    all_samples = []
    rbp_counts = {}

    for rbp in circ_rbps:
        pos_file = CIRC_DIR / rbp / "positive"
        neg_file = CIRC_DIR / rbp / "negative"

        # Get protein sequence
        mapped = NAME_MAP.get(rbp, rbp)
        if mapped not in proteins:
            mapped2 = UNIPROT_MAP.get(rbp, rbp)
            if mapped2 not in proteins:
                print(f"  WARNING: No protein sequence for {rbp} (tried {mapped}, {mapped2})")
                continue
            mapped = mapped2
        protein_seq = proteins[mapped]

        pos_seqs = []
        neg_seqs = []

        for fpath, seqs in [(pos_file, pos_seqs), (neg_file, neg_seqs)]:
            if fpath.exists():
                raw = fpath.read_text()
                current_seq = ""
                for line in raw.split('\n'):
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq:
                            seqs.append(current_seq)
                            current_seq = ""
                    else:
                        current_seq += line
                if current_seq:
                    seqs.append(current_seq)

        n_pos = len(pos_seqs)
        n_neg = len(neg_seqs)
        rbp_counts[rbp] = (n_pos, n_neg)

        # Balance: take equal number from positive and negative
        n_samples = min(n_pos, n_neg, 5000)  # Cap at 5000 per class
        if n_samples < 100:
            print(f"  WARNING: {rbp} only {n_samples} samples, skipping")
            continue

        np.random.seed(42)
        pos_idx = np.random.choice(n_pos, n_samples, replace=False)
        neg_idx = np.random.choice(n_neg, n_samples, replace=False)

        for i in pos_idx:
            all_samples.append({
                'rbp': rbp,
                'protein': protein_seq,
                'circrna': pos_seqs[i],
                'label': 1,
            })
        for i in neg_idx:
            all_samples.append({
                'rbp': rbp,
                'protein': protein_seq,
                'circrna': neg_seqs[i],
                'label': 0,
            })

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"RBP distribution:")
    for rbp, (p, n) in sorted(rbp_counts.items()):
        print(f"  {rbp:<12} pos={p:<6} neg={n:<6}")

    return all_samples


def preprocess():
    samples = load_circ_data()

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(samples)

    # Encode all sequences
    print("\nEncoding sequences...")
    X_rna = np.stack([one_hot_encode(s['circrna']) for s in samples])
    y = np.array([s['label'] for s in samples], dtype=np.float32)
    rbp_list = [s['rbp'] for s in samples]

    # Save
    np.save(OUT_DIR / "X_circrna.npy", X_rna)
    np.save(OUT_DIR / "y_circrna.npy", y)
    with open(OUT_DIR / "rbp_labels.json", 'w') as f:
        json.dump(rbp_list, f)

    # Save unique RBPs for protein embedding lookups
    unique_rbps = sorted(set(rbp_list))
    proteins = load_protein_sequences()
    rbp_protein_map = {}
    for rbp in unique_rbps:
        mapped = NAME_MAP.get(rbp, rbp)
        if mapped not in proteins:
            mapped = UNIPROT_MAP.get(rbp, rbp)
        rbp_protein_map[rbp] = proteins.get(mapped, '')

    with open(OUT_DIR / "rbp_proteins.json", 'w') as f:
        json.dump(rbp_protein_map, f)

    print(f"\nSaved:")
    print(f"  X_circrna.npy: {X_rna.shape}")
    print(f"  y_circrna.npy: {y.shape}")
    print(f"  Positive samples: {int(y.sum())}")
    print(f"  Negative samples: {int(len(y) - y.sum())}")
    print(f"  Unique RBPs: {len(unique_rbps)}")


if __name__ == '__main__':
    preprocess()
