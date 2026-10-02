"""
Pure Python RNA secondary structure prediction using Nussinov algorithm.
Implements dot-bracket notation output and conversion to RNAshapes alphabet (F/T/I/H/M/S).
"""

import gzip
import math
from typing import List, Tuple

# Base pairing rules (including wobble G-U)
PAIRING = {
    ('A', 'U'): True, ('U', 'A'): True,
    ('C', 'G'): True, ('G', 'C'): True,
    ('G', 'U'): True, ('U', 'G'): True,
}

def can_pair(b1: str, b2: str) -> bool:
    return (b1, b2) in PAIRING

def nussinov_dp(seq: str, min_loop_len: int = 3) -> List[List[float]]:
    """
    Nussinov algorithm: fill DP table for RNA secondary structure prediction.
    Returns DP table where dp[i][j] = max number of base pairs in seq[i:j+1].
    """
    n = len(seq)
    dp = [[0.0] * n for _ in range(n)]

    # Fill DP table (lower triangle, i < j)
    for length in range(1, n):
        for i in range(n - length):
            j = i + length
            # Option 1: i unpaired
            best = dp[i + 1][j]
            # Option 2: j unpaired
            best = max(best, dp[i][j - 1])
            # Option 3: i paired with j (if they can pair and loop length >= min_loop_len)
            if can_pair(seq[i], seq[j]) and (j - i - 1) >= min_loop_len:
                pair_score = 1 + (dp[i + 1][j - 1] if i + 1 <= j - 1 else 0)
                best = max(best, pair_score)
            # Option 4: bifurcation
            for k in range(i + 1, j):
                best = max(best, dp[i][k] + dp[k + 1][j])
            dp[i][j] = best
    return dp


def traceback(seq: str, dp: List[List[float]], min_loop_len: int = 3) -> List[int]:
    """
    Trace back through DP table to get the optimal structure.
    Returns: list of same length as seq, where:
        -1 = unpaired
        pair_partner_index = index of paired base
    """
    n = len(seq)
    pairs = [-1] * n

    def tb(i: int, j: int):
        if i >= j:
            return
        if dp[i][j] == dp[i + 1][j]:
            tb(i + 1, j)
            return
        if dp[i][j] == dp[i][j - 1]:
            tb(i, j - 1)
            return
        if can_pair(seq[i], seq[j]) and (j - i - 1) >= min_loop_len:
            if dp[i][j] == 1 + (dp[i + 1][j - 1] if i + 1 <= j - 1 else 0):
                pairs[i] = j
                pairs[j] = i
                tb(i + 1, j - 1)
                return
        for k in range(i + 1, j):
            if dp[i][j] == dp[i][k] + dp[k + 1][j]:
                tb(i, k)
                tb(k + 1, j)
                return

    tb(0, n - 1)
    return pairs


def get_dot_bracket(pairs: List[int], n: int) -> str:
    """
    Convert pairing information to dot-bracket notation.
    Uses simple ( ) for paired, . for unpaired.
    For nested pairs, uses < > etc. but we'll keep simple ( ) by using a stack.
    """
    # Actually, dot-bracket uses only ( and ) - nested pairs are just more ( )
    # We need to count nesting level and use different brackets, but standard is just ( )
    # For simplicity, use ( ) for all pairs (standard dot-bracket)
    result = ['.'] * n
    for i in range(n):
        if pairs[i] > i:  # i is 5' side of pair
            result[i] = '('
        elif pairs[i] >= 0:  # i is 3' side of pair
            result[i] = ')'
    return ''.join(result)


def dot_bracket_to_shapes(db: str) -> str:
    """
    Convert dot-bracket notation to RNAshapes alphabet:
    F = unpaired (flat)
    S = stem (paired)
    H = hairpin loop
    I = interior loop
    M = multi-loop
    T = two-loop (recognized by RNAshapes)

    This is a SIMPLIFIED approximation. True RNAshapes requires thermodynamic params.
    Our approximation:
    - ( or ) -> 'S' (stem)
    - . -> check context:
        - Isolated . or short run -> 'H' (hairpin-like)
        - . flanked by stems -> 'I' (interior loop-like)
        - Long run of . -> 'F' (flexible/unstructured)
    """
    n = len(db)
    result = []

    for i, c in enumerate(db):
        if c == '(' or c == ')':
            result.append('S')  # Stem
        else:
            # Unpaired - classify based on context
            # Count neighboring paired bases
            left_paired = 0
            right_paired = 0
            for j in range(max(0, i - 5), i):
                if db[j] in '()':
                    left_paired += 1
            for j in range(i + 1, min(n, i + 6)):
                if db[j] in '()':
                    right_paired += 1

            if left_paired >= 2 and right_paired >= 2:
                result.append('I')  # Interior loop (surrounded by stems)
            elif left_paired >= 1 and right_paired >= 1:
                result.append('T')  # Two-loop
            elif left_paired >= 1 or right_paired >= 1:
                result.append('H')  # Hairpin-like (next to stem)
            else:
                result.append('F')  # Flat/unstructured

    return ''.join(result)


def predict_structure(seq: str, min_loop_len: int = 3) -> Tuple[str, str]:
    """
    Predict RNA secondary structure for a single sequence.
    Returns: (dot_bracket_string, shapes_string)
    """
    seq = seq.upper().replace('T', 'U')  # RNA uses U not T
    # Filter invalid characters
    seq = ''.join(c for c in seq if c in 'ACGU')

    if len(seq) < 2:
        db = '.' * len(seq)
        shapes = 'F' * len(seq)
        return db, shapes

    dp = nussinov_dp(seq, min_loop_len)
    pairs = traceback(seq, dp, min_loop_len)
    db = get_dot_bracket(pairs, len(seq))
    shapes = dot_bracket_to_shapes(db)
    return db, shapes


def process_fasta_file(input_path: str, output_path: str = None,
                       is_gz: bool = None):
    """
    Process a FASTA file (possibly gzipped) and output structure file.
    Output format: one line per sequence, in shapes alphabet (F/T/I/H/M/S).

    If output_path is None, writes to input_path with .structure suffix (gzipped).
    """
    if is_gz is None:
        is_gz = input_path.endswith('.gz')

    sequences = []
    headers = []

    opener = gzip.open if is_gz else open
    mode = 'rt' if is_gz else 'r'

    with opener(input_path, mode) as f:
        seq = ''
        header = ''
        for line in f:
            line = line.strip()
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                headers.append(line)
                seq = ''
            else:
                seq = seq + line
        if seq:
            sequences.append(seq)

    # Predict structure for each sequence
    structures_shapes = []
    for seq in sequences:
        _, shapes = predict_structure(seq)
        structures_shapes.append(shapes)

    # Write output
    if output_path is None:
        if input_path.endswith('.gz'):
            output_path = input_path.replace('.fa.gz', '.structure.gz').replace('.fasta.gz', '.structure.gz')
        else:
            output_path = input_path.replace('.fa', '.structure.gz').replace('.fasta', '.structure.gz')

    with gzip.open(output_path, 'wt') as f:
        for s in structures_shapes:
            f.write(s + '\n')

    print(f"Wrote {len(structures_shapes)} structures to {output_path}")
    return output_path


def generate_all_structures(data_dir: str):
    """
    Generate structure.gz files for all 24 RBP datasets.
    data_dir: path to data/clip_24rbp/
    For each RBP subdirectory with train.fa.gz and test.fa.gz,
    generate corresponding structure.gz files.
    """
    import os
    rbp_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for rbp_name in rbp_dirs:
        rbp_path = os.path.join(data_dir, rbp_name)

        for split in ['train', 'test']:
            fa_path = os.path.join(rbp_path, f'{split}.fa.gz')
            if not os.path.exists(fa_path):
                print(f"  Skipping {fa_path} (not found)")
                continue

            # Output: for PreRBP format, we need structure.gz in a specific directory structure
            # PreRBP expects: datasets/clip/{dataset_name}/30000/training_sample_0/structure.gz
            # But for simplicity, we'll output alongside the .fa.gz files
            out_path = os.path.join(rbp_path, f'{split}.structure.gz')
            print(f"Processing {fa_path}...")
            process_fasta_file(fa_path, out_path)

    print("\nDone generating structure files for all 24 RBP datasets.")


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else None
        process_fasta_file(input_file, output_file)
    else:
        # Test with a sample sequence
        test_seq = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
        db, shapes = predict_structure(test_seq)
        print(f"Sequence: {test_seq[:50]}...")
        print(f"Dot-bracket: {db[:50]}...")
        print(f"Shapes: {shapes[:50]}...")
        print(f"Paired fraction: {db.count('(') / len(db):.2%}")
