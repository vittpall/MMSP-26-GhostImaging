"""
Generate measurement patterns for ghost imaging simulations.

Usage:
    python generate_patterns.py --type hadamard --num_patterns 188 --image_size 256
    python generate_patterns.py --type speckle  --num_patterns 188 --image_size 256
    python generate_patterns.py --type hadamard --num_patterns 512 --image_size 32 --out data/hadamard_32.pt
"""

import argparse
import os
import numpy as np
import torch


def make_hadamard_s_patterns(image_size: int, num_patterns: int,
                              row_start: int = 1) -> np.ndarray:
    """
    Return M rows of the binary Hadamard S-matrix for N = image_size² pixels.

    H_S[i, j] = (H[i,j] + 1) / 2  where  H[i,j] = (-1)^popcount(i & j)
    Equivalently: H_S[i,j] = 1 - parity(popcount(i & j))

    Each row has exactly N/2 ones (for rows i >= 1), so each measurement
    illuminates exactly half the pixels — maximising signal per shot.

    Args:
        image_size:   Side length of the square image (must give N = power of 2).
        num_patterns: M — number of rows to select.
        row_start:    First row index (default 1 skips the DC row).

    Returns:
        np.ndarray of shape (M, image_size, image_size), dtype float32, values in {0,1}.
    """
    N = image_size * image_size
    assert (N & (N - 1)) == 0, f"N={N} must be a power of 2 for the Walsh-Hadamard matrix"
    assert row_start + num_patterns - 1 < N, \
        f"Requested rows {row_start}..{row_start+num_patterns-1} exceed matrix size {N}"

    rows = np.arange(row_start, row_start + num_patterns, dtype=np.int64)  # (M,)
    cols = np.arange(N, dtype=np.int64)                                     # (N,)

    # Compute parity of popcount(row & col) via the parallel XOR trick.
    # Valid for values up to 2^32 (here values ≤ N-1 ≤ 65535 < 2^16).
    x = rows[:, None] & cols[None, :]   # (M, N) — broadcast AND
    x ^= x >> 1
    x ^= x >> 2
    x ^= x >> 4
    x ^= x >> 8
    x ^= x >> 16
    parity = (x & 1).astype(np.float32)  # 0 → even popcount (+1), 1 → odd (−1)

    # H_S = 1 when H = +1 (parity even), 0 when H = −1 (parity odd)
    h_s = 1.0 - parity                  # (M, N), values in {0, 1}
    return h_s.reshape(num_patterns, image_size, image_size)


def make_speckle_patterns(image_size: int, num_patterns: int,
                          sparsity: float = 0.5, seed: int = 42) -> np.ndarray:
    """
    Generate random binary speckle patterns matching the existing format.

    Args:
        image_size:   Side length of the square image.
        num_patterns: M — number of patterns.
        sparsity:     Fraction of pixels set to 1 (default 0.5).
        seed:         RNG seed for reproducibility.

    Returns:
        np.ndarray of shape (M, image_size, image_size), dtype float32, values in {0,1}.
    """
    rng = np.random.default_rng(seed)
    patterns = (rng.random((num_patterns, image_size, image_size)) < sparsity).astype(np.float32)
    return patterns


def main():
    parser = argparse.ArgumentParser(description='Generate ghost imaging measurement patterns')
    parser.add_argument('--type', choices=['hadamard', 'speckle'], default='hadamard',
                        help='Pattern type (default: hadamard)')
    parser.add_argument('--num_patterns', type=int, default=188,
                        help='Number of measurement patterns M (default: 188)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image side length in pixels (default: 256)')
    parser.add_argument('--row_start', type=int, default=1,
                        help='First Hadamard row index — 0 is DC, 1 skips it (default: 1)')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='Speckle density — fraction of lit pixels (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for speckle patterns (default: 42)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output path (default: data/{type}_pattern.pt)')
    args = parser.parse_args()

    out_path = args.out or f'data/{args.type}_pattern.pt'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    N = args.image_size ** 2
    print(f"Generating {args.type} patterns: M={args.num_patterns}, "
          f"image_size={args.image_size} (N={N})")

    if args.type == 'hadamard':
        if (N & (N - 1)) != 0:
            raise ValueError(f"N={N} must be a power of 2 for Hadamard patterns. "
                             f"Use image_size = 32, 64, 128, or 256.")
        patterns = make_hadamard_s_patterns(args.image_size, args.num_patterns,
                                            row_start=args.row_start)
        ones_per_row = patterns.reshape(args.num_patterns, -1).sum(axis=1)
        print(f"  Ones per row — mean: {ones_per_row.mean():.1f}, "
              f"expected: {N/2:.1f}  (should be N/2 = {N//2})")
    else:
        patterns = make_speckle_patterns(args.image_size, args.num_patterns,
                                         sparsity=args.sparsity, seed=args.seed)
        ones_per_row = patterns.reshape(args.num_patterns, -1).sum(axis=1)
        print(f"  Ones per row — mean: {ones_per_row.mean():.1f}, "
              f"expected: {int(N * args.sparsity)}")

    tensor = torch.tensor(patterns)  # float32, (M, H, W)
    torch.save(tensor, out_path)
    print(f"Saved: {out_path}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")

    # Quick SNR estimate: multiplex gain for Hadamard vs speckle
    if args.type == 'hadamard':
        import math
        gain = math.sqrt(args.num_patterns / 2)
        print(f"\nExpected multiplex SNR gain over sparse speckle: √(M/2) ≈ {gain:.2f}×")


if __name__ == '__main__':
    main()
