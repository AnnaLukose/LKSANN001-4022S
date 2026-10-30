# Step 5: red channel, static mask and POD background removal (port of Song et al. 2025).
# Output: <seq>/bgrem/bgrem_XXXX.png.
import argparse, glob, os
import numpy as np
import cv2


def pod_background_removal(reds, mask, n_modes=1, chunk=1_000_000):
    """reds: (N, H, W) uint8 red channels."""
    N, H, W = reds.shape
    P = H * W
    M = reds.reshape(N, P)
    m = mask.reshape(P).astype(np.float32)
    # 1. the N x N matrix M M^T, one block of pixels at a time
    G = np.zeros((N, N), np.float64)
    for s in range(0, P, chunk):
        blk = M[:, s:s + chunk].astype(np.float32) * m[s:s + chunk]
        G += blk.astype(np.float64) @ blk.T.astype(np.float64)
    w, U = np.linalg.eigh(G)
    U = U[:, np.argsort(w)[::-1][:n_modes]]                      # leading modes = the static background
    # 2. remove the modes block by block, straight into the uint8 result
    out = np.empty((N, P), np.uint8)
    for s in range(0, P, chunk):
        blk = M[:, s:s + chunk].astype(np.float64) * m[s:s + chunk]
        blk -= U @ (U.T @ blk)
        blk *= m[s:s + chunk]
        out[:, s:s + chunk] = np.clip(np.round(blk), 0, 255).astype(np.uint8)
    return out.reshape(N, H, W)


def run(seq_dir, mask_path=None, n_modes=1, raw_sub='raw', out_sub='bgrem'):
    """Background-remove every raw frame of one sequence."""
    files = sorted(glob.glob(os.path.join(seq_dir, raw_sub, '*.jpg')) + glob.glob(os.path.join(seq_dir, raw_sub, '*.png')))
    if not files:
        raise FileNotFoundError('no frames in ' + os.path.join(seq_dir, raw_sub))
    mask_path = mask_path or os.path.join(seq_dir, 'truth', 'static_mask.png')
    reds = np.stack([cv2.imread(f)[..., 2] for f in files])                 # OpenCV is BGR: [..., 2] = red
    mask = (cv2.imread(mask_path, 0) > 0) if os.path.exists(mask_path) else np.ones(reds.shape[1:], bool)
    bg = pod_background_removal(reds, mask, n_modes)
    od = os.path.join(seq_dir, out_sub)
    os.makedirs(od, exist_ok=True)
    names = []
    for f, im in zip(files, bg):
        n = 'bgrem_' + os.path.splitext(os.path.basename(f))[0].split('_')[-1] + '.png'
        cv2.imwrite(os.path.join(od, n), im)
        names.append(n)
    return names


def is_done(seq_dir):
    """True when every raw frame has its bgrem frame."""
    raw = glob.glob(os.path.join(seq_dir, 'raw', 'frame_*.jpg'))
    return len(raw) > 0 and len(glob.glob(os.path.join(seq_dir, 'bgrem', 'bgrem_*.png'))) == len(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('seq_dir')
    ap.add_argument('--mask')
    ap.add_argument('--modes', type=int, default=1)
    a = ap.parse_args()
    print(len(run(a.seq_dir, a.mask, a.modes)), 'frames written')


if __name__ == '__main__':
    main()
