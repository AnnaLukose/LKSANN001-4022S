# Pixel scale for the comparison scripts, read from the data (truth/config.json).
# Converts settings given at the reference scale: lengths x K, areas x K^2, K = 21.3 / um per px.
# Used by: hybrid_segment.py, piv_compare.py, cls_compare.py, cls_compare2.py.
import glob
import json
import math
import os

REF_UM_PER_PX = 21.3             # reference scale (GelCam, Song et al. 2025)
REF_MIN_AREA_PX = 50             # Song's area filter at that scale (~170 um ESD)

UM_PER_PX = 13.0
K = REF_UM_PER_PX / UM_PER_PX
MIN_AREA = 50
MIN_AREA_MODE = 'pixels'
CANVAS, FIT = 80, 67             # classifier crop size and the largest particle kept at native scale (px)
SOURCE = 'default'               # where the current values came from

_HOOKS = []


def length(px_at_ref):
    """A length in reference pixels -> pixels now."""
    return px_at_ref * K


def ilength(px_at_ref, minimum=1):
    """The same, rounded to a whole number of pixels (at least `minimum`)."""
    return max(minimum, int(round(px_at_ref * K)))


def odd(px_at_ref, minimum=3):
    """A window size that must be odd (kernels, Lucas-Kanade windows)."""
    n = ilength(px_at_ref, minimum)
    return n if n % 2 else n + 1


def mult8(px_at_ref, minimum=16):
    """A window size rounded to a multiple of 8 (PIV interrogation windows)."""
    return max(minimum, int(round(px_at_ref * K / 8)) * 8)


def area(px_at_ref):
    """An area in reference pixels -> pixels now."""
    return px_at_ref * K * K


def iarea(px_at_ref, minimum=1):
    return max(minimum, int(round(px_at_ref * K * K)))


def um(px):
    """Pixels at the current resolution -> micrometres."""
    return px * UM_PER_PX


def register(fn):
    """Called by every *13 module with a function that refreshes its settings."""
    _HOOKS.append(fn)
    fn()


def configure(um_per_px=13.0, min_area=None, min_area_mode='pixels', canvas=None, source='configure()'):
    """Set the resolution for this session (and refresh every loaded *13 module)."""
    global UM_PER_PX, K, MIN_AREA, MIN_AREA_MODE, CANVAS, FIT, SOURCE
    UM_PER_PX = float(um_per_px)
    K = REF_UM_PER_PX / UM_PER_PX
    MIN_AREA_MODE = min_area_mode
    if min_area is not None:
        MIN_AREA = int(min_area)
    elif min_area_mode == 'physical':
        MIN_AREA = max(4, int(round(REF_MIN_AREA_PX * K * K)))
    else:
        MIN_AREA = REF_MIN_AREA_PX
    CANVAS = int(canvas) if canvas else int(min(max(round(48 * K / 8) * 8, 32), 96))
    FIT = int(round(CANVAS * 40 / 48))
    SOURCE = source
    for fn in _HOOKS:
        fn()
    return summary()


def read_resolution(path):
    """The 'resolution' block the pipeline saved with a dataset (path = a dataset folder or one sequence folder)."""
    for f in [os.path.join(path, 'truth', 'config.json')] + sorted(glob.glob(os.path.join(path, '*', 'truth',
                                                                                          'config.json'))):
        if os.path.exists(f):
            meta = json.load(open(f))
            res = meta.get('resolution') or {}
            if 'um_per_px' not in res and 'um_per_px' in meta.get('cfg', {}):
                res = dict(um_per_px=meta['cfg']['um_per_px'],
                           min_area_px=meta['cfg'].get('min_detectable_area_px'))
            if 'um_per_px' in res:
                return res, f
    return None, None


def configure_from_data(path, fallback=13.0):
    """Use the pixel size, smallest particle and crop size stored with the data (the pipeline truth/config.json)."""
    res, f = read_resolution(path)
    if res is None:
        print(f'no resolution found under {path}: using {fallback} um/px')
        return configure(fallback, source='fallback')
    return configure(res['um_per_px'], min_area=res.get('min_area_px'), min_area_mode=res.get('min_area_mode', 'pixels'),
                     canvas=res.get('canvas_px'), source=f)


def summary():
    return dict(um_per_px=UM_PER_PX, scale_k=round(K, 4), min_area_px=MIN_AREA, min_area_mode=MIN_AREA_MODE,
                min_esd_um=round(2 * math.sqrt(MIN_AREA / math.pi) * UM_PER_PX, 1), canvas_px=CANVAS, fit_px=FIT,
                source=SOURCE)
