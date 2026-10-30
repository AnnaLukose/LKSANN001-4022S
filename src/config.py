# Step 1: the pixel scale (13 um/px) and every pixel setting derived from it, plus the particle classes.
# Output: module-level settings read by every other file.
# Used by: every file in src/.
import math
import re

# reference camera: GelCam (Song et al. 2025), 21.3 um/px, 20 min between frames
REF_UM_PER_PX = 21.3
REF_WIDTH, REF_HEIGHT = 2592, 1944
REF_MIN_AREA_PX = 50
INTERVAL_MIN = 20.0

# set by configure()
UM_PER_PX = 13.0                 # pixel size of SnoCam+
K = REF_UM_PER_PX / UM_PER_PX    # lengths x K, areas x K^2
WIDTH, HEIGHT = 4246, 3186
MIN_AREA = REF_MIN_AREA_PX       # px: smallest particle detected and counted
MIN_AREA_MODE = 'pixels'
CANVAS, FIT = 80, 67             # classifier crop size and largest particle kept at native scale (px)


def configure(um_per_px=13.0, keep_fov=True, min_area_mode='pixels', canvas=None, canvas_max=96):
    """Set the pixel scale; keep_fov keeps the 55.2 x 41.4 mm field, 'pixels' keeps the 50 px minimum."""
    global UM_PER_PX, K, WIDTH, HEIGHT, MIN_AREA, MIN_AREA_MODE, CANVAS, FIT
    if min_area_mode not in ('physical', 'pixels'):
        raise ValueError("min_area_mode must be 'physical' or 'pixels'")
    UM_PER_PX = float(um_per_px)
    K = REF_UM_PER_PX / UM_PER_PX
    WIDTH, HEIGHT = (_even(REF_WIDTH * K), _even(REF_HEIGHT * K)) if keep_fov else (REF_WIDTH, REF_HEIGHT)
    MIN_AREA_MODE = min_area_mode
    MIN_AREA = max(4, int(round(area(REF_MIN_AREA_PX)))) if min_area_mode == 'physical' else REF_MIN_AREA_PX
    CANVAS = int(canvas) if canvas else int(min(max(round(48 * K / 8) * 8, 32), canvas_max))
    FIT = int(round(CANVAS * 40 / 48))
    return summary()


def _even(x):
    return int(2 * round(x / 2))


def length(px_at_ref):
    return px_at_ref * K


def area(px_at_ref):
    return px_at_ref * K * K


def ilength(px_at_ref, minimum=1):
    return max(minimum, int(round(length(px_at_ref))))


def iarea(px_at_ref, minimum=1):
    return max(minimum, int(round(area(px_at_ref))))


def esd_um(area_px):
    """ESD (um) of an area in pixels."""
    return 2 * math.sqrt(max(area_px, 0) / math.pi) * UM_PER_PX


def summary():
    return dict(um_per_px=UM_PER_PX, scale_k=round(K, 4), width=WIDTH, height=HEIGHT,
                fov_mm=(round(WIDTH * UM_PER_PX / 1000, 1), round(HEIGHT * UM_PER_PX / 1000, 1)),
                min_area_px=MIN_AREA, min_area_mode=MIN_AREA_MODE, min_esd_um=round(esd_um(MIN_AREA), 1),
                canvas_px=CANVAS, canvas_mm=round(CANVAS * UM_PER_PX / 1000, 2))


def data_name(um_per_px=None, prefix='run'):
    """Dataset folder name, e.g. run_13um."""
    u = UM_PER_PX if um_per_px is None else um_per_px
    return f'{prefix}_{u:g}um'


# pixels per micron of Durkin's microscope cameras (from her processing scripts)
PX_PER_UM = {
    'stingray': {7: 0.065, 20: 0.187, 40: 0.375, 80: 0.708, 115: 1.116},
    'lumenera': {7: 0.1268, 20: 0.364, 50: 0.928, 115: 2.179},
}
CRUISE_CAMERA = {'EN572': 'stingray', 'EN581': 'stingray',
                 'FK170124': 'lumenera', 'FK': 'lumenera', 'RR1813': 'lumenera', 'RR': 'lumenera'}
_MAG = re.compile(r'^(\d+)x$', re.I)


def parse_name(fname):
    """'EN581_NBST_7x_bf_8_88.jpg' -> dict(cruise, platform, mag, illum, micrograph, number, sample)."""
    stem = fname.rsplit('.', 1)[0]
    tok = stem.split('_')
    plat = tok[1] if len(tok) > 1 and not _MAG.match(tok[1]) else ''
    out = dict(cruise=tok[0], platform=plat, mag=None, illum='bf',
               micrograph=re.sub(r'(\d+)[a-z]$', r'\1', stem.rsplit('_', 1)[0]), number=tok[-1])
    out['sample'] = f"{out['cruise']}_{plat}" if plat else out['cruise']
    for t in tok:
        m = _MAG.match(t)
        if m:
            out['mag'] = int(m.group(1))
        if t.lower() in ('bf', 'brightfield'):
            out['illum'] = 'bf'
        if t.lower() in ('obl', 'oblique', 'ob'):
            out['illum'] = 'obl'
    return out


def um_per_px_for(cruise, mag, override=None):
    """Source pixel size of a crop (None if the camera is unknown)."""
    if override:
        for key in (cruise, '*'):
            if key in override and mag in override[key]:
                return float(override[key][mag])
    cam = next((v for k, v in CRUISE_CAMERA.items() if cruise.upper().startswith(k)), None)
    if cam is None or mag not in PX_PER_UM[cam]:
        return None
    return 1.0 / PX_PER_UM[cam][mag]


# classes: Durkin's archive folders; swimmers and contaminants are not flux
FLUX = ['aggregate', 'dense_detritus', 'large_loose_pellet', 'long_fecal_pellet', 'short_pellet', 'mini_pellet',
        'salp_pellet', 'rhizaria', 'phytoplankton']
SWIMMERS = ['copepod', 'amphipod', 'pteropod', 'zooplankton', 'zooplankton_part', 'foraminifera']
CONTAMINANTS = ['fiber', 'pollen']
ALL_FINE = FLUX + SWIMMERS + CONTAMINANTS + ['unidentifiable']

NOISE = 'false_detection'        # a detection that is no particle at all
NONFLUX = 'non_flux'             # swimmers and contaminants
NO_CARBON = {NOISE, NONFLUX, 'unidentifiable'}

# the classifier's 8 classes (amaral7u) and two other schemes
SCHEMES = {
    'amaral7u': dict(aggregate='aggregate', dense_detritus='aggregate', unidentifiable='aggregate',
                     large_loose_pellet='long_pellet', long_fecal_pellet='long_pellet', short_pellet='short_pellet',
                     mini_pellet='mini_pellet', salp_pellet='salp_pellet', rhizaria='rhizaria',
                     phytoplankton='phytoplankton'),
    # Song et al. (2025): small ellipsoids put in dense detritus at GelCam resolution
    'song6u': dict(aggregate='aggregate', dense_detritus='dense_detritus', short_pellet='dense_detritus',
                   mini_pellet='dense_detritus', unidentifiable='aggregate', large_loose_pellet='large_loose_pellet',
                   long_fecal_pellet='long_pellet', salp_pellet='salp_pellet', rhizaria='rhizaria',
                   phytoplankton='phytoplankton'),
    # Durkin et al. (2021): all nine flux classes
    'durkin9': {c: c for c in FLUX},
}
CLASSES_8 = ['aggregate', 'long_pellet', 'mini_pellet', 'non_flux', 'phytoplankton', 'rhizaria', 'salp_pellet',
             'short_pellet']

# merged 5-class view used for reporting
GELCAM5 = dict(aggregate='aggregate', short_pellet='aggregate', mini_pellet='aggregate', long_pellet='long_pellet',
               salp_pellet='salp_pellet', rhizaria='organisms', phytoplankton='organisms', non_flux='non_flux',
               false_detection='false_detection')


def scheme_label(source_class, scheme='amaral7u'):
    """Fine (source) class -> class of a scheme."""
    import numpy as np
    import pandas as pd
    m = SCHEMES[scheme]
    s = pd.Series(source_class)
    return np.where(s == NOISE, NOISE, s.map(m).fillna(NONFLUX)).astype(object)
