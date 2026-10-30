# Step 10: carbon per particle, C = A * V**B (Durkin et al. 2021, Table 2).
# Used by: render.py (true carbon), classify.py, hitl.py (predicted carbon).
import math
import numpy as np

TABLE2 = {  # class: (shape, A [mg um^-3], B)
    'aggregate': ('sphere', 0.1e-9, 0.80),
    'dense_detritus': ('sphere', 0.1e-9, 0.83),
    'large_loose_pellet': ('cylinder', 0.1e-9, 0.83),
    'long_fecal_pellet': ('cylinder', 0.1e-9, 1.0),
    'short_pellet': ('ellipsoid', 0.1e-9, 1.0),
    'mini_pellet': ('sphere', 0.1e-9, 1.0),
    'salp_pellet': ('cuboid', 0.04e-9, 1.0),
    'rhizaria': ('sphere', 0.004e-9, 0.939),
    'phytoplankton': ('sphere', 0.288e-9, 0.811),
}
HYPERBOLIC = {'long_fecal_pellet': (187.0, 424.0), 'large_loose_pellet': (553.0, 996.0)}
LINEAR_W = {'short_pellet': 0.54, 'salp_pellet': 0.63}

# a merged (classifier) class uses the carbon model of its main member
CARBON_OF = {'long_pellet': 'long_fecal_pellet'}


def volume_um3(cls, esd_um):
    """Modelled particle volume (um^3) from class and ESD (Table 2 shapes)."""
    if cls not in TABLE2 or not esd_um or esd_um <= 0 or esd_um != esd_um:
        return float('nan')
    shape = TABLE2[cls][0]
    a = math.pi / 4.0 * esd_um ** 2
    if shape == 'sphere':
        return math.pi / 6.0 * esd_um ** 3
    if cls in HYPERBOLIC:
        X, Y = HYPERBOLIC[cls]
        w = X * esd_um / (Y + esd_um)
    else:
        w = LINEAR_W[cls] * esd_um
    if shape == 'cylinder':
        return math.pi / 4.0 * w * w * max(a / w, w)
    if shape == 'ellipsoid':
        return math.pi / 6.0 * w * w * max(4.0 * a / (math.pi * w), w)
    if shape == 'cuboid':
        return w * max(a / w, w) * (w / 4.0)
    return float('nan')


def carbon_mg(cls, esd_um):
    """Carbon of one particle (mg C)."""
    cls = CARBON_OF.get(cls, cls)
    if cls not in TABLE2:
        return float('nan')
    _, A, B = TABLE2[cls]
    v = volume_um3(cls, esd_um)
    return A * v ** B if v == v else float('nan')


def carbon_array(classes, esd_um, scales=None):
    """Carbon (mg) of many particles."""
    scales = scales or {}
    out = np.zeros(len(esd_um))
    for i, (c, e) in enumerate(zip(classes, esd_um)):
        v = carbon_mg(c, float(e))
        out[i] = 0.0 if v != v else v * scales.get(c, 1.0)
    return out
