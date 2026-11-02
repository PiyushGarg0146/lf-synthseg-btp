"""Random tumor shapes: unions of randomly rotated and warped ellipsoids (paper III-A1).

One tumor = three nested-ish shapes on a GRID^3 voxel grid (1 mm voxels):
  m1 (NCR, smallest), m2 (ET, middle), m3 (ED, largest), each moved by a random displacement Z(.).
Our choices where the paper is silent (documented in README "Synthesizer design choices"):
  - whole-tumor diameter ~ U(35, 85) mm  (BraTS whole tumors are about 20k-400k voxels)
  - m2 volume = U(0.35, 0.7) x m3, m1 volume = U(0.08, 0.3) x m3
  - 2-4 ellipsoids per set, boundary roughened by smooth noise
"""
import numpy as np
from scipy import ndimage

GRID = 128
D_RANGE = (35.0, 85.0)
V2_FRAC = (0.35, 0.7)
V1_FRAC = (0.08, 0.3)


def _rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _axes(grid):
    c = (grid - 1) / 2
    a = np.arange(grid, dtype=np.float32) - c
    return a[:, None, None], a[None, :, None], a[None, None, :]


def _params(rng, n, radius):
    """n random ellipsoids around the origin: (center, rotation, radii) for a tumor of the given base radius."""
    out = []
    for _ in range(n):
        out.append((rng.normal(scale=0.35 * radius, size=3), _rotation(rng), radius * rng.uniform(0.45, 1.0, size=3)))
    return out


def _field(params, noise, scale, grid):
    x = _axes(grid)
    f = None
    for c, R, r in params:
        c, r = c * scale, r * scale
        d = [x[0] - c[0], x[1] - c[1], x[2] - c[2]]
        q2 = 0
        for j in range(3):
            q = R[0, j] * d[0] + R[1, j] * d[1] + R[2, j] * d[2]
            q2 = q2 + (q / r[j]) ** 2
        f = 1 - q2 if f is None else np.maximum(f, 1 - q2)
    return f + noise


def _largest(m):
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    return lab == 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))


def random_shape(rng, volume, grid=GRID):
    """Union of 2-4 random ellipsoids with a rough boundary and (about) the requested volume in voxels."""
    radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
    params = _params(rng, int(rng.integers(2, 5)), radius)
    noise = ndimage.gaussian_filter(rng.normal(size=(grid,) * 3).astype(np.float32), 10)
    noise *= rng.uniform(0.1, 0.3) / noise.std()
    scale = 1.0
    for _ in range(3):                      # rescale so the volume matches the request
        m = _field(params, noise, scale, grid) > 0
        v = int(m.sum())
        if v == 0:
            scale *= 1.3
            continue
        scale *= (volume / v) ** (1 / 3)
    m = _largest(_field(params, noise, scale, grid) > 0)
    return m


def random_tumor_shapes(rng, diameter=None, grid=GRID, max_tries=20):
    """Return (m1, m2, m3) boolean masks (NCR, ET, ED shapes). Shapes that touch the grid edge are redrawn."""
    for _ in range(max_tries):
        d = diameter or rng.uniform(*D_RANGE)
        v3 = np.pi / 6 * d ** 3
        m3 = random_shape(rng, v3, grid)
        m2 = random_shape(rng, v3 * rng.uniform(*V2_FRAC), grid)
        m1 = random_shape(rng, v3 * rng.uniform(*V1_FRAC), grid)
        reach = 0.2 * (3 * v3 / (4 * np.pi)) ** (1 / 3)         # displacement Z(.) of the inner shapes
        out = []
        for m in (m1, m2):
            shift = rng.uniform(-reach, reach, size=3)
            out.append(ndimage.shift(m.astype(np.float32), shift, order=1) > 0.5)
        m1, m2 = out
        edge = lambda m: m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() or m[:, :, 0].any() or m[:, :, -1].any()
        if not any(edge(m) for m in (m1, m2, m3)) and m1.any() and m2.any():
            return m1, m2, m3
    raise RuntimeError("could not draw tumor shapes that fit the grid")
