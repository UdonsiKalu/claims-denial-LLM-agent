# fft_diag.py
import numpy as np

def _hann(n):  # simple window to reduce leakage
    return 0.5 - 0.5 * np.cos(2*np.pi*np.arange(n)/max(n-1, 1))

def power_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    x: 1D array. Returns (freqs, power). DC (index 0) included; caller can skip it.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    x = x * _hann(len(x))
    S = np.fft.rfft(x)
    P = (S.real**2 + S.imag**2)
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    return freqs, P

def rowwise_power(series, axis=0):
    """
    Compute row-wise FFT power spectra.
    series: np.ndarray (2D) with rows or columns as series
    axis: 0 = chunks, 1 = queries
    """
    if series is None or len(series) == 0:
        return np.array([]), np.array([[]])  # safe empty return

    # Ensure series is at least 2D
    series = np.atleast_2d(series)

    try:
        freqs, _ = power_spectrum(series[0])
    except Exception:
        # fallback in case series[0] is invalid
        return np.array([]), np.array([[]])

    P = np.zeros((len(series), len(freqs)))
    for i, s in enumerate(series):
        if s is None or len(s) == 0:
            continue  # skip empty rows
        _, pi = power_spectrum(s)
        P[i, :] = pi

    return freqs, P

def spectrum_features(freqs: np.ndarray, power_row: np.ndarray) -> dict:
    """
    Compute peak frequency (ignore DC), estimated period in 'steps',
    PNR in dB (peak power vs median floor), and spectral flatness.
    Safe against empty inputs.
    """
    p = np.asarray(power_row, dtype=float).copy()

    # --- Guard: empty spectrum ---
    if p.size == 0 or freqs is None or len(freqs) == 0:
        return {
            "peak_idx": None,
            "peak_freq": None,
            "period": None,
            "pnr_db": None,
            "flatness": None,
        }

    # --- IGNORE DC component safely ---
    if p.size > 0:
        p[0] = 0.0

    i_peak = int(p.argmax())
    f0 = float(freqs[i_peak]) if i_peak < len(freqs) else 0.0

    # safe period calculation
    period = (1.0 / f0) if f0 > 1e-12 else np.inf

    peak_power = float(p[i_peak]) + 1e-18
    noise_floor = float(np.median(p[1:])) + 1e-18
    pnr_db = 10.0 * np.log10(peak_power / noise_floor)

    # spectral flatness: geom_mean / arith_mean
    eps = 1e-18
    if p.size > 1:
        geom = np.exp(np.mean(np.log(p[1:] + eps)))
        arith = np.mean(p[1:] + eps)
        flatness = float(geom / arith)
    else:
        flatness = None

    return {
        "peak_idx": i_peak,
        "peak_freq": f0,
        "period": period,
        "pnr_db": pnr_db,
        "flatness": flatness,
    }


    # ignore DC
    p[0] = 0.0

    i_peak = int(p.argmax())
    f0 = float(freqs[i_peak]) if i_peak < len(freqs) else 0.0

    # safe period calculation
    period = (1.0 / f0) if f0 > 1e-12 else np.inf

    peak_power = float(p[i_peak]) + 1e-18
    noise_floor = float(np.median(p[1:])) + 1e-18
    pnr_db = 10.0 * np.log10(peak_power / noise_floor)

    # spectral flatness: geom_mean / arith_mean
    eps = 1e-18
    if p.size > 1:
        geom = np.exp(np.mean(np.log(p[1:] + eps)))
        arith = np.mean(p[1:] + eps)
        flatness = float(geom / arith)
    else:
        flatness = None

    return {
        "peak_idx": i_peak,
        "peak_freq": f0,
        "period": period,
        "pnr_db": pnr_db,
        "flatness": flatness,
    }



def detect_rois(power_matrix: np.ndarray,
                peak_idxs: np.ndarray,
                target_idx: int,
                pnr_db: np.ndarray,
                min_pnr_db: float = 6.0,
                min_run: int = 5) -> list[tuple[int, int]]:
    """
    Given rowwise power_matrix (rows = chunks or queries) and per-row peak index & PNR,
    find contiguous row ranges whose peak aligns with target_idx and exceeds threshold.
    Returns list of (start, end_inclusive) ranges = ROIs.
    """
    rows = power_matrix.shape[0]
    mask = (peak_idxs == target_idx) & (pnr_db >= min_pnr_db)
    rois = []
    i = 0
    while i < rows:
        if mask[i]:
            j = i
            while j + 1 < rows and mask[j + 1]:
                j += 1
            if (j - i + 1) >= min_run:
                rois.append((i, j))
            i = j + 1
        else:
            i += 1
    return rois
