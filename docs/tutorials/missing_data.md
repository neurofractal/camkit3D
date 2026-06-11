# Missing data

In motion capture, missing data is common. It arises from reprojection errors, occluded markers, poor video quality etc.

In CamKit3D each coordinate (x, y, z) is treated as an independent signal, and gaps (NaN segments) are filled along the time axis. Method choice depends on gap length, the noise level of the surrounding data, and the relative priority of smoothness versus fidelity to the observed samples.

## Strategies

### Linear

Each gap is filled by connecting the last valid sample before it to the first valid sample after it with a straight line. The result is the simplest possible estimate, assuming constant velocity across the gap with no curvature. It is exact at the endpoints but carries no information about the underlying trajectory shape.

**Pros**
- Fast, simple, and numerically stable.

**Cons**
- Introduces velocity discontinuities at gap boundaries.
- Does not account for local motion trends, so curved trajectories are flattened across the gap.
- Suitable only for short gaps.

### Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)

PCHIP fits a separate cubic polynomial to each inter-sample interval with derivatives constrained to preserve local monotonicity. This yields a trajectory that passes through all observed points and remains smooth in both position and velocity, avoiding the kinks that would otherwise introduce spurious spikes in the derived kinematics. Unlike cubic spline interpolation, the approach also suppresses overshoot at rapid transitions, preventing excursions beyond the measured data and preserving the underlying movement profile.

**Pros**
- Preserves local motion trends.
- Avoids ringing.

**Cons**
- Interpolates only within observed data; does not extrapolate beyond gap edges.
- Accuracy degrades over long gaps, where the interpolated curve diverges increasingly from true motion.
- Marginally higher computational cost than linear interpolation.

### Savitzky-Golay

Gaps are first filled by linear interpolation, then the entire signal is passed through a Savitzky-Golay filter, which fits a low-order polynomial to a sliding window of samples by least squares and replaces each point with the fitted value. This smooths high-frequency noise while preserving the width and height of genuine features such as peaks and turning points, which a simple moving average would attenuate. The approach targets noise reduction across the whole signal rather than reconstruction of the gap alone.

**Pros**
- Reduces measurement noise while preserving peak shape better than a moving average.
- Blends the filled segment into surrounding data when those data are noisy.

**Cons**
- Smoothing modifies all frames within the window, including observed samples, not only the gap.
- Sensitive to window length and polynomial order; suboptimal settings over-smooth or under-smooth.
- Can attenuate genuinely fast movements.
- Output quality is bounded by the underlying linear interpolation used to fill the gap.


### Options you can tweak in `interpolate_nans(...)`

- **`method`**  
  Interpolation strategy:
    - `"pchip"` – Smooth, shape-preserving, no overshoot. Best default for mocap.
    - `"linear"` – Straight-line interpolation. Fast and safe, but creates visible
    kinks at gap boundaries.
    - `"savgol"` – Linear interpolation followed by Savitzky–Golay smoothing.
    Useful when surrounding data are noisy.

- **`max_gap_seconds`**  
  Maximum duration (in seconds) of a bounded NaN gap to fill.  
  Longer gaps are left as NaN.

- **`fps`**  
  Frame rate of the data. Used to convert `max_gap_seconds` into a frame count.

- **`savgol_window`** *(only for `method="savgol"`)*  
  Window length of the Savitzky–Golay filter (must be an odd integer).  
  Larger values give smoother results but may blur fast motion.

- **`savgol_polyorder`** *(only for `method="savgol"`)*  
  Polynomial order for the Savitzky–Golay filter.  
  Higher orders can follow curvature better but may fit noise.

- **`verbose`**  
  If `True`, prints a summary report of how many gaps and values were filled or
  skipped.


