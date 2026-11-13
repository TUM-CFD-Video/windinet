# Video specification: original text

Original text by Harish, received 2026-11-04; docs/video_spec.md is the annotated version this repository implements.

# Standardized CFD Rollout Video Specification

**Purpose:** Produce videos that can be compared directly across independently trained surrogate models. Each collaborator receives the same dataset-level channel statistics and uses **identical visualization parameters**. No collaborator needs access to another model's predictions.

## 1. Required three-column layout

For every dataset, test trajectory, physical channel, and output time:

| Column 1 | Column 2 | Column 3 |
|---|---|---|
| Ground truth | Model prediction | Pointwise absolute error |

All three panels must display the **same physical time and spatial region**. Use the same panel dimensions, spatial orientation, aspect ratio, frame rate, and channel units across collaborators.

**Mandatory video canvas and frame dimensions:** Every video must be rendered from a Matplotlib figure that is **9.6 × 11.2 inches at 100 DPI**, yielding **960 × 1120 pixels (width × height)** for every output frame. The final encoded video must also be exactly **960 × 1120 px**. Do not resize, crop, add padding to, or otherwise change the frame dimensions during export. This applies to all collaborators, trajectories, and channels.

```python
FIGSIZE = (9.6, 11.2)  # inches (width, height)
DPI = 100
FRAME_SIZE = (960, 1120)  # pixels (width, height)

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
assert tuple(round(v * fig.dpi) for v in fig.get_size_inches()) == FRAME_SIZE
# Render/export at the same DPI; do NOT use bbox_inches='tight'.
# Video encoder must preserve 960 x 1120 px (no scaling/padding/cropping).
```


**Default:** use a fixed `asinh` (inverse hyperbolic sine) color normalization for all three panels. It preserves relatively fine contrast near a reference value and compresses large magnitudes. It is strictly increasing, so it does not discard extreme values *within the declared display bounds*.

## 2. Statistics to distribute (one set per dataset and channel)

Calculate these from **training-set ground truth in physical units**, pooling over training trajectories, spatial points, and timesteps:

- $\mu_c$: mean of channel $c$.
- $\sigma_c$: standard deviation of channel $c$.
- $a_c$: minimum of channel $c$ (**full training-data minimum; display lower bound**).
- $b_c$: maximum of channel $c$ (**full training-data maximum; display upper bound**).
- $L_c=Q_{0.005}(u_c^{\mathrm{train}})$: 0.5th percentile (**robust lower reference**).
- $U_c=Q_{0.995}(u_c^{\mathrm{train}})$: 99.5th percentile (**robust upper reference**).

Here $Q_p$ is the quantile at cumulative probability $p$ (so $Q_{0.005}$ is the 0.5th percentile, not the median).

**Distinction:** $a_c,b_c$ determine the actual normalization/display endpoints; $L_c,U_c$ are additional, informative percentiles that **must be displayed on every video frame** but are **not clipping limits**. Always use $a_c\le L_c\le U_c\le b_c$.

Require $\sigma_c > 0$ and $b_c > a_c$. Handle constant channels separately: display them at one constant color and flag them as unsuitable for relative-error normalization. Do not recompute these statistics from test trajectories or model predictions.

## 3. Columns 1 and 2: ground truth and prediction

Let $u_c(\mathbf{x},t)$ be ground truth and $\hat u_c(\mathbf{x},t)$ be the prediction for channel $c$.

**Center at the training mean** before applying `asinh`. This is useful for channels whose physical values have a large offset (e.g., pressure): it gives the approximately linear region of `asinh` to deviations of order $\sigma_c$ around the typical value.

Define the fixed channel-wise mapping

$$
\boxed{
\Phi_c(v)=
\frac{
\operatorname{asinh}\!\left(\dfrac{v-\mu_c}{\sigma_c}\right)
-
\operatorname{asinh}\!\left(\dfrac{a_c-\mu_c}{\sigma_c}\right)
}{
\operatorname{asinh}\!\left(\dfrac{b_c-\mu_c}{\sigma_c}\right)
-
\operatorname{asinh}\!\left(\dfrac{a_c-\mu_c}{\sigma_c}\right)
}
}
$$

Then apply **the same mapping** to both columns:

$$
\boxed{
I_{\mathrm{GT},c}(\mathbf{x},t)=\Phi_c\bigl(u_c(\mathbf{x},t)\bigr),
\qquad
I_{\mathrm{pred},c}(\mathbf{x},t)=\Phi_c\bigl(\hat u_c(\mathbf{x},t)\bigr).
}
$$

Properties:

- $\Phi_c(a_c)=0$ and $\Phi_c(b_c)=1$.
- Values between $a_c$ and $b_c$ are not percentile-clipped: they receive distinct, monotonically ordered colors.
- Identical physical values receive identical colors across **all baselines, trajectories, and frames**.
- Values *outside* $[a_c,b_c]$ are outside the display scale. They are **not deleted or clipped in the data**, but a finite colormap cannot show arbitrarily large values distinctly. Mark these as out-of-range (see Section 6).

**Default physical-field colormap:** `viridis`. If a signed field needs a diverging palette centered on physical zero, that should be a *separately specified, fixed channel policy* with symmetric physical bounds—not a per-model choice.

## 4. Column 3: normalized absolute pointwise error

First compute the **unmodified physical error**:

$$
\boxed{\Delta_c(\mathbf{x},t)=\left|\hat u_c(\mathbf{x},t)-u_c(\mathbf{x},t)\right|.}
$$

Normalize by the **training ground-truth standard deviation**:

$$
\boxed{e_c(\mathbf{x},t)=\frac{\Delta_c(\mathbf{x},t)}{\sigma_c}.}
$$

The error $e_c=1$ means an absolute error of one training standard deviation. The error is never normalized using a model's own predicted errors.

To avoid an arbitrary cutoff at $e_c=1$, set the error display upper bound from the **full training ground-truth range**:

$$
\boxed{E_{\max,c}=\frac{b_c-a_c}{\sigma_c}.}
$$

Use a fixed `asinh` mapping for the error panel:

$$
\boxed{
I_{\mathrm{err},c}(\mathbf{x},t)
=
\Psi_c\bigl(e_c(\mathbf{x},t)\bigr),
\qquad
\Psi_c(e)=\frac{\operatorname{asinh}(e)}{\operatorname{asinh}(E_{\max,c})}.
}
$$

**What is actually plotted:** $I_{\mathrm{err},c}$ is the **color-normalized error intensity plotted in the third panel**, not the raw error $\Delta_c$ and not the intermediate normalized error $e_c$. At each pixel, the `inferno` colormap is evaluated at $I_{\mathrm{err},c}=\Psi_c(e_c)$. In Matplotlib, calling `imshow(e_c, norm=error_norm)` performs the $\Psi_c$ transformation automatically; you must **not apply $\Psi_c$ a second time**.

This mapping is equivalent to Matplotlib `AsinhNorm(linear_width=1, vmin=0, vmax=E_max,c)` applied to $e_c$. It is linear-like for small normalized errors and compresses large errors. All collaborators use the **same** $E_{\max,c}$, not a limit estimated from their own results.

**Default error colormap:** `inferno`; minimum is always zero. Errors larger than $E_{\max,c}$ must be explicitly marked as exceeding the display scale and included in the numerical statistics; they must **not** be silently removed.

> **Interpretation:** Colorbar tick labels should show actual physical values for columns 1–2 and either $e_c$ (multiples of $\sigma_c$) or $\Delta_c$ in physical units for column 3. Do not label nonlinear colorbars as if the colors were linearly spaced in physical values.

## 5. Frame annotations and implementation

### 5.1 Mandatory **complete transformations** and statistics on every video frame

**Every exported video frame must visibly contain the full set of transformations below** in a fixed, legible annotation panel. A note in the README, a tooltip, or formula names alone **does not satisfy this requirement**. Do not show only $I_{\mathrm{GT}}=\Phi(u)$, $I_{\mathrm{pred}}=\Phi(\hat u)$ and $I_{\mathrm{err}}=\Psi(e)$ without also displaying the definitions of $\Phi$, $\Psi$, the normalizers, and their input quantities **in that same frame**.

**Ground-truth and prediction transformations:** introduce the denominator, and explicitly show how each column is normalized:

$$
D_c=\operatorname{asinh}\!\left(\frac{b_c-\mu_c}{\sigma_c}\right)
-\operatorname{asinh}\!\left(\frac{a_c-\mu_c}{\sigma_c}\right),
$$

$$
\Phi_c(v)=\frac{\operatorname{asinh}\!\left(\frac{v-\mu_c}{\sigma_c}\right)
-\operatorname{asinh}\!\left(\frac{a_c-\mu_c}{\sigma_c}\right)}{D_c},
$$

$$
\boxed{
I_{\mathrm{GT},c}=\Phi_c(u_c)=
\frac{\operatorname{asinh}((u_c-\mu_c)/\sigma_c)
-\operatorname{asinh}((a_c-\mu_c)/\sigma_c)}{D_c}}
$$

$$
\boxed{
I_{\mathrm{pred},c}=\Phi_c(\hat u_c)=
\frac{\operatorname{asinh}((\hat u_c-\mu_c)/\sigma_c)
-\operatorname{asinh}((a_c-\mu_c)/\sigma_c)}{D_c}}
$$

**Absolute-error transformation:** show the physical error, standardized error, shared error limit, nonlinear transform, and the actual displayed intensity:

$$
\Delta_c=|\hat u_c-u_c|,\qquad
 e_c=\frac{\Delta_c}{\sigma_c}=\frac{|\hat u_c-u_c|}{\sigma_c},\qquad
E_{\max,c}=\frac{b_c-a_c}{\sigma_c},
$$

$$
\Psi_c(e)=\frac{\operatorname{asinh}(e)}{\operatorname{asinh}(E_{\max,c})},
$$

$$
\boxed{
I_{\mathrm{err},c}=\Psi_c(e_c)=
\frac{\operatorname{asinh}(|\hat u_c-u_c|/\sigma_c)}
{\operatorname{asinh}((b_c-a_c)/\sigma_c)}
\quad\text{(PLOTTED ERROR INTENSITY)}
}
$$

**Numeric channel statistics (visible on *every* frame, with units):**

$$
L_c=Q_{0.005}(u_c^{\mathrm{train}}),\quad
U_c=Q_{0.995}(u_c^{\mathrm{train}}),\quad
a_c=\min u_c^{\mathrm{train}},\quad
b_c=\max u_c^{\mathrm{train}},\quad
\mu_c,\quad\sigma_c.
$$

Show the **actual numeric values** of $L_c,U_c,a_c,b_c,\mu_c,\sigma_c$, not just these symbols, as well as the channel/unit and current physical time. The first four values have physical units; $\mu_c$ and $\sigma_c$ do too. All six remain fixed for the same dataset and channel across times and models.

**Layout constraint:** reserve a dedicated formula/metadata panel within the **unchanged 960 × 1120 px canvas**, not on top of the flow fields. If the formulas do not fit legibly, reduce margins or reposition panels; **never omit formulas, shrink the video, or dynamically vary the annotation layout**. On-screen fraction and RMSE statistics may appear in that panel as well.

**What is plotted:** the displayed colors correspond to $I_{\mathrm{GT},c}$, $I_{\mathrm{pred},c}$, and **$I_{\mathrm{err},c}$**, respectively. With Matplotlib, `imshow(gt - mu, norm=field_norm)`, `imshow(pred - mu, norm=field_norm)`, and `imshow(error, norm=error_norm)` perform the corresponding intensity mappings internally; **do not transform the data twice**. Calculate RMSE/MAE from original physical fields, never plotted intensities.

### 5.2 Minimal Matplotlib implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm


def render_three_columns(ax_gt, ax_pred, ax_err, gt, pred, stats, time_label):
    """gt and pred: 2D arrays in ORIGINAL PHYSICAL UNITS.

    stats: fixed training-ground-truth statistics for this dataset/channel:
           {'mean': ..., 'std': ..., 'min': ..., 'max': ...,
            'p0_5': ..., 'p99_5': ..., 'unit': '...'}
    """
    mu = float(stats['mean'])
    sigma = float(stats['std'])
    a = float(stats['min'])
    b = float(stats['max'])
    L = float(stats['p0_5'])
    U = float(stats['p99_5'])
    unit = str(stats['unit'])
    if not (a <= L <= U <= b):
        raise ValueError('Percentiles must lie inside training extrema.')
    if not (sigma > 0 and b > a):
        raise ValueError('Constant/invalid channel: define an explicit special case.')

    # Center at training mean; norm and bounds remain fixed across models and time.
    field_norm = AsinhNorm(
        linear_width=sigma,
        vmin=a - mu,
        vmax=b - mu,
        clip=False,
    )

    # Normalize absolute error by training GT std, with a GT-derived upper limit.
    error = np.abs(pred - gt) / sigma
    error_limit = (b - a) / sigma
    error_norm = AsinhNorm(
        linear_width=1.0,
        vmin=0.0,
        vmax=error_limit,
        clip=False,
    )

    # Dedicated colors for out-of-range and invalid pixels.
    field_cmap = plt.get_cmap('viridis').with_extremes(
        under='cyan', over='magenta', bad='gray'
    )
    error_cmap = plt.get_cmap('inferno').with_extremes(
        over='magenta', bad='gray'
    )

    im_gt = ax_gt.imshow(gt - mu, norm=field_norm, cmap=field_cmap,
                         interpolation='nearest')
    im_pred = ax_pred.imshow(pred - mu, norm=field_norm, cmap=field_cmap,
                             interpolation='nearest')
    # Matplotlib applies Psi_c to `error` internally: the plotted intensity
    # in the third panel is I_err = Psi_c(error), not `error` itself.
    im_err = ax_err.imshow(error, norm=error_norm, cmap=error_cmap,
                           interpolation='nearest')

    # Burn FULL formulas (including definitions AND expanded panel mappings)
    # into EVERY frame, in the same reserved area.
    fig = ax_gt.figure
    fig.subplots_adjust(bottom=0.38, top=0.94, left=0.04, right=0.98,
                        wspace=0.14)
    annotation_lines = [
        r"$D_c=\mathrm{asinh}((b_c-\mu_c)/\sigma_c)"
        r"-\mathrm{asinh}((a_c-\mu_c)/\sigma_c)$",
        r"$\Phi_c(v)=[\mathrm{asinh}((v-\mu_c)/\sigma_c)"
        r"-\mathrm{asinh}((a_c-\mu_c)/\sigma_c)]/D_c$",
        r"$I_{\mathrm{GT},c}=\Phi_c(u_c)="
        r"[\mathrm{asinh}((u_c-\mu_c)/\sigma_c)"
        r"-\mathrm{asinh}((a_c-\mu_c)/\sigma_c)]/D_c$",
        r"$I_{\mathrm{pred},c}=\Phi_c(\hat u_c)="
        r"[\mathrm{asinh}((\hat u_c-\mu_c)/\sigma_c)"
        r"-\mathrm{asinh}((a_c-\mu_c)/\sigma_c)]/D_c$",
        r"$\Delta_c=|\hat u_c-u_c|,\quad "
        r"e_c=\Delta_c/\sigma_c,\quad E_{\max,c}=(b_c-a_c)/\sigma_c$",
        r"$\Psi_c(e)=\mathrm{asinh}(e)/\mathrm{asinh}(E_{\max,c})$",
        r"$I_{\mathrm{err},c}=\Psi_c(e_c)="
        r"\mathrm{asinh}(|\hat u_c-u_c|/\sigma_c)"
        r"/\mathrm{asinh}((b_c-a_c)/\sigma_c)$  (PLOTTED)",
        r"$L_c$=" + f"{L:.6g} {unit}   " + r"$U_c$=" + f"{U:.6g} {unit}   "
        + r"$a_c$=" + f"{a:.6g} {unit}   " + r"$b_c$=" + f"{b:.6g} {unit}",
        r"$\mu_c$=" + f"{mu:.6g} {unit}   " + r"$\sigma_c$="
        + f"{sigma:.6g} {unit}   t={time_label}",
    ]
    annotation = "\n".join(annotation_lines)
    if not hasattr(fig, '_cfd_formula_artist'):
        fig._cfd_formula_artist = fig.text(
            0.04, 0.36, annotation,
            va='top', ha='left', fontsize=10.5, linespacing=1.45,
        )
    else:
        fig._cfd_formula_artist.set_text(annotation)

    # These indicators are independent of color saturation.
    finite = np.isfinite(gt) & np.isfinite(pred)
    fraction_nonfinite = 1.0 - np.mean(finite)
    if finite.any():
        fraction_error_over = np.mean(error[finite] > error_limit)
        fraction_pred_outside = np.mean(
            (pred[finite] < a) | (pred[finite] > b)
        )
    else:
        fraction_error_over = np.nan
        fraction_pred_outside = np.nan

    # A non-finite frame is not assigned a misleading finite RMSE.
    rmse = (np.sqrt(np.mean((pred - gt) ** 2))
            if finite.all() else np.nan)

    return {
        'images': (im_gt, im_pred, im_err),
        'rmse_physical_units': rmse,
        'fraction_error_over_limit': fraction_error_over,
        'fraction_prediction_outside_field_range': fraction_pred_outside,
        'fraction_nonfinite': fraction_nonfinite,
        # Remember: field colorbar ticks must add mu to show physical units.
    }
```

For physical-field colorbars, an `imshow` colorbar tick at centered value $z$ corresponds to physical value $z+\mu_c$. For the error colorbar, values are measured in units of $\sigma_c$. Use `extend='both'` for the field colorbars and `extend='max'` for the error colorbar to signal possible out-of-range values.

**Important:** `clip=False` does not create infinitely many colors beyond `vmin`/`vmax`; the colormap still has endpoint/over-range behavior. Its purpose is to let you identify values outside the configured range rather than silently forcing their normalized values to 0 or 1.

## 6. What every collaborator must deliver

1. **One three-column video per agreed trajectory and physical channel**, using the shared statistics and formulas above.
2. **Identical physical timestamps, frame count, time sampling, crop, aspect ratio, and spatial orientation** for all baselines. Do not dynamically zoom or rescale.
3. **Fixed colorbars** showing channel units and appropriate nonlinear tick labels. **Every rendered video frame must visibly include the complete expanded formulas for $D_c$, $\Phi_c$, $I_{\mathrm{GT}}$, $I_{\mathrm{pred}}$, $\Delta_c$, $e_c$, $E_{\max,c}$, $\Psi_c$, and $I_{\mathrm{err}}$; explicitly label $I_{\mathrm{err}}$ as the plotted error intensity.** The caption must also display numeric $L_c$, $U_c$, $a_c$, $b_c$, $\mu_c$, and $\sigma_c$, with units, plus the current timestamp. See Section 5.1.
4. **Per-frame RMSE in original physical units**, computed on the full, un-clipped fields; optionally MAE. If any pixel is non-finite, flag the frame and report RMSE as undefined rather than silently scoring the finite subset.
5. **Per-frame fractions of (i) predicted values outside $[a_c,b_c]$, (ii) errors above $E_{\max,c}$, and (iii) non-finite predicted values**. Non-finite pixels must have an explicit visual marker. If a prediction contains non-finite values, flag the frame; never silently omit them from benchmark metrics.
6. A metadata/configuration file recording dataset, test trajectory ID, model ID, channel name/units, the shared channel statistics (including $L_c,U_c$), plot limits, colormaps, spatial domain, frame timestamps, and frame rate.
7. **Resolution check:** Verify both the rendered frames and the encoded video are exactly **960 × 1120 pixels**, created with **100 DPI and a 9.6 × 11.2 inch figure**; do not permit automatic tight cropping or encoder resizing.

**Do not:** dynamically compute min/max per frame; compute statistics from predictions; use model-specific error limits; clip the source data before calculating errors; or apply normalization to already rendered RGB images.

## 7. Acceptance checks for the agent

- A prediction identical to the ground truth yields a zero error field.
- The same ground-truth field renders identically for every model, and a particular physical value always has the same color across frames.
- $\Phi_c(a_c)=0$, $\Phi_c(b_c)=1$, $\Psi_c(0)=0$, and $\Psi_c(E_{\max,c})=1$ (up to floating-point precision).
- A rare large value **within** training extrema is compressed by `asinh`, not replaced by the 99.5th-percentile color.
- A value beyond a display bound or a NaN is visibly flagged and counted.
- All reported RMSE/MAE calculations use the original physical fields, not the transformed visualization arrays.
- Every frame has legible, correctly rendered **complete expanded formulas** for all three panel transforms, including their denominators and all intermediate normalizers, and explicitly identifies $I_{\mathrm{err}}$ as the **plotted** quantity, and displays the shared fixed numeric values $L_c$, $U_c$, $a_c$, $b_c$, $\mu_c$, $\sigma_c$ with units.
- Figure size is **9.6 × 11.2 inches**, DPI is **100**, and every exported frame and encoded video is **960 × 1120 px** with no crop, padding, or resize.

**Bottom line:** One set of ground-truth training statistics per channel; identical mean-centered `asinh` for ground truth and prediction; normalized absolute error with its own fixed ground-truth-derived `asinh` scale; never adjust any visualization limits for an individual collaborator or frame.
