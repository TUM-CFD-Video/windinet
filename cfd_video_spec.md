# Standardized CFD Rollout Video Specification

★ = changed against the first version. Reference implementation: `cfd_video.py` (its ★ comments mark the same changes).

**★ Changes and why**

- ★ **Display range: training mean ± 2σ instead of training min..max.** The extremes come from a few shocks (density
  up to 30, momentum ±30, while typical values are 0.2–2 and ±1), so the whole flow used a thin slice of the colormap
  and looked flat.
- ★ **asinh linear width 0.1σ for the fields (was σ).** Fronts and vortices differ by 0.1–0.5σ; the narrower width
  gives them contrast.
- ★ **Signed error (prediction − truth)/σ on ±2σ, asinh width 0.05σ, diverging colormap (was |error| up to
  (max − min)/σ, width 1, inferno).** Typical errors are 0.01–0.5σ: on the old scale they sat in the bottom few percent
  of the colormap and the panel was black, so models looked better than they are. The sign shows whether a prediction
  is too high or too low, and whether a front is ahead or behind.
- ★ **Signed channels (momentum) diverging around physical 0** with symmetric bounds, as a fixed channel policy (the
  spec already allowed it).
- ★ **One row per channel in one video** (all fields side by side, e.g. density and pressure fronts together);
  one video per channel stays an option. The frame is 960 px wide and its height follows the rows (4 channels:
  960 × 1150 px) instead of a fixed 960 × 1120 px with empty space.
- ★ **Frame text: title and time on top, one centred scale line under the panels.** The colorbars show physical
  values; the formulas and the numeric statistics go into the per-video metadata file instead of on every frame.

Everything else is unchanged: shared training statistics, the same fixed scales for every model and frame, physical
units, out-of-range values flagged in colour and counted, 100 DPI and 960 px width.

## 1. Required three-column layout

For every dataset, test trajectory and output time, ★ one row per physical channel:

| Column 1 | Column 2 | Column 3 |
|---|---|---|
| Ground truth | Model prediction | ★ Signed pointwise error |

All three panels must display the **same physical time and spatial region**. Use the same panel dimensions, spatial
orientation, aspect ratio, frame rate, and channel units across collaborators.

**Video canvas:** ★ a Matplotlib figure 9.6 inches wide at 100 DPI (960 px), 2.5 inches per channel row plus title
and text (4 channels: 960 × 1150 px); the same for every collaborator with the same channels. Do not resize, crop or
pad during export.

**Default:** a fixed `asinh` color normalization for all three panels. It is linear near a reference value and
compresses large magnitudes; it is strictly increasing.

## 2. Statistics to distribute (one set per dataset and channel)

From **training-set ground truth in physical units**, pooled over training trajectories, spatial points, and timesteps:

- $\mu_c$: mean, $\sigma_c$: standard deviation.
- $a_c$, $b_c$: minimum and maximum. ★ No longer the display bounds; the lower display bound is kept $\ge a_c$.
- $L_c=Q_{0.005}$, $U_c=Q_{0.995}$: 0.5th and 99.5th percentiles, ★ recorded in the metadata file.

Require $\sigma_c > 0$ and $b_c > a_c$; constant channels need an explicit special case. Never recompute these
statistics from test trajectories or model predictions.

## 3. Columns 1 and 2: ground truth and prediction

★ Display range and mapping, the same for ground truth and prediction:

$$
\mathrm{lo}_c=\max(\mu_c-2\sigma_c,\ a_c),\qquad \mathrm{hi}_c=\mu_c+2\sigma_c,
$$

$$
\Phi_c(v)=\frac{\operatorname{asinh}\!\left(\frac{v-\mu_c}{0.1\,\sigma_c}\right)
-\operatorname{asinh}\!\left(\frac{\mathrm{lo}_c-\mu_c}{0.1\,\sigma_c}\right)}
{\operatorname{asinh}\!\left(\frac{\mathrm{hi}_c-\mu_c}{0.1\,\sigma_c}\right)
-\operatorname{asinh}\!\left(\frac{\mathrm{lo}_c-\mu_c}{0.1\,\sigma_c}\right)}
$$

so $\Phi_c(\mathrm{lo}_c)=0$ and $\Phi_c(\mathrm{hi}_c)=1$. In Matplotlib:
`AsinhNorm(linear_width=0.1*sigma, vmin=lo-mu, vmax=hi-mu)` on `value - mu`. Values outside the range are **not
clipped in the data**; they are drawn cyan (below) or magenta (above) and counted.

Colormap `viridis`. ★ Signed channels (momentum): `RdBu_r` centred at physical 0, bounds
$\pm(|\mu_c|+2\sigma_c)$.

## 4. Column 3: signed pointwise error

★ $$e_c=\frac{\hat u_c-u_c}{\sigma_c}$$ (was $|\hat u_c-u_c|/\sigma_c$), shown on $[-2, 2]$ with
`AsinhNorm(linear_width=0.05, vmin=-2, vmax=2)` and `RdBu_r` (0 = white). Errors beyond ±2σ are drawn cyan/magenta and
counted, never removed. The error is always normalized by the training $\sigma_c$, never by a model's own errors.

## 5. Frame annotations

★ On every frame: model, trajectory and physical time on top; one centred line under the panels with the display
ranges and asinh widths. The numeric $\mu_c, \sigma_c, a_c, b_c, L_c, U_c$ are in the metadata file. Colorbars show physical values
for columns 1–2 and multiples of $\sigma_c$ for column 3. The complete mapping is written to the metadata file of every
video. RMSE and fractions are computed on the physical fields, never on plotted intensities.

## 6. What every collaborator must deliver

1. ★ One video per agreed trajectory with a row per channel (`rollout.mp4`; `--per-channel` gives `<channel>.mp4`).
2. Identical physical timestamps, frame count, crop, aspect ratio and orientation for all baselines; no dynamic zoom
   or rescaling.
3. Fixed colorbars with physical tick labels.
4. Per-frame RMSE in physical units; undefined (null) if any pixel is non-finite.
5. Per-frame fractions of (i) predictions outside the display range, ★ (ii) errors beyond ±2σ, (iii) non-finite
   values.
6. A metadata file per video (`.json`): title, statistics, display ranges, mapping, frame size, fps and the
   per-frame numbers per channel.
7. ★ Every frame and the encoded video at 100 DPI, 960 px wide, height from the number of rows, no resizing.

**Do not:** compute ranges per frame, from predictions or per model; clip the data before computing errors; normalize
already rendered RGB images.

## 7. Acceptance checks

- A prediction equal to the ground truth gives a white (zero) error panel.
- A physical value has the same color for every model, trajectory and frame.
- $\Phi_c(\mathrm{lo}_c)=0$, $\Phi_c(\mathrm{hi}_c)=1$; the error maps $-2, 0, 2$ to $0, 0.5, 1$.
- Values beyond a display bound and NaNs are visibly flagged and counted.
- Frames and video have the same size for every collaborator with the same channels.

## Usage

```bash
python cfd_video.py rollout.npz stats.json out/model_0456 --signed momentum_x momentum_y --title "model, trajectory 0456"
python cfd_video.py rollout.npz stats.json out/model_0456 --signed momentum_x momentum_y --frame 40       # one png
python cfd_video.py rollout.npz stats.json out/model_0456 --signed momentum_x momentum_y --per-channel   # a video per channel
```

`rollout.npz`: `gt`, `pred` [T, C, H, W] in physical units, `times` [T], `channels` [C]. `stats.json`:
`{channel: {mean, std, min, max, p0_5, p99_5, unit}}`. Needs numpy, matplotlib and imageio-ffmpeg. From Python:
`write_video(...)` for a video, `write_frame(...)` for one png.
