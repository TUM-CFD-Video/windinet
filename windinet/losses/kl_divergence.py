"""
KL divergence loss (standard VAE regularizer).

Penalizes the encoder's posterior q(z|x) = N(mean, exp(logvar)) for
deviating from the standard normal prior N(0, I) -- the usual VAE ELBO
regularization term. Unlike the other loss components in this package,
this one does not take (pred, target) -- it needs the raw encoder
distribution instead, so it is wired into reconstruction_losses as an
optional pair of extra arguments rather than always computed. See
VaeTrainer._encode, which now returns the posterior mean/logvar alongside
the (rescaled) latents used for reconstruction.

Expected input:
    mean, logvar: same shape, e.g. [B, C, T, H, W] -- the RAW encoder
    output distribution, before the latents_mean/latents_std/scaling_factor
    affine rescale `_encode` applies for the reconstruction path. KL
    regularizes the encoder's own probabilistic output; that rescale is a
    downstream numerical-convenience transform and is not part of it.

Implementation choice (the only KL in this repo; 2026-11-21):

    1. KL is averaged over latent elements (this function), and
    2. whenever the KL weight is non-zero, training decodes a reparameterized
       posterior sample z = mu + sigma * eps (VaeTrainer._sample_posterior).
       Validation, test eval and DiT encoding still use the posterior mean.

Two earlier variants were removed:

    * Thesis variant: KL summed per sample, decoder fed the posterior MEAN.
      Without sampling, sigma never reaches the decoder, so the variance
      term costs reconstruction nothing; KL then only shrinks mu^2 and the
      model is not a VAE at all. Wrong regardless of weight.
    * Summed variant: KL summed over the ~1e5 latent elements per sample
      (128 x 13 x 8 x 8 at 256x256 / 97 frames), with sampling. This is the
      textbook ELBO, but every reconstruction loss here is a MEAN over
      pixels, so the ratio was off by ~1e5: the stock posterior's KL
      (~7.5e5) dwarfed the reconstruction loss (~0.05) even at weight 1e-7,
      and every arm collapsed to a mean prediction.

Sum and mean differ only by the constant N_latent (w_sum = w_mean / N), so
the mean is the same objective as the summed ELBO with a rescaled weight
(a beta-VAE). It is the one that matches our mean-reduced reconstruction
losses, gives weights a readable scale (old summed 1e-7 ~ mean 1e-2), and
keeps a weight meaning the same at every resolution and clip length.
Result (jupiter, s44): mean KL 1e-5 VAE 0.0798, end-to-end 0.4255 vs
baseline 0.4333; 1e-3 hurts (0.5098).
"""

import torch


def kl_divergence_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL(q(z|x) || N(0, I)) per latent element, averaged over every
    element and the batch.

    KL = 0.5 * mean(mean^2 + exp(logvar) - logvar - 1)

    Note this deliberately does NOT use diffusers' own
    DiagonalGaussianDistribution.kl(), which hardcodes `dim=[1, 2, 3]`
    (assumes 4D image latents [B, C, H, W]). This project's video-VAE
    latents are 5D [B, C, T, H, W], so that hardcoded reduction would
    silently leave one spatial axis unreduced. This implementation reduces
    over every dimension instead, whatever the input's rank.

    Args:
        mean:
            Encoder posterior mean. Shape [B, ...]

        logvar:
            Encoder posterior log-variance, same shape as mean.

    Returns:
        Scalar KL loss (nats per latent element).
    """

    # float32: exp(logvar) in bf16 loses most of its precision
    mean = mean.float()
    logvar = logvar.float()
    return 0.5 * torch.mean(mean.square() + logvar.exp() - logvar - 1.0)
