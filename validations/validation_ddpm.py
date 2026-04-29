"""
Per-epoch DDIM validation for the Swin-DDPM model.

Mirrors the control flow of `validations/validation.py` used by the cGAN:
subsamples slices to keep validation cost low, computes PSNR / MAE / SSIM /
MS-SSIM on the masked anatomy, and returns the same four lists.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch
from tqdm import tqdm


@torch.no_grad()
def validation_per_epoch_ddpm(
    diffusion,                 # GaussianDiffusion
    valid_loader,
    device: torch.device,
    fold: int,
    epoch: int,
    subsample_rate: int,
    dataset_name: str,
    result: str,
    psnr_metric,
    mae_metric,
    ssim_metric,
    ms_ssim_metric,
    num_ddim_steps: int = 50,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Run DDIM sampling on (subsampled) validation slices and compute metrics."""
    diffusion.eval()

    all_psnr, all_mae, all_ssim, all_ms_ssim = [], [], [], []

    with tqdm(enumerate(valid_loader), total=len(valid_loader)) as pbar:
        for step, batch in pbar:
            images_mri = batch["mr"].to(device)
            images_ct = batch["ct"].to(device)
            mask = batch["mask"].to(device)

            # Strip MONAI MetaTensor wrapper before reshaping/slicing so that
            # the per-slice metadata dicts do not need to be re-collated.
            if hasattr(images_mri, "as_tensor"):
                images_mri = images_mri.as_tensor()
            if hasattr(images_ct, "as_tensor"):
                images_ct = images_ct.as_tensor()
            if hasattr(mask, "as_tensor"):
                mask = mask.as_tensor()

            # MONAI slice-wise loader convention (see validation.py)
            images_mri = torch.swapaxes(images_mri, -1, 0)
            images_ct  = torch.swapaxes(images_ct,  -1, 0)
            mask       = torch.swapaxes(mask,       -1, 0)

            mri  = images_mri.squeeze(-1)
            ct   = images_ct.squeeze(-1)
            m    = mask.squeeze(-1).float()

            # Subsample slices for speed (DDIM sampling is expensive)
            if subsample_rate and subsample_rate > 1:
                mri = mri[::subsample_rate]
                ct  = ct[::subsample_rate]
                m   = m[::subsample_rate]

            if mri.shape[0] == 0:
                continue

            # After swapaxes/squeeze/slice the tensors are non-contiguous;
            # torchmetrics internally calls `.view(-1)` which requires
            # contiguity (otherwise a RuntimeError is raised).
            mri = mri.contiguous()
            ct  = ct.contiguous()
            m   = m.contiguous()

            sct = diffusion.sample(mri, num_steps=num_ddim_steps)
            sct = sct.clamp(-1.0, 1.0)

            # Apply mask before metrics so background air does not dominate
            sct_m = (sct * m).contiguous()
            ct_m  = (ct  * m).contiguous()

            # Reset so per-volume metrics are not accumulated across patients
            psnr_metric.reset()
            mae_metric.reset()
            ssim_metric.reset()
            ms_ssim_metric.reset()

            all_psnr.append(float(psnr_metric(sct_m, ct_m).item()))
            all_mae.append(float(mae_metric(sct_m, ct_m).item()))
            all_ssim.append(float(ssim_metric(sct_m, ct_m).item()))
            all_ms_ssim.append(float(ms_ssim_metric(sct_m, ct_m).item()))

            pbar.set_description(
                f"VAL-DDPM [Fold:{fold}] [Epoch {epoch}] "
                f"[Batch {step}/{len(valid_loader)}] "
                f"[PSNR {all_psnr[-1]:.2f} SSIM {all_ssim[-1]:.3f}]"
            )

    # Optional: save a small preview grid for qualitative tracking
    try:
        out_dir = os.path.join(
            "validations", "test_images", dataset_name, "ddpm", result, f"fold_{fold}"
        )
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass  # non-fatal – preview saving is optional

    return all_psnr, all_mae, all_ssim, all_ms_ssim
