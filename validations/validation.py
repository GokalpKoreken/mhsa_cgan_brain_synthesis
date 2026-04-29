import os
import torch
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer
from utils.images_utils import save_generated_images, save_generated_nifti_mha
from utils.images_utils import from_normalize_to_HU


def characterise_distribution(X):
    return {
        "shape": X.shape,
        "sum": X.sum(),
        "mean": X.mean(),
        "std": X.std(),
        "min": X.min(),
        "max": X.max(),
        "range": X.max() - X.min(),
    }


def image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:

    # Extract patches using unfold
    patches = image.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )

    # Reshape to (Batch, Channels, num_patches, patch_size, patch_size)
    B, C, H_p, W_p, _, _ = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        B, H_p * W_p, C, patch_size, patch_size
    )
    patches = patches.reshape(-1, patches.shape[2], patch_size, patch_size)

    return patches


def validation_per_epoch_gen(
    device,
    model,
    subsample_rate,
    criterion,
    valid_loader,
    fold,
    epoch,
    n_epochs,
    dataset_name,
    result,
    psnr,
    mae,
    ssim,
    ms_ssim,
):

    model.eval()

    val_loss = 0.0
    all_psnr_batches = []
    all_mae_batches = []
    all_ssim_batches = []
    all_ms_ssim_batches = []

    with tqdm(enumerate(valid_loader), total=len(valid_loader)) as pbar:
        with torch.no_grad():
            for step, batch in pbar:

                images_mri = batch["mr"].to(device)
                images_ct = batch["ct"].to(device)
                mask = batch["mask"].to(device)

                images_mri = torch.swapaxes(images_mri, -1, 0)
                images_ct = torch.swapaxes(images_ct, -1, 0)
                mask = torch.swapaxes(mask, -1, 0)

                image_mri = images_mri.squeeze(-1)
                image_ct = images_ct.squeeze(-1)
                mask = mask.squeeze(-1)  # binary tensor (0s and 1s)
                # print(torch.unique(image_mri))
                # print(torch.unique(image_ct))
                # print(torch.unique(mask))
                # print(f"outside mask: {image_mri[mask==0]}")
                # print(f"outside mask: {image_ct[mask==0]}")
                
                # --------------------
                #  Validate Generator
                # --------------------

                # Subsample slices (takes very long otherwise) - from the first to the last slice, subsample_rate to subsample_rate
                image_mri = image_mri.as_tensor()[::subsample_rate, :, :, :]
                image_ct = image_ct.as_tensor()[::subsample_rate, :, :, :]
                mask = mask.as_tensor()[::subsample_rate, :, :, :]

                inferer = SlidingWindowInferer(
                    roi_size=(256, 256),
                    mode="constant",
                    progress=False,
                    overlap=0.0,
                )

                preds = inferer(image_mri, model)
                masked_sct = preds * mask.float()
                # print(f"Masked sct: {masked_sct.shape}")

                # Compute validation loss
                loss = criterion(preds, image_ct)
                masked_loss = loss * mask.float()
                loss = masked_loss.sum()
                non_zero_elements = mask.sum()
                mae_loss_val = loss / non_zero_elements

                # Metrics for a specific batch 
                psnr_per_batch = psnr(masked_sct, image_ct).item()
                mask_bool = mask == 1
                mae_per_batch = mae(
                    masked_sct[mask_bool], image_ct[mask_bool]
                ).item()
                ssim_per_batch = ssim(masked_sct, image_ct).item()
                ms_ssim_per_batch = ms_ssim(masked_sct, image_ct).item()

                # Metrics for a specific fold
                all_psnr_batches.append(psnr_per_batch)
                all_mae_batches.append(mae_per_batch)
                all_ssim_batches.append(ssim_per_batch)
                all_ms_ssim_batches.append(ms_ssim_per_batch)

                val_loss += mae_loss_val.item()

                pbar.set_description(
                    "VALID [Fold:{}] [Epoch {}/{}] [Batch {}/{}]".format(
                        fold, epoch, n_epochs, step, len(valid_loader)
                    )
                )

                # Saves the first image/batch each 10 epochs
                '''
                if step == 0 and epoch % 10 == 0:
                    save_generated_images(
                        image_mri,
                        image_ct,
                        masked_sct,
                        fold,
                        dataset_name,
                        result,
                        None,
                        epoch,
                        "generator",
                        "all",
                        "images",
                    )
                '''

    return (
        val_loss,
        all_psnr_batches,
        all_mae_batches,
        all_ssim_batches,
        all_ms_ssim_batches,
    )


def validation_per_epoch_cgan(
    generator,
    valid_loader,
    device,
    fold,
    epoch,
    subsample_rate,
    dataset_name,
    result,
    psnr,
    mae,
    ssim,
    ms_ssim,
):
    generator.eval()

    all_psnr_batches = []
    all_mae_batches = []
    all_ssim_batches = []
    all_ms_ssim_batches = []

    with tqdm(enumerate(valid_loader), total=len(valid_loader)) as pbar:
        with torch.no_grad():
            for step, batch in pbar:

                # Model inputs
                real_A = batch["mr"].to(device)  # Shape: [B, C, H, W, D], C=1, B=1
                real_B = batch["ct"].to(device)  # Shape: [B, C, H, W, D]
                mask = batch["mask"].to(device)

                real_A = torch.swapaxes(real_A, -1, 0)  # Shape: [D, C, H, W, B], C=1, B=1
                real_B = torch.swapaxes(real_B, -1, 0)
                mask = torch.swapaxes(mask, -1, 0)

                real_A = real_A.squeeze(
                    -1
                )  # Shape: [D, 1, H, W] -> n(=D) images 2D
                real_B = real_B.squeeze(-1)
                mask = mask.squeeze(-1)

                # --------------------
                #  Validate Generator
                # --------------------

                # Subsample slices (takes very long otherwise)
                real_A = real_A.as_tensor()[::subsample_rate, :, :, :]
                real_B = real_B.as_tensor()[::subsample_rate, :, :, :]
                mask = mask.as_tensor()[::subsample_rate, :, :, :]

                # --------------------
                # Slice Filtering: Identify slices with anatomical content (>1% occupancy)
                # --------------------
                slice_occupancy = mask.sum(dim=(1, 2, 3)) / (mask.shape[2] * mask.shape[3])
                meaningful_indices = (slice_occupancy > 0.01).squeeze()
                
                # GAN loss
                inferer = SlidingWindowInferer(
                    roi_size=(256, 256),
                    mode="constant",
                    progress=False,
                    overlap=0.0,
                )

                fake_B = inferer(real_A, generator)
                # print(f"MRI GT in validation: {torch.unique(real_A)}")
                # print(f"CT GT in validation: {torch.unique(real_B)}")
                # print(f"sCT obtained in validation: {torch.unique(fake_B)}")

                masked_sct = fake_B * mask.float()

                # Metrics for a specific batch - Apply filtering here
                if meaningful_indices.sum() > 0:
                    psnr_per_batch = psnr(masked_sct[meaningful_indices], real_B[meaningful_indices]).item()
                    ssim_per_batch = ssim(masked_sct[meaningful_indices], real_B[meaningful_indices]).item()
                    ms_ssim_per_batch = ms_ssim(masked_sct[meaningful_indices], real_B[meaningful_indices]).item()
                else:
                    # Fallback if no slices are "meaningful" (should be rare)
                    psnr_per_batch = psnr(masked_sct, real_B).item()
                    ssim_per_batch = ssim(masked_sct, real_B).item()
                    ms_ssim_per_batch = ms_ssim(masked_sct, real_B).item()
                
                # Verification to see if it's "nan"
                if psnr_per_batch != psnr_per_batch:  
                    print(f"mask: {characterise_distribution(mask)}")
                    print(
                        f"masked_sct: {characterise_distribution(masked_sct)}"
                    )
                    print(f"real_B: {characterise_distribution(real_B)}")
                mask_bool = mask == 1
                mae_per_batch = mae(
                    masked_sct[mask_bool], real_B[mask_bool]
                ).item()
                # print(f"ms_ssim: {ms_ssim_per_batch}")

                # Metrics for a specific fold
                all_psnr_batches.append(psnr_per_batch)
                all_mae_batches.append(mae_per_batch)
                all_ssim_batches.append(ssim_per_batch)
                all_ms_ssim_batches.append(ms_ssim_per_batch)

                pbar.set_description(
                    "VALID [Fold:{}] [Batch {}/{}]".format(
                        fold, step, len(valid_loader)
                    )
                )

                # if step == 0 and epoch % 10 == 0:
                #    save_generated_images(real_A, real_B, masked_sct, fold, dataset_name, result, None, epoch, "cGAN", "all", "images")

    return (
        all_psnr_batches,
        all_mae_batches,
        all_ssim_batches,
        all_ms_ssim_batches,
    )


def best_validation_model(
    generator,
    valid_loader,
    fold,
    dataset_name,
    result,
    psnr,
    mae,
    ssim,
    ms_ssim,
    device,
    model_type,
    mode,
    all_test_patients_path,
    file_format
):
    generator.eval()

    all_psnr_per_batch = []
    all_mae_per_batch = []
    all_ssim_per_batch = []
    all_ms_ssim_per_batch = []
    all_preds = []
    all_reals = []
    all_masks = []

    # with tqdm(enumerate(valid_loader), total=len(valid_loader)) as pbar:
    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            # Model inputs
            real_A = batch["mr"].to(device)  # Shape: [B, C, H, W, D], C=1, B=1
            real_B = batch["ct"].to(device)  # Shape: [B, C, H, W, D]
            mask = batch["mask"].to(device)

            real_A = torch.swapaxes(real_A, -1, 0)  # Shape: [D, C, H, W, B], C=1, B=1
            real_B = torch.swapaxes(real_B, -1, 0)
            mask = torch.swapaxes(mask, -1, 0)

            real_A = real_A.squeeze(-1)  # Shape: [D, 1, H, W] -> n(=D) images 2D
            real_B = real_B.squeeze(-1)
            mask = mask.squeeze(-1)

            # ----------------
            #  Test Generator
            # ----------------

            # --------------------
            # Slice Filtering: Identify slices with anatomical content (>1% occupancy)
            # --------------------
            slice_occupancy = mask.sum(dim=(1, 2, 3)) / (mask.shape[2] * mask.shape[3])
            meaningful_indices = (slice_occupancy > 0.01).squeeze()
            
            patient_name = all_test_patients_path[step].split("/")[-1]
            
            # ----------------
            # Inference
            # ----------------
            print(f"--- Starting inference for patient {patient_name}...", flush=True)

            if mode == "test":
                inferer = SlidingWindowInferer(
                    roi_size=(256, 256),
                    sw_batch_size=1,
                    mode="gaussian",
                    progress=False,
                    overlap=0.25,
                )
            elif mode == "validation":
                inferer = SlidingWindowInferer(
                    roi_size=(256, 256),
                    sw_batch_size=1,
                    mode="constant",
                    progress=False,
                    overlap=0.0,
                )

            fake_B = inferer(real_A, generator)
            print(f"--- Inference DONE for patient {patient_name}.", flush=True)

            fake_B_mask = fake_B * mask.float()  # masked prediction
            
            # 2D normalise not masked (besides MAE)
            mask_bool = (mask == 1) 
            fake_B_masks_modified = torch.where(mask == 1, fake_B_mask, -1.0)
            fake_B_mask_HU = from_normalize_to_HU(fake_B_masks_modified)
            real_B_HU = from_normalize_to_HU(real_B)
            
            p_comp = fake_B_mask_HU[mask_bool]
            t_comp = real_B_HU[mask_bool]

            mae_per_batch = mae(p_comp, t_comp).item()
            
            if meaningful_indices.sum() > 0:
                psnr_per_batch = psnr(fake_B_mask[meaningful_indices], real_B[meaningful_indices]).item()
                ssim_per_batch = ssim(fake_B_mask[meaningful_indices], real_B[meaningful_indices]).item()
                ms_ssim_per_batch = ms_ssim(fake_B_mask[meaningful_indices], real_B[meaningful_indices]).item()
            else:
                psnr_per_batch = psnr(fake_B_mask, real_B).item()
                ssim_per_batch = ssim(fake_B_mask, real_B).item()
                ms_ssim_per_batch = ms_ssim(fake_B_mask, real_B).item()
            
            # Metrics for a specific fold
            all_psnr_per_batch.append(psnr_per_batch)
            all_mae_per_batch.append(mae_per_batch)
            all_ssim_per_batch.append(ssim_per_batch)
            all_ms_ssim_per_batch.append(ms_ssim_per_batch)
            
            # Verification to see if it's "nan"
            if psnr_per_batch != psnr_per_batch:
                print(f"mask: {characterise_distribution(mask)}")
                print(f"fake_B: {characterise_distribution(fake_B)}")
                print(
                    f"masked_sct: {characterise_distribution(fake_B_mask)}"
                )
                print(f"real_B: {characterise_distribution(real_B)}")
                print(f"real_A: {characterise_distribution(real_A)}")
               
            
            patient_name = all_test_patients_path[step].split("/")[-1]
            patient_mr_path = os.path.join(all_test_patients_path[step], f"mr.{file_format}")
            
            print(" ")
            print(f"---Patient {patient_name}---")
            print(f"PSNR: {psnr_per_batch}")
            print(f"MAE HU MASKED: {mae_per_batch}")
            print(f"SSIM: {ssim_per_batch}")
            print(f"MS-SSIM: {ms_ssim_per_batch}")
            
            all_preds.append(fake_B_mask.cpu())
            all_reals.append(real_B.cpu())
            all_masks.append(mask.cpu())

            if mode == "test":
                print(f"--- Saving NIfTI/MHA for patient {patient_name}...", flush=True)
                # Without clipping in HU
                # NifTi
                save_generated_nifti_mha(
                    "tests",
                    patient_mr_path,
                    fake_B_mask_HU,
                    fold,
                    dataset_name,
                    result,
                    model_type,
                    "sCT_generated",
                    patient_name,
                    file_format="nii.gz",
                )
                
                # MHA decompressed
                save_generated_nifti_mha(
                    "tests",
                    patient_mr_path,
                    fake_B_mask_HU,
                    fold,
                    dataset_name,
                    result,
                    model_type,
                    "sCT_generated",
                    patient_name,
                    file_format="mha",
                )

                # With clipping in HU (just to image analysis purpose)
                # PNG
                save_generated_images(
                    real_A,
                    real_B,
                    fake_B_mask,
                    fold,
                    dataset_name,
                    result,
                    patient_name,
                    None,
                    model_type,
                    "best",
                    "test_images",
                )
                print(f"--- PNG SAVED for patient {patient_name}.", flush=True)
                print(".")
            else:
                print(".")

    return (
        all_psnr_per_batch,
        all_mae_per_batch,
        all_ssim_per_batch,
        all_ms_ssim_per_batch,
        all_preds,
        all_reals,
        all_masks,
    )


def mean_image_across_folds(
    all_preds,
    all_real,
    all_masks,
    dataset_name,
    result,
    psnr,
    mae,
    ssim,
    ms_ssim,
    all_patients_path,
    file_format
):  

    # Fold 1
    first_fold = list(all_preds.keys())[0]
    num_images = len(all_preds[first_fold]) # all_preds already has the mask
    
    all_psnr_ensemble = []
    all_mae_ensemble = []
    all_ssim_ensemble = []
    all_ms_ssim_ensemble = []

    for img_idx in range(num_images):
        # collect a certain img from all folds (generated sCT)
        img_across_folds = [all_preds[fold][img_idx] for fold in all_preds.keys()]
        
        if not img_across_folds:
            continue
        
        mean_img = torch.stack(img_across_folds).mean(
            dim=0
        )  # new image - mean of all x images across the 5 folds (with all slices) - D,1,H,W
        
        # 3D HU masked (MAE and PSNR), 3D norm (SSIM and MS-SSIM)
        '''
        x = torch.swapaxes(mean_img, 0, 1)
        mean_img_3D = torch.unsqueeze(x, 0) # (1, 1, H, W, D)
        #mean_img_3D_HU =from_normalize_to_HU(mean_img_3D)
        print(f"shape mean sct 3D imgs: {mean_img_3D.shape}") #  - 1,H,W,D
        
        # corresponding real image and mask (same from any fold, in this case fold 1)
        real_img = all_real[1][img_idx]
        x_real = torch.swapaxes(real_img, 0, 1)
        real_img_3D = torch.unsqueeze(x_real, 0) # (1, 1, H, W, D)
        #real_img_3D_HU =from_normalize_to_HU(real_img_3D)
        print(f"shape real ct img: {real_img_3D.shape}") # torch.Size([134, 1, 587, 527])
        
        mask = all_masks[1][img_idx]
        x_mask = torch.swapaxes(mask, 0, 1)
        mask_3D = torch.unsqueeze(x_mask, 0)
        
        mean_img_modified = torch.where(mask_3D == 1, mean_img_3D, -1.0)
        mean_img_HU = from_normalize_to_HU(mean_img_modified)
        real_img_HU = from_normalize_to_HU(real_img_3D)

        mask_bool = mask_3D == 1
        mean_img_mae = mae(
            mean_img_HU[mask_bool], real_img_HU[mask_bool]
        ).item()
        
        mean_img_psnr = psnr(mean_img_HU[mask_bool], real_img_HU[mask_bool]).item()
        mean_img_ssim = ssim(mean_img, real_img).item()
        mean_img_ms_ssim = ms_ssim(mean_img, real_img).item()
        '''
        
        # 2D normalise not masked (besides MAE)
        
        mask = all_masks[first_fold][img_idx]
        real_img = all_real[first_fold][img_idx]
        
        mask_bool = (mask == 1) 
        fake_B_mask_HU = from_normalize_to_HU(mean_img_modified)
        real_B_HU = from_normalize_to_HU(real_img)
        
        # DEBUG
        # print(f"DEBUG: patient {patient_name}")
        # print(f"DEBUG: mean_img shape: {mean_img.shape}")
        # print(f"DEBUG: real_img shape: {real_img.shape}")
        # print(f"DEBUG: mask shape: {mask.shape}")
        # print(f"DEBUG: mask_bool shape: {mask_bool.shape}")
        # print(f"DEBUG: mask_bool sum: {mask_bool.sum()}")
        
        try:
            val_comp = fake_B_mask_HU[mask_bool]
            real_comp = real_B_HU[mask_bool]
            # print(f"DEBUG: val_comp shape: {val_comp.shape}")
            # print(f"DEBUG: real_comp shape: {real_comp.shape}")
        except Exception as e:
            print(f"ERROR patient {patient_name}: {e}")
            print(f"DEBUG: mean_img shape: {mean_img.shape}")
            print(f"DEBUG: real_img shape: {real_img.shape}")
            print(f"DEBUG: mask shape: {mask.shape}")
            print(f"DEBUG: mask_bool shape: {mask_bool.shape}")
        mean_img_mae = mae(
                        fake_B_mask_HU[mask_bool], real_B_HU[mask_bool]
                    ).item()
        mean_img_psnr = psnr(mean_img, real_img).item()
        mean_img_ssim = ssim(mean_img, real_img).item()
        mean_img_ms_ssim = ms_ssim(mean_img, real_img).item()
        
        
        patient_name = all_patients_path[img_idx].split("/")[-1]
        patient_mr_path = os.path.join(all_patients_path[img_idx], f"mr.{file_format}") 

        print("\nEnsemble of models - grouped by different HP")
        print(
            f"\nFor patient {patient_name} - Mean image across all folds vs real image:"
        )
        print(
            f"PSNR: {mean_img_psnr:.4f}  MAE (HU): {mean_img_mae:.4f}  SSIM: {mean_img_ssim:.4f} MS-SSIM: {mean_img_ms_ssim:.4f} \n"
        )
        
        # MHA decompressed
        '''
        save_generated_nifti_mha(
            "tests",
            patient_mr_path,
            fake_B_mask_HU,
            "all",
            dataset_name,
            result,
            "AE",
            "mean_sCTs_generated_GaussianOverlap_centers",
            patient_name,
            file_format="mha",
        )
        '''

        all_psnr_ensemble.append(mean_img_psnr)
        all_mae_ensemble.append(mean_img_mae)
        all_ssim_ensemble.append(mean_img_ssim)
        all_ms_ssim_ensemble.append(mean_img_ms_ssim)
        
    return (
        all_psnr_ensemble,
        all_mae_ensemble,
        all_ssim_ensemble,
        all_ms_ssim_ensemble,
    )
