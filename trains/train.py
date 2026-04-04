from tqdm import tqdm
import torch


def image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:

    # Extract patches using unfold
    # For input 4D: batch_size, channels, height, width
    # On dimension 2 (height) extracts patches along the height dimension: 256/32=8
    # On dimension 3 (width) extracts patches along the width dimension: 256/32=8
    patches = image.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    # print(f"Patches {patches.shape}") # torch.Size([16, 1, 8, 8, 32, 32])

    # Reshape to (Batch, Channels, num_patches, patch_size, patch_size)
    B, C, H_p, W_p, _, _ = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        B, H_p * W_p, C, patch_size, patch_size
    )
    # print(f"Patches 1 {patches.shape}") # torch.Size([16, 64, 1, 32, 32])
    patches = patches.reshape(-1, patches.shape[2], patch_size, patch_size)
    # print(f"Patches 2 {patches.shape}") # torch.Size([1024, 1, 32, 32])

    return patches


def train_per_epoch_gen(
    criterion, device, model, train_loader, optimizer, fold, epoch, n_epochs
):

    model.train()
    train_loss = 0.0

    with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        for step, batch in pbar:

            images_mri = batch["mr"].to(device)
            images_ct = batch["ct"].to(device)
            mask = batch["mask"].to(device)

            image_mri = images_mri.squeeze(-1)
            image_ct = images_ct.squeeze(-1)
            mask = mask.squeeze(-1)  # binary tensor (0s and 1s)
            optimizer.zero_grad()

            preds = model(image_mri)

            # Compute loss
            loss = criterion(preds, image_ct)
            # print(loss.shape)
            # print(f"LOSS: {loss}")
            masked_loss = (
                loss * mask.float()
            )  # mask.shape = 16,1,256,256; loss.shape = 16,1,256,256
            # print(f"Masked loss: {masked_loss}")
            loss = (
                masked_loss.sum()
            )  # ignoring the elements where mask is 0 (background) - summing up the remaining unmasked elements (where the info is)
            # print(f"LOSS after multiplying by mask values: {loss}")
            non_zero_elements = (
                mask.sum()
            )  # number of non-zero elements in the mask which is the number of elements contributing to the loss)
            # print(f"Non zero elements: {non_zero_elements}")
            # print(f"Total elements: {mask.numel()}")
            mae_loss = loss / non_zero_elements  # mse over the unmasked region
            # print(f"LOSS after normalizing the loss by the number of non-zero elements: {mae_loss}")

            mae_loss.backward()
            optimizer.step()

            # Update running training loss
            train_loss += mae_loss.item()

            pbar.set_description(
                f"TRAIN [Fold:{fold}] [Epoch {epoch}/{n_epochs}] "
                f"[Batch {step}/{len(train_loader)}] "
                f"[G loss: {mae_loss.item():.3f}]"
            )

    return train_loss


def train_per_epoch_cgan(
    generator,
    discriminator,
    train_loader,
    device,
    optimizer_G,
    optimizer_D,
    fold,
    epoch,
    n_epochs,
    patch_size,
    criterion_GAN,
    criterion_pixelwise,
    criterion_pl,
    perceptual_loss,
    lambda_pixel,
    lambda_adversarial,
    lambda_pl,
    label_smoothing: bool = False,
    bone_weight: float = 1.0,
):

    generator.train()
    discriminator.train()
    train_loss_G = 0.0
    train_loss_D = 0.0
    train_loss_l1 = 0.0
    train_loss_gan = 0.0
    train_loss_pl = 0.0
    train_loss_real = 0.0
    train_loss_fake = 0.0
    count = 0

    with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        for step, batch in pbar:
            # Model inputs
            real_A = batch["mr"].to(device)  # Shape: [B, C, H, W, D], C=1, D=1
            real_B = batch["ct"].to(device)  # Shape: [B, C, H, W, D]
            mask = batch["mask"].to(device)

            real_A = real_A.squeeze(-1)  # Shape: [B, 1, H, W]
            real_B = real_B.squeeze(-1)
            # print(f"batch {step} has CT shape: {real_B.shape}")
            mask = mask.squeeze(-1)
            
            '''
            # Skipp all black slices from CT or sCT
            if torch.all(real_A == 0) or torch.all(real_B == 0):
                print(torch.unique(real_A))
                print(torch.unique(real_B))
                print(f"Skipping slice at step {step}")
                count += 1
                print(
                    f"Total skipped batches on epoch {epoch}: {count}/{len(train_loader)}"
                )
                continue
            '''

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            # print(f"MRI GT in training: {torch.unique(real_A)}")
            # print(f"CT GT in training: {torch.unique(real_B)}")
            # print(f"sCT obtained in training: {torch.unique(fake_B)}")

            # Create patches
            fake_B_patches = image_to_patches(fake_B * mask.float(), patch_size)
            # Already masked
            real_A_patches = image_to_patches(real_A, patch_size)
            real_B_patches = image_to_patches(real_B, patch_size)

            # Original labels (no smoothing)
            valid = torch.ones(
                (fake_B_patches.shape[0], 1),
                device=device,
                requires_grad=False,
            )  # label of real images = 1's
            
            if label_smoothing:
                valid *= 0.9

            # Adversarial ground truths
            fake = torch.zeros(
                (real_A_patches.shape[0], 1),
                device=device,
                requires_grad=False,
            )  # label of fake images = 0's

            pred_fake = discriminator(
                fake_B_patches, real_A_patches
            )  # FAKE INPUT (because this contains the generated image by the generator)

            loss_GAN = criterion_GAN(
                pred_fake, valid
            )  # img,label (how well the generator fooled the discriminator)
            # print(f"G gan loss:{loss_GAN}")

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            # print(f"LOSS: {loss_pixel}")
            
            # Apply bone-weighted penalty if specified
            if bone_weight > 1.0:
                # Dense bone is roughly HU > 300, mapping to normalized CT > -0.34
                bone_mask = (real_B > -0.34).float()
                anatomy_weights = torch.ones_like(real_B) + (bone_mask * (bone_weight - 1.0))
                masked_loss = loss_pixel * mask.float() * anatomy_weights
            else:
                masked_loss = loss_pixel * mask.float()
                
            # print(f"Loss for masked: {masked_loss}")
            loss_pixel = masked_loss.sum()
            # print(f"Loss for masked SUM: {loss_pixel}")

            non_zero_elements = mask.sum()
            l1_loss_val = loss_pixel / non_zero_elements
            # print(f"LOSS after normalizing the loss by the number of non-zero elements: {l1_loss_val}")

            # Total loss
            loss_G = lambda_adversarial * loss_GAN + lambda_pixel * l1_loss_val
            if perceptual_loss is True:
                # Perceptual Loss
                p_loss = criterion_pl(fake_B * mask.float(), real_B)
                loss_G += lambda_pl * p_loss

                train_loss_pl += p_loss.item()

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B_patches, real_A_patches)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B_patches.detach(), real_A_patches)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            train_loss_G += loss_GAN.item()
            train_loss_D += loss_D.item()
            train_loss_l1 += l1_loss_val.item()
            train_loss_gan += loss_GAN.item()

            train_loss_real += loss_real.item()
            train_loss_fake += loss_fake.item()

            description = [
                "TRAIN",
                f"[Fold:{fold}]",
                f"[Epoch:{epoch}]",
                f"[Batch:{step}]",
                f"[D loss:{train_loss_D / (step + 1):.3f}]",
                f"[G loss:{train_loss_G / (step + 1):.3f}]",
                f"[pixel:{train_loss_l1 / (step + 1):.3f}]",
            ]
            if perceptual_loss is True:
                description.append(f"[perceptual:{p_loss:.3f}]")
            pbar.set_description(" ".join(description))

    return (
        train_loss_G,
        train_loss_D,
        train_loss_l1,
        train_loss_gan,
        train_loss_pl,
        train_loss_real,
        train_loss_fake,
    )
