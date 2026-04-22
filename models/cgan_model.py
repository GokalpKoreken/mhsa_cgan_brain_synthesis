import click
import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import time

# Training and Validation per epoch
from trains.train import train_per_epoch_cgan
from validations.validation import validation_per_epoch_cgan

# Loading data and initial transformations
from utils.load_data import load_images

# Discriminator architecture model
from models.discriminator import Discriminator

# Generator architecture model (MHSA-enhanced UNet)
from models.generator import GeneratorUNet

# Plotting results and Data Augmentation
from utils.plot_utils import (
    plot_losses_cgan,
    plot_losses_reconst,
    plot_mean_metrics,
    plot_all_losses_cgan,
    plot_Dlosses_cgan,
)
from utils.images_utils import (
    data_augmentation_transformations,
    augmentation_args,
)

# K-fold cross-valdation
from sklearn.model_selection import KFold, StratifiedKFold

# PyTorch
import torch
from torch.utils.data import Subset

# Pytorch Lightning
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
)

# MONAI
from monai.utils import set_determinism
from monai.data import DataLoader, Dataset
from monai.losses import PerceptualLoss
from monai.networks.layers.factories import Norm


# Calculate output of image discriminator (PatchGAN) - for training set
patch_size = 32
# patch = (1, 256 // 2**4, 256 // 2**4)


def parse_list(ctx, param, value):
    return list(map(int, [x.strip() for x in value.split(",")]))


@click.command()
@click.option(
    "--dataset_path",
    type=str,
    default=os.getenv("DATASET_PATH"),
    help="Path that has the dataset synthRAD2023",
)
@click.option(
    "--best_model_path",
    type=str,
    default="result_lr_0.001_300_resunetpl",
    help="Path that has the best model so far",
)
@click.option(
    "--epoch_start", type=int, default=0, help="Epoch to start training from"
)  # change
@click.option(
    "--n_epochs", type=int, default=99, help="Number of epochs of training"
)  # change
@click.option(
    "--dataset_name",
    type=str,
    default="SynthRAD2023",
    help="Name of the dataset",
)
@click.option(
    "--batch_size_train",
    type=int,
    default=32,
    help="Size of the batches for training",
)
@click.option(
    "--batch_size_val",
    type=int,
    default=1,
    help="Size of the batches for validation",
)
@click.option(
    "--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer"
)
@click.option(
    "--lr_d",
    type=float,
    default=0.0002,
    help="Learning rate for Adam optimizer - Discriminator",
)
@click.option(
    "--b1",
    type=float,
    default=0.5,
    help="Adam: decay of first order momentum of gradient",
)
@click.option(
    "--b2",
    type=float,
    default=0.999,
    help="Adam: decay of first order momentum of gradient",
)
@click.option(
    "--checkpoint_interval",
    default=100,
    help="Interval between model checkpoints",
)
@click.option(
    "--result",
    type=str,
    default=os.getenv("RESULT_NAME", "results_lr"),
    help="Name of the folder to save results",
)  # change
@click.option(
    "--subsample_rate",
    type=int,
    default=7,
    help="Subsample rate for validation image slices",
)
@click.option(
    "--perceptual_loss",
    type=bool,
    default=True,
    help="Choose if you want to introduce the perceptual loss into the generator loss",
)
@click.option(
    "--apply_da",
    type=bool,
    default=True,
    help="Choose if you want to apply data augmentation as a regularization method",
)
@click.option(
    "--lambda_pixel",
    type=int,
    default=300,
    help="Loss weight of L1 pixel-wise loss between generated image and real image",
)
@click.option(
    "--lambda_adversarial",
    type=int,
    default=1,
    help="Loss weight of MSE adversarial loss",
)
@click.option(
    "--lambda_pl",
    type=float,
    default=10.0,
    help="Loss weight of Perceptual loss between generated image and real image",
)
@click.option(
    "--label_smoothing",
    type=bool,
    default=True,
    help="Choose if you want to apply one-sided label smoothing as a regularization method for the Discriminator",
)

@click.option(
    "--channels",
    type=str,
    default="64, 128, 256, 512, 512",
    callback=parse_list,
    help="channels of the generator",
)
@click.option(
    "--strides",
    type=str,
    default="2, 2, 2, 2",
    callback=parse_list,
    help="strides of the generator",
)
@click.option(
    "--num_res_units",
    type=int,
    default=0,
    help="num of residual units per layer",
)
@click.option(
    "--type_norm",
    type=str,
    default="BATCH",
    help='Choose between "INSTANCE" or "BATCH"',
)
@click.option(
    "--fold",
    type=int,
    default=-1,
    help="Fold to train. If -1, train all folds",
)
@click.option(
    "--region_filter",
    type=str,
    default="all",
    help="Region to train (AB, HN, TH). If 'all', train all regions",
)
@click.option(
    "--dataset_version",
    type=int,
    default=25,
    help="Dataset version (23 or 25)",
)
@click.option(
    "--use_mhsa",
    type=bool,
    default=True,
    help="Whether to integrate MHSA Blocks in deep skip connections",
)
@click.option(
    "--bone_weight",
    type=float,
    default=1.0,
    help="Multiplier penalty for MAE errors on high-intensity (bone) regions",
)
def main(
    epoch_start,
    n_epochs,
    dataset_name,
    batch_size_train,
    batch_size_val,
    lr,
    lr_d,
    b1,
    b2,
    result,
    dataset_path,
    best_model_path,
    subsample_rate,
    perceptual_loss,
    checkpoint_interval,
    apply_da,
    lambda_pixel,
    lambda_adversarial,
    lambda_pl,
    channels,
    strides,
    num_res_units,
    type_norm,
    label_smoothing,
    fold,
    region_filter,
    dataset_version,
    use_mhsa,
    bone_weight
):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    start_time = time.time()

    mae = MeanAbsoluteError().to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)

    # Loss functions
    # criterion_GAN = torch.nn.BCELoss().to(device)
    # criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_pixelwise = torch.nn.L1Loss(reduction="none").to(device)  # MAE
    
    if perceptual_loss:
        criterion_pl = PerceptualLoss(
            spatial_dims=2, network_type="radimagenet_resnet50"
        ).to(device)
    else:
        criterion_pl = None

    set_determinism(seed=42)

    '''
    # SynthRAD2023
    train_val_dataset, _, train_val_patient_folders, _ = load_images(
        dataset_path, 23
    )
    # print("train_val_dataset:", train_val_dataset)
    # print("Keys:", train_val_dataset.keys())

    combined_data = torch.utils.data.ConcatDataset(
        list(train_val_dataset.values())
    )  # pelvis data from center A and C
    all_patients_path = torch.utils.data.ConcatDataset(
        list(train_val_patient_folders.values())
    )

    a_keys = list(filter(lambda k: k[0] == "A", train_val_dataset))
    c_keys = list(filter(lambda k: k[0] == "C", train_val_dataset))

    y_centers = np.concatenate(
        [
            np.zeros(sum(len(train_val_dataset[k]) for k in a_keys)),
            np.ones(sum(len(train_val_dataset[k]) for k in c_keys)),
        ]
    )
    print("Total A samples:", sum(len(train_val_dataset[k]) for k in a_keys))
    print("Total C samples:", sum(len(train_val_dataset[k]) for k in c_keys))
    '''

    # Dataset Loading (SynthRAD2023 or 2025)
    
    train_val_dataset, _, train_val_patient_folders, _ = load_images(dataset_path, dataset_version, False)
    
    # AB+TH+HN
    if region_filter == "all":
        region_list = ["AB", "HN", "TH"]
    else:
        region_list = [region_filter]

    combined_data_list = []
    y_centers_list = []
    
    # Map center to an index for stratification
    center_map = {"A": 0, "B": 1, "C": 2}
    
    for (center, region), dataset in train_val_dataset.items():
        if region in region_list:
            combined_data_list.append(dataset)
            y_centers_list.append(np.full(len(dataset), center_map[center]))
            
    combined_data = torch.utils.data.ConcatDataset(combined_data_list)
    y_centers = np.concatenate(y_centers_list)
    
    print(f"Total samples for training/val: {len(combined_data)}")
    for center, idx in center_map.items():
        print(f"Total samples for Center {center}: {sum(y_centers == idx)}")
 
    # A+B+C 
    '''
    combined_data = torch.utils.data.ConcatDataset(list(train_val_dataset.values()))  # A+B+C 
    
    a_keys = list(filter(lambda k: k[0] == "A", train_val_dataset))
    b_keys = list(filter(lambda k: k[0] == "B", train_val_dataset))
    c_keys = list(filter(lambda k: k[0] == "C", train_val_dataset))
    
    y_centers = np.concatenate([
        np.zeros(sum(len(train_val_dataset[k]) for k in a_keys)), # A
        np.ones(sum(len(train_val_dataset[k]) for k in b_keys)), # B
        np.full(sum(len(train_val_dataset[k]) for k in c_keys), 2) # C
    ])
    print("Total A samples:", sum(len(train_val_dataset[k]) for k in a_keys))
    print("Total B samples:", sum(len(train_val_dataset[k]) for k in b_keys))
    print("Total C samples:", sum(len(train_val_dataset[k]) for k in c_keys))
    '''

    # Dictionary that saves all folds losses per epoch
    fold_train_losses_G = {}
    fold_train_losses_D = {}
    fold_train_l1_loss = {}
    fold_train_gan_loss = {}
    fold_train_pl_loss = {}
    fold_train_real_loss = {}
    fold_train_fake_loss = {}

    # Dictionary for all folds metrics
    fold_psnr_values = {}
    fold_mae_values = {}
    fold_ssim_values = {}
    fold_ms_ssim_values = {}

    # List that saves all folds losses for all epochs
    all_fold_train_losses_G = []
    all_fold_train_losses_D = []
    all_fold_train_l1_loss = []
    all_fold_train_gan_loss = []
    all_fold_train_pl_loss = []
    all_fold_train_real_loss = []
    all_fold_train_fake_loss = []

    all_fold_val_psnr = []
    all_fold_val_mae = []
    all_fold_val_ssim = []
    all_fold_val_ms_ssim = []

    # List that saves mean loss value
    fold_losses_G = []
    fold_losses_D = []

    splits = 5

    # Set up Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    for i, (train_index, val_index) in enumerate(skf.split(combined_data, y_centers)):
        if fold != -1 and i != fold:
            continue
        
        print(f"Fold {i}:")
        # Initialize discriminator
        discriminator = Discriminator()

        generator = GeneratorUNet(in_channels=1, out_channels=1, use_mhsa=use_mhsa)

        generator = generator.to(device)
        discriminator = discriminator.to(device)

        # Optimizers
        optimizer_G = torch.optim.Adam(
            generator.parameters(), lr=lr, betas=(b1, b2)
        )
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(), lr=lr_d, betas=(b1, b2)
        )

        print(f"Fold {fold}:")
        print(f"  Train set size: {len(train_index)}")
        print(
            f"  Class distribution in training set (A,B,C): {np.bincount(y_centers[train_index].astype(int))}"
        )
        print(f"  Validation set size: {len(val_index)}")
        print(
            f"  Class distribution in validation set (A,B,C): {np.bincount(y_centers[val_index].astype(int))}"
        )

        """
        # Set up k-fold cross-validation
        kfold = KFold(n_splits=splits, shuffle=True, random_state=42)

        for fold, (train_index, val_index) in enumerate(
            kfold.split(train_val_dataset)
        ):
        print(
            f"Fold:{fold},train indexes:{train_index}, val indexes:{val_index}"
        )
        
        print("Patients in training set")
        for idx in train_index:
            print(f"Index on dataset: {idx} - Patient: {patient_folder_name[idx]}")
    
        print("Patients in validation set")
        for idx in val_index:
            print(f"Index on dataset: {idx} - Patient: {patient_folder_name[idx]}")
        train_set = Subset(train_val_dataset, train_index)
        valid_set = Subset(train_val_dataset, val_index)
        """
        # This was used before
        # if fold == 0 or fold == 1 or fold == 2 or fold == 4:
        #    continue  # skip 0,1,2,4

        # This allows the training process to continue from a previous state, without starting over from scratch everytime it runs until total number of epochs
        if epoch_start != 0:
            # Load pretrained models
            print("Loading pretrained models at epoch:", epoch_start)
            # fold_path = os.path.join(f"/mnt/big_disk/catarina_caldeira/validations/saved_models/{dataset_name}/cGAN/{best_model_path}", f"fold_{fold}")

            # if os.path.isdir(fold_path):
            #   for file in os.listdir(fold_path):
            # print(file)
            #        if file.startswith("validation_fold") and file.endswith(".pth"):
            #            best_model_path_fold = os.path.join(fold_path, file)
            best_model_path_fold = "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/cGAN/7layers_unet_ININ_1000_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025_regionABHNTH/fold_3/val_bestmodel_fold3_epoch518_interval_0-599.pth"
            if best_model_path_fold:
                print(
                    f"The pretrained model for fold {fold}: {best_model_path_fold}"
                )
                checkpoint = torch.load(
                    best_model_path_fold, weights_only=False
                )
                generator.load_state_dict(checkpoint["G_state_dict"])
                discriminator.load_state_dict(checkpoint["D_state_dict"])

        # For saving the best validation model
        os.makedirs(
            f"validations/saved_models/{dataset_name}/cGAN/{result}/fold_{fold}",
            exist_ok=True,
        )

        # Initialize lists for fold losses
        fold_train_losses_G[fold] = []
        fold_train_losses_D[fold] = []
        fold_train_l1_loss[fold] = []
        fold_train_gan_loss[fold] = []
        fold_train_pl_loss[fold] = []
        fold_train_real_loss[fold] = []
        fold_train_fake_loss[fold] = []

        fold_psnr_values[fold] = []
        fold_mae_values[fold] = []
        fold_ssim_values[fold] = []
        fold_ms_ssim_values[fold] = []

        # Split the dataset
        train_set = Subset(combined_data, train_index)
        valid_set = Subset(combined_data, val_index)

        #val_patients = [all_patients_path[i] for i in val_index]
        #print(val_patients)

        # Data augmentation and MRI normalization of intensities

        transforms_train_da_crop = data_augmentation_transformations(
            **augmentation_args, training=True
        )

        monai_train_dataset = Dataset(
            data=train_set, transform=transforms_train_da_crop
        )

        transforms_val = data_augmentation_transformations(training=False)

        monai_valid_dataset = Dataset(data=valid_set, transform=transforms_val)

        train_loader = DataLoader(
            monai_train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        valid_loader = DataLoader(
            monai_valid_dataset,
            batch_size=batch_size_val,
            num_workers=0,
            pin_memory=True,
        )

        # Initialize best values
        best_epoch_val = -1
        best_global_models = (
            {}
        )  # saves the best model from epoch 0 until each checkpoint interval (0-99, 0-199, 0-299...)
        best_global_mae = float("inf")

        for epoch in range(epoch_start, n_epochs):

            # -> Training phase
            (
                train_loss_G,
                train_loss_D,
                train_loss_l1,
                train_loss_gan,
                train_loss_pl,
                train_loss_real,
                train_loss_fake,
            ) = train_per_epoch_cgan(
                generator=generator,
                discriminator=discriminator,
                train_loader=train_loader,
                device=device,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                fold=fold,
                epoch=epoch,
                n_epochs=n_epochs,
                patch_size=patch_size,
                criterion_GAN=criterion_GAN,
                criterion_pixelwise=criterion_pixelwise,
                criterion_pl=criterion_pl,
                perceptual_loss=perceptual_loss,
                lambda_pixel=lambda_pixel,
                lambda_adversarial=lambda_adversarial,
                lambda_pl=lambda_pl,
                label_smoothing=label_smoothing,
                bone_weight=bone_weight,
            )

            # Compute average training loss (loss was calculated across all batches)
            train_loss_G_avg = train_loss_G / len(train_loader)
            train_loss_D_avg = train_loss_D / len(train_loader)
            l1_loss_avg = train_loss_l1 / len(train_loader)
            gan_loss_avg = train_loss_gan / len(train_loader)
            pl_loss_avg = train_loss_pl / len(train_loader)
            real_loss_avg = train_loss_real / len(train_loader)
            fake_loss_avg = train_loss_fake / len(train_loader)

            # Save training losses per fold
            fold_train_losses_G[fold].append(train_loss_G_avg)
            fold_train_losses_D[fold].append(train_loss_D_avg)
            fold_train_l1_loss[fold].append(l1_loss_avg)
            fold_train_gan_loss[fold].append(gan_loss_avg)
            fold_train_pl_loss[fold].append(pl_loss_avg)
            fold_train_real_loss[fold].append(real_loss_avg)
            fold_train_fake_loss[fold].append(fake_loss_avg)

            # -> Validation phase
            (
                all_psnr_batches,
                all_mae_batches,
                all_ssim_batches,
                all_ms_ssim_batches,
            ) = validation_per_epoch_cgan(
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
            )

            mean_psnr_batches = np.mean(all_psnr_batches)
            mean_mae_batches = np.mean(all_mae_batches)
            # print(mean_mae_batches)
            mean_ssim_batches = np.mean(all_ssim_batches)
            # print(len(all_ms_ssim_batches))
            mean_ms_ssim_batches = np.mean(all_ms_ssim_batches)

            # saving all mean values for x batches for each epoch in each fold
            fold_psnr_values[fold].append(mean_psnr_batches)
            fold_mae_values[fold].append(mean_mae_batches)
            fold_ssim_values[fold].append(mean_ssim_batches)
            fold_ms_ssim_values[fold].append(mean_ms_ssim_batches)

            # Save metrics to CSV for persistent logging
            log_folder = os.path.join("results", dataset_name, "cGAN", result)
            os.makedirs(log_folder, exist_ok=True)
            log_file = os.path.join(log_folder, f"val_metrics_fold{fold}.csv")
            file_exists = os.path.isfile(log_file)
            with open(log_file, "a") as f:
                if not file_exists:
                    f.write("epoch,mae,psnr,ssim,ms_ssim\n")
                f.write(f"{epoch},{mean_mae_batches:.4f},{mean_psnr_batches:.4f},{mean_ssim_batches:.4f},{mean_ms_ssim_batches:.4f}\n")

            # Track the best validation model based on mean mae per 100 epochs
            if mean_mae_batches < best_global_mae:
                best_global_mae = mean_mae_batches
                best_epoch_val = epoch
                best_fold_val = fold
                best_global_models = {
                    "G_state_dict": generator.state_dict(),
                    "D_state_dict": discriminator.state_dict(),
                }
                print(
                    f"New lowest MAE observed in epoch {epoch}",
                    f"\n\tPSNR: {mean_psnr_batches}",
                    f"\n\tMAE: {mean_mae_batches}",
                    f"\n\tSSIM: {mean_ssim_batches}",
                    f"\n\tMS-SSIM: {mean_ms_ssim_batches}",
                )

            # Checkopint to save models in an interval of 100 epochs
            if (epoch + 1) % checkpoint_interval == 0 or epoch == n_epochs - 1:
                torch.save(
                    {
                        "G_state_dict": best_global_models["G_state_dict"],
                        "D_state_dict": best_global_models["D_state_dict"],
                    },
                    f"validations/saved_models/{dataset_name}/cGAN/{result}/fold_{best_fold_val}/val_bestmodel_fold{best_fold_val}_epoch{best_epoch_val}_interval_0-{epoch}.pth",
                )

                torch.save(
                    {
                        "G_state_dict": generator.state_dict(),
                        "D_state_dict": discriminator.state_dict(),
                    },
                    f"validations/saved_models/{dataset_name}/cGAN/{result}/fold_{fold}/val_checkpoints_fold{fold}_epoch{epoch}.pth",
                )

        all_fold_train_losses_D.append(fold_train_losses_D[fold])
        all_fold_train_losses_G.append(fold_train_losses_G[fold])
        all_fold_train_l1_loss.append(fold_train_l1_loss[fold])
        all_fold_train_gan_loss.append(fold_train_gan_loss[fold])
        all_fold_train_pl_loss.append(fold_train_pl_loss[fold])
        all_fold_train_real_loss.append(fold_train_real_loss[fold])
        all_fold_train_fake_loss.append(fold_train_fake_loss[fold])

        all_fold_val_psnr.append(fold_psnr_values[fold])
        all_fold_val_mae.append(fold_mae_values[fold])
        all_fold_val_ssim.append(fold_ssim_values[fold])
        all_fold_val_ms_ssim.append(fold_ms_ssim_values[fold])

        # mean of validation losses and metrics for each fold
        fold_losses_G.append(np.mean(fold_train_losses_G[fold]))
        fold_losses_D.append(np.mean(fold_train_losses_D[fold]))

        #print(
        #    f"Train: Mean generator loss for this fold: {fold_losses_G[fold]}"
        #)
        #print(
        #    f"Train: Mean discriminator loss for this fold: {fold_losses_D[fold]}"
        #)

    # Calculate and print the mean and std for all folds
    mean_loss_G = np.mean(fold_losses_G)
    std_loss_G = np.std(fold_losses_G, ddof=1)
    mean_loss_D = np.mean(fold_losses_D)
    std_loss_D = np.std(fold_losses_D, ddof=1)

    # during validation how well did the performance increased
    #print("\nFinal Results after Cross-Validation (Training):")
    #print(f"Mean Loss for generator: {mean_loss_G:.4f}±{std_loss_G:.4f}")
    #print(f"Mean Loss for discriminator: {mean_loss_D:.4f}±{std_loss_D:.4f}")

    output_folder = f"../plots/{dataset_name}/cGAN/{result}"
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(all_fold_train_losses_G)):
        plot_losses_cgan(
            all_fold_train_losses_G[i],
            all_fold_train_losses_D[i],
            os.path.join(output_folder, f"train_losses_fold{i}.png"),
        )
        plot_Dlosses_cgan(
            all_fold_train_losses_D[i],
            os.path.join(output_folder, f"train_Dlosses_fold{i}.png"),
        )
        plot_losses_reconst(
            all_fold_train_l1_loss[i],
            os.path.join(output_folder, f"train_l1_loss_fold{i}.png"),
        )

        plot_all_losses_cgan(
            all_fold_train_gan_loss[i],
            all_fold_train_l1_loss[i],
            all_fold_train_pl_loss[i],
            all_fold_train_real_loss[i],
            all_fold_train_fake_loss[i],
            os.path.join(output_folder, f"all_train_losses_fold{i}.png"),
            "Training",
        )

        plot_mean_metrics(
            all_fold_val_ssim[i],
            os.path.join(output_folder, f"fold{i}_SSIM.png"),
            "SSIM",
            "Mean validation SSIM across Epochs",
        )

        plot_mean_metrics(
            all_fold_val_ms_ssim[i],
            os.path.join(output_folder, f"fold{i}_MS-SSIM.png"),
            "MS-SSIM",
            "Mean validation MS-SSIM across Epochs",
        )

        plot_mean_metrics(
            all_fold_val_mae[i],
            os.path.join(output_folder, f"fold{i}_MAE.png"),
            "MAE",
            "Mean validation MAE across Epochs",
        )

        plot_mean_metrics(
            all_fold_val_psnr[i],
            os.path.join(output_folder, f"fold{i}_PSNR.png"),
            "PSNR",
            "Mean validation PSNR across Epochs",
        )

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    #print(f"Total time: {hours}h {minutes}m {seconds}s")

    pass


if __name__ == "__main__":
    main()

# python -m  models.cgan_model
