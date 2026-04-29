import click
import os
from tqdm import tqdm
import numpy as np

# Loading data and initial transformations
from utils.load_data import load_images

# Generator architecture model
from models.generator_monai import generator_unet

# Plotting results
from utils.plot_utils import plot_metrics

from validations.validation import best_validation_model

# K-fold cross-valdation
from sklearn.model_selection import StratifiedKFold, KFold

# PyTorch
import torch
from torch.utils.data import Subset

# Pytorch Lightning
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure,MultiScaleStructuralSimilarityIndexMeasure

# MONAI
from monai.utils import set_determinism
from monai.data import DataLoader,Dataset
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    MaskIntensityd,
    ScaleIntensityRanged,
    ToTensord,
    RandSpatialCropd,
    SpatialPadd,
    Lambdad,
)
from monai.networks.layers.factories import Norm

from collections import defaultdict

def parse_list(ctx, param, value):
    return list(map(int, value.split(",")))
        
@click.command()
@click.option('--dataset_path', type=str, default="/home/catarina_caldeira/Imagens/SynthRAD2023dataset/Task1/pelvis", help='Path that has the dataset')
@click.option('--best_models_path', type=str, default="/mnt/big_disk/catarina_caldeira/validations/saved_models/SynthRAD2023/generator/result_lr_0.0005_600_resunet_da", help='Path where best models are') 
@click.option('--batch_size_val', type=int, default=1, help='Size of the batches for validation')
@click.option('--channels', type=str, default="64, 128, 256, 512, 512", callback=parse_list, help='channels of the generator')
@click.option('--strides', type=str, default="2, 2, 2, 2", callback=parse_list, help='strides of the generator')
@click.option('--num_res_units', type=int, default=0, help='num of residual units per layer')
@click.option('--type_norm', type=str, default="BATCH", help='Choose between "INSTANCE" or "BATCH"')
@click.option('--ending_saved_model', type=str, default="0-99.pth", help='Choose until which best model epoch you want to validate')


def main(best_models_path, batch_size_val, dataset_path, channels, strides, type_norm, num_res_units, ending_saved_model):
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu') 

    set_determinism(seed=42)

    # SynthRAD2023 
    
    train_val_dataset, _, train_val_patient_folders, _ = load_images(dataset_path, 23, True)
    #print("train_val_dataset:", train_val_dataset)
    #print("Keys:", train_val_dataset.keys())

    combined_data = torch.utils.data.ConcatDataset(list(train_val_dataset.values())) # pelvis data from center A and C
    all_patients_path = torch.utils.data.ConcatDataset(list(train_val_patient_folders.values())) 
    #print(all_patients_path)
    
    a_keys = list(filter(lambda k: k[0] == "A", train_val_dataset))
    c_keys = list(filter(lambda k: k[0] == "C", train_val_dataset))

    y = np.concatenate([
        np.zeros(sum(len(train_val_dataset[k]) for k in a_keys)),
        np.ones(sum(len(train_val_dataset[k]) for k in c_keys)),
    ])
    
    
     # A+B+C 
    '''
    train_val_dataset, _, train_val_patient_folders, _ = load_images(
        dataset_path, 25, True
    )
    combined_data = torch.utils.data.ConcatDataset(list(train_val_dataset.values()))  # A+B+C 
    all_patients_path = torch.utils.data.ConcatDataset(
        list(train_val_patient_folders.values())
    )
    
    a_keys = list(filter(lambda k: k[0] == "A", train_val_dataset))
    b_keys = list(filter(lambda k: k[0] == "B", train_val_dataset))
    c_keys = list(filter(lambda k: k[0] == "C", train_val_dataset))
    
    y = np.concatenate([
        np.zeros(sum(len(train_val_dataset[k]) for k in a_keys)), # A
        np.ones(sum(len(train_val_dataset[k]) for k in b_keys)), # B
        np.full(sum(len(train_val_dataset[k]) for k in c_keys), 2) # C
    ])
    print("Total A samples:", sum(len(train_val_dataset[k]) for k in a_keys))
    print("Total B samples:", sum(len(train_val_dataset[k]) for k in b_keys))
    print("Total C samples:", sum(len(train_val_dataset[k]) for k in c_keys))
    '''
    
    # AB+TH+HN
    '''
    train_val_dataset, _, train_val_patient_folders, _ = load_images(
        dataset_path, 25, True
    )
    # Filter keys by anatomical region
    a_keys = [k for k in train_val_dataset if k[1] == "AB"]
    b_keys = [k for k in train_val_dataset if k[1] == "TH"]
    c_keys = [k for k in train_val_dataset if k[1] == "HN"]

    # Extract datasets per region
    a_data = [train_val_dataset[k] for k in a_keys]
    b_data = [train_val_dataset[k] for k in b_keys]
    c_data = [train_val_dataset[k] for k in c_keys]
    
    combined_data = torch.utils.data.ConcatDataset(a_data+b_data+c_data)
    print(len(combined_data)) # 464
    
    
    a_paths = [train_val_patient_folders[k] for k in a_keys]
    b_paths = [train_val_patient_folders[k] for k in b_keys]
    c_paths = [train_val_patient_folders[k] for k in c_keys]
    all_patients_path = torch.utils.data.ConcatDataset(
        a_paths+b_paths+c_paths
    )
    print(all_patients_path)

    y = np.concatenate([
    np.zeros(sum(len(d) for d in a_data)),  # AB
    np.ones(sum(len(d) for d in b_data)),   # TH
    np.full(sum(len(d) for d in c_data), 2) # HN
    ])

    print("Total AB samples:", sum(len(d) for d in a_data))
    print("Total TH samples:", sum(len(d) for d in b_data))
    print("Total HN samples:", sum(len(d) for d in c_data))
    '''
    
    # AB / TH / HN
    '''
    # Filter keys by anatomical region
    train_val_dataset, _, train_val_patient_folders, _ = load_images(
        dataset_path, 25, True
    )
    a_keys = [k for k in train_val_dataset if k[1] == "HN"]

    # Extract datasets per region
    a_data = [train_val_dataset[k] for k in a_keys]
    
    combined_data = torch.utils.data.ConcatDataset(a_data)
    
    a_paths = [train_val_patient_folders[k] for k in a_keys]

    all_patients_path = torch.utils.data.ConcatDataset(
        a_paths
    )
    print(all_patients_path)
    '''
    
    mae = MeanAbsoluteError().to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)
    
    
    fold_psnr = []
    fold_mae = []
    fold_ssim = []
    fold_ms_ssim = []
    
    all_results = {}
    
    all_results_region_center = defaultdict(list)
    all_results_region_center_folds = defaultdict(list)

    splits = 5

    # Set up k-fold cross-validation
    #kfold = KFold(n_splits=splits, shuffle=True, random_state=42)

    #for fold, (_, val_index) in enumerate(
    #    kfold.split(combined_data)
    #):
    #    print(
    #        f"Fold:{fold}, val indexes:{val_index}"
    #    )
        
    #    valid_set = Subset(combined_data, val_index)
        
    # Set up Stratified k-fold cross-validation
    #splits = 5
    skf  = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    
    for fold, (_, val_index) in enumerate(skf.split(combined_data, y)):
        valid_set = Subset(combined_data, val_index)
        # Split the dataset
        val_patients_path = [all_patients_path[i] for i in val_index.tolist()]
        print(f"Validation patients: {val_patients_path}")

        transforms_val = Compose(
            [
                MaskIntensityd(keys=["mr", "ct"], mask_key="mask"),
                ToTensord(keys=["mr", "ct"]),
            ]
        )
        
        monai_valid_dataset = Dataset(
            data=valid_set, transform=transforms_val
        )  
        
        valid_loader = DataLoader(monai_valid_dataset, batch_size=batch_size_val, num_workers=8, persistent_workers=True,
            pin_memory=True)
        
        fold_model_path = os.path.join(best_models_path, f"fold_{fold}")
        #print(best_models_path)
        #print(fold_model_path)
        model_file = None
        all_results[fold] = {}
            
        if os.path.isdir(fold_model_path):
            for file in os.listdir(fold_model_path):
                #print(file)
                if file.startswith("val_bestmodel_fold") and file.endswith(ending_saved_model):
                    model_file = os.path.join(fold_model_path, file)
                    #print(model_file)
                    
        if model_file:
            if type_norm == "INSTANCE": 
                model = generator_unet(channels, strides, type_norm, num_res_units)
            elif type_norm == "BATCH":
                type_norm = Norm.BATCH
                model = generator_unet(channels, strides, type_norm, num_res_units)
            
            checkpoint = torch.load(model_file, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            print(" ")
            print(f"The best validation model for fold {fold}: {model_file}")
            
            dataset_name = model_file.split("/")[-5]
            result = model_file.split("/")[-3]
            
            all_psnr_per_batch, all_mae_per_batch, all_ssim_per_batch,all_ms_ssim_per_batch,_,_,_ = best_validation_model(model, valid_loader, fold, dataset_name, result, psnr, mae, ssim, ms_ssim, device, "generator", "validation", val_patients_path, "mha")

            
            # --- SYNTHRAD2025
            '''
            for i in range(len(val_patients_path)):
                patient_name = val_patients_path[i].split("/")[-1]
                
                if os.path.isdir(val_patients_path[i]):
                    for region in {"AB", "TH", "HN"}: # AB
                        if patient_name[1:3] == region: # 1ABA051
                            center = patient_name[3]
                            all_results_region_center[(center, region)].append({
                                'patient': patient_name,
                                'psnr':all_psnr_per_batch[i],
                                'mae':all_mae_per_batch[i],
                                'ssim':all_ssim_per_batch[i],
                                'msssim': all_ms_ssim_per_batch[i]
                            })
                            
                    
            for (center,region), values in all_results_region_center.items():
                
                region_center_psnr = [entry['psnr'] for entry in values]
                region_center_mae = [entry['mae'] for entry in values]
                region_center_ssim = [entry['ssim'] for entry in values]
                region_center_msssim = [entry['msssim'] for entry in values]
        
                # Mean and std of metrics for the (center,region) for 1 fold
                print(f"\nSummary for Center {center}, Region {region}:")
                print(f"- PSNR: {np.mean(region_center_psnr):.4f}±{np.std(region_center_psnr):.4f}")
                print(f"- MAE: {np.mean(region_center_mae):.4f}±{np.std(region_center_mae):.4f}")
                print(f"- SSIM: {np.mean(region_center_ssim):.4f}±{np.std(region_center_ssim):.4f}")
                print(f"- MR-SSIM: {np.mean(region_center_msssim):.4f}±{np.std(region_center_msssim):.4f}")
                        
                all_results_region_center_folds[(center,region)].append({
                    'psnr': np.mean(region_center_psnr),
                    'mae': np.mean(region_center_mae),
                    'ssim': np.mean(region_center_ssim),
                    'msssim': np.mean(region_center_msssim)
                })
                

            '''
            # -----------------

            # --- SYNTHRAD2023
            
            all_results[fold] = {
                        'psnr':all_psnr_per_batch,
                        'mae':all_mae_per_batch,
                        'ssim':all_ssim_per_batch,
                        'msssim':all_ms_ssim_per_batch
                    }
                    
            output_folder = f"../plots/{dataset_name}/generator/{result}"
            os.makedirs(output_folder, exist_ok=True)
            
            # Mean of metrics for each fold
            fold_psnr.append(np.mean(all_psnr_per_batch))
            fold_mae.append(np.mean(all_mae_per_batch))
            fold_ssim.append(np.mean(all_ssim_per_batch))
            fold_ms_ssim.append(np.mean(all_ms_ssim_per_batch))
                    
            print(f"For fold {fold}, the mean validation metrics were:")
            print(f"- PSNR: {fold_psnr[-1]:.4f}±{np.std(all_psnr_per_batch):.4f}")
            print(f"- MAE: {fold_mae[-1]:.4f}±{np.std(all_mae_per_batch):.4f}")
            print(f"- SSIM: {fold_ssim[-1]:.4f}±{np.std(all_ssim_per_batch):.4f}")
            print(f"- MS-SSIM: {fold_ms_ssim[-1]:.4f}±{np.std(all_ms_ssim_per_batch):.4f}")
            
    
    
    mean_psnr = np.mean(fold_psnr)
    std_psnr = np.std(fold_psnr)
    mean_mae = np.mean(fold_mae)
    std_mae = np.std(fold_mae)
    mean_ssim = np.mean(fold_ssim)
    std_ssim = np.std(fold_ssim)
    mean_ms_ssim = np.mean(fold_ms_ssim)
    std_ms_ssim = np.std(fold_ms_ssim)

    all_psnr_values, all_ssim_values, all_ms_ssim_values, all_mae_values = [], [], [], []

    for fold_data in all_results.values():
        all_psnr_values += fold_data['psnr']
        all_ssim_values += fold_data['ssim']
        all_ms_ssim_values += fold_data['msssim']
        all_mae_values += fold_data['mae']
        
    print(f"psnr: {all_psnr_values}")
    print(f"ssim: {all_ssim_values}")
    print(f"ms_ssim: {all_ms_ssim_values}")
    print(f"mae: {all_mae_values}")
            
    mean_psnr_folds = np.mean(all_psnr_values)
    std_psnr_folds = np.std(all_psnr_values)
    mean_mae_folds = np.mean(all_mae_values)
    std_mae_folds = np.std(all_mae_values)
    mean_ssim_folds = np.mean(all_ssim_values)
    std_ssim_folds = np.std(all_ssim_values)
    mean_ms_ssim_folds = np.mean(all_ms_ssim_values)
    std_ms_ssim_folds = np.std(all_ms_ssim_values)
            
    # Mean of the folds averages results for each metric
    print("\nFinal Results after Cross-Validation attending the mean of each fold:")
    print(f"Mean PSNR: {mean_psnr:.4f}±{std_psnr:.4f}")
    print(f"Mean MAE: {mean_mae:.4f}±{std_mae:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}±{std_ssim:.4f}")
    print(f"Mean MS-SSIM: {mean_ms_ssim:.4f}±{std_ms_ssim:.4f}")
    
    print("\nFinal Results after Cross-Validation not considering the mean of each fold:")
    print(f"Mean PSNR: {mean_psnr_folds:.4f}±{std_psnr_folds:.4f}")
    print(f"Mean MAE: {mean_mae_folds:.4f}±{std_mae_folds:.4f}")
    print(f"Mean SSIM: {mean_ssim_folds:.4f}±{std_ssim_folds:.4f}")
    print(f"Mean MS-SSIM: {mean_ms_ssim_folds:.4f}±{std_ms_ssim_folds:.4f}")
                
    plot_metrics(fold_ssim, os.path.join(output_folder, "all_folds_SSIM.png"), "SSIM", "Mean Validation SSIM across Folds")
    plot_metrics(fold_mae, os.path.join(output_folder, "all_folds_MAE.png"), "MAE (HU)", "Mean Validation MAE across Folds")
    plot_metrics(fold_psnr, os.path.join(output_folder, "all_folds_PSNR.png"), "PSNR (dB)", "Mean Validation PSNR across Folds")
    plot_metrics(fold_ms_ssim, os.path.join(output_folder, "all_folds_MS-SSIM.png"), "MS-SSIM", "Mean Validation MS-SSIM across Folds")
    
    
    # ----
    
    # --- SYNTHRAD2025
    '''
    # Across all folds (not the mean of each fold mean)
    
    # Initialize empty lists for each metric
    psnr_list = []
    ssim_list = []
    msssim_list = []
    mae_list = []

    # Iterate through each key-value pair in the dictionary
    for key, patient_data_list in all_results_region_center.items():
            # Iterate through each patient dictionary in the list
            for patient_data in patient_data_list:
                  # Append each metric to its respective list
                  psnr_list.append(patient_data['psnr'])
                  ssim_list.append(patient_data['ssim'])
                  msssim_list.append(patient_data['msssim'])
                  mae_list.append(patient_data['mae'])

    print(psnr_list)
    print(mae_list)
    print(ssim_list)
    print(msssim_list)

    
    print(f"\n--- FINAL RESULTS ACROSS ALL FOLDS ---")
    for (center,region), fold_values in all_results_region_center_folds.items():
        folds_region_center_psnr = [fold['psnr'] for fold in fold_values]
        folds_region_center_mae = [fold['mae'] for fold in fold_values]
        folds_region_center_ssim = [fold['ssim'] for fold in fold_values]
        folds_region_center_msssim = [fold['msssim'] for fold in fold_values]
        
        print(f"\n -> Summary for Center {center}, Region {region}:")
        print(f"Mean PSNR: {np.mean(folds_region_center_psnr):.4f}±{np.std(folds_region_center_psnr):.4f}")
        print(f"Mean MAE (HU): {np.mean(folds_region_center_mae):.4f}±{np.std(folds_region_center_mae):.4f}")
        print(f"Mean SSIM: {np.mean(folds_region_center_ssim):.4f}±{np.std(folds_region_center_ssim):.4f}")
        print(f"Mean MS-SSIM: {np.mean(folds_region_center_msssim):.4f}±{np.std(folds_region_center_msssim):.4f}")
    '''
      
    
    pass

if __name__ == "__main__":
    main()
    
# python -m  validations.best_model_generator 
# python -m  validations.best_model_generator --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/result_lr_0.0005_500_resunet_da > log_best_model_metrics_generator_lr_0.0005_400_resunet_da.txt



