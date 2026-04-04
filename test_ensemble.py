import click
import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas, researchpy
import json

# Loading data and initial transformations
from utils.load_data import load_images, load_hn_centerD_images

# Generator architecture model
from models.generator import GeneratorUNet

from validations.validation import best_validation_model,mean_image_across_folds

# PyTorch
import torch

# Pytorch Lightning
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure

# MONAI
from monai.utils import set_determinism
from monai.data import DataLoader, Dataset
from monai.transforms import Compose,MaskIntensityd,ToTensord
from monai.networks.layers.factories import Norm   
from collections import defaultdict

def parse_list(ctx, param, value):
    return list(map(int, value.split(",")))

def patient_in_list(regions_patient_paths):
    patients=[]
    for region in regions_patient_paths:
        for center_paths in region:
            for patient_path in center_paths:
                patient_id = os.path.basename(patient_path)
                patients.append(patient_id)
    return patients
        
@click.command()
@click.option('--dataset_path', type=str, default=os.getenv("DATASET_PATH"), help='Path that contains the dataset')
@click.option('--best_models_path', type=str, default="/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS", help='Path where best models are') 
@click.option('--batch_size_test', type=int, default=1, help='Size of the batches for validation')
@click.option('--channels', type=str, default="64, 128, 256, 512, 512, 512, 512, 512", callback=parse_list, help='channels of the generator')
@click.option('--strides', type=str, default="2, 2, 2, 2, 2, 2, 2", callback=parse_list, help='strides of the generator')
@click.option('--num_res_units', type=int, default=0, help='num of residual units per layer')
@click.option('--type_norm', type=str, default="INSTANCE", help='Choose between "INSTANCE" or "BATCH"')
@click.option('--ending_saved_model', type=str, default="0-999.pth", help='Choose until which best model epoch you want to validate')
@click.option("--file_format",type=str,default="nii.gz",help="Choose which file format the data is using: 'mha' (SynthRAD2025) or 'nii.gz' (SynthRAD2023)")
@click.option("--tested_in",type=str,default="AB",help="Choose the metrics json file name according to where you are testing the model")
@click.option("--model_state",type=str,default='G_state_dict',help="Choose the state to load: for AE (='model_state_dict') and cGAN (= 'G_state_dict')")

def main(best_models_path, batch_size_test, dataset_path, channels, strides, type_norm, num_res_units, ending_saved_model, file_format, tested_in, model_state):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    set_determinism(seed=42)
    
    
    # SynthRAD2023 
    '''
    _, tests_datasets, _, test_patient_folders = load_images(dataset_path, 23, is_test_set=True)
    
    combined_data = torch.utils.data.ConcatDataset(
        list(tests_datasets.values())
    )
    
    all_patients_path = torch.utils.data.ConcatDataset(list(test_patient_folders.values())) 
    '''
    
    # SynthRAD2023 or 2025 based on region
    ds_version = 23 if tested_in == "B" else 25
    _, tests_datasets, _, test_patient_folders = load_images(dataset_path, ds_version, True)

    # Choose test dataset based on argument
    if tested_in in ["AB", "TH", "HN", "B"]:
        a_keys = [k for k in tests_datasets if k[1] == tested_in]
        a_data = [tests_datasets[k] for k in a_keys]
        combined_data = torch.utils.data.ConcatDataset(a_data)
        
        a_paths = [test_patient_folders[k] for k in a_keys]
        all_patients_path = torch.utils.data.ConcatDataset(a_paths)
    elif tested_in == "CenterD":
        combined_data, test_patient_folders = load_hn_centerD_images(dataset_path) 
        all_patients_path = test_patient_folders
    else:
        # Default all datasets
        combined_data = torch.utils.data.ConcatDataset(list(tests_datasets.values()))  
        all_patients_path = torch.utils.data.ConcatDataset(list(test_patient_folders.values()))
    
    mae = MeanAbsoluteError(num_outputs=1).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)

    dataset_name = None
    result = None
    all_results = {}
    all_preds_fold = {}
    all_reals_fold = {}
    all_masks_fold = {}
    
    all_results_region_center = defaultdict(list)
    all_results_region_center_folds = defaultdict(list)
  
    for fold in range(5):
    
        transforms_test = Compose(
            [
                MaskIntensityd(keys=["mr", "ct"], mask_key="mask"),
                ToTensord(keys=["mr", "ct"]),
            ]
        )
        
        monai_test_dataset = Dataset(
            data=combined_data, transform=transforms_test
        )
        
        test_loader = DataLoader(
            monai_test_dataset,
            batch_size=batch_size_test,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )
        
        model_file = None
        all_results[fold] = {}
        all_preds_fold[fold] = {}
        all_reals_fold[fold] = {}
        all_masks_fold[fold] = {}
        
        fold_model_path = os.path.join(best_models_path, f"fold_{fold}") 
            
        if os.path.isdir(fold_model_path):
            for file in os.listdir(fold_model_path):
                if (file.startswith("val_bestmodel_fold") or file.startswith("val_checkpoints_fold")) and file.endswith(ending_saved_model): 
                    model_file = os.path.join(fold_model_path, file)
                  
                        
        if model_file:
            # Use the correctly imported GeneratorUNet
            generator = GeneratorUNet(in_channels=1, out_channels=1)
            
            checkpoint = torch.load(model_file, weights_only=False)
            generator.load_state_dict(checkpoint[model_state]) # different for AE (='model_state_dict') and cGAN (= 'G_state_dict')
            generator = generator.to(device)
                
            print(" ")
            print(
                f"The best validation model for fold {fold}: {model_file}"
            )
            
            dataset_name = model_file.split("/")[-5]
            result = model_file.split("/")[-3]
            
            all_psnr_per_batch, all_mae_per_batch, all_ssim_per_batch, all_ms_ssim_per_batch, all_preds, all_reals, all_masks = best_validation_model(generator, test_loader, fold, dataset_name, result, psnr, mae, ssim, ms_ssim, device, "AE", "test", all_patients_path, file_format)
            
            
            # --- SYNTHRAD2025
            
            for i in range(len(all_patients_path)):
                patient_name = all_patients_path[i].split("/")[-1]
                
                if os.path.isdir(all_patients_path[i]):
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
                print(f"- MS-SSIM: {np.mean(region_center_msssim):.4f}±{np.std(region_center_msssim):.4f}")
                        
                all_results_region_center_folds[(center,region)].append({
                    'psnr': np.mean(region_center_psnr),
                    'mae': np.mean(region_center_mae),
                    'ssim': np.mean(region_center_ssim),
                    'msssim': np.mean(region_center_msssim)
                })
                
                # ------
            
            
            # Metrics per fold: CT GT <-> sCT
            all_results[fold] = {
                        'psnr':all_psnr_per_batch,
                        'mae':all_mae_per_batch,
                        'ssim':all_ssim_per_batch,
                        'msssim': all_ms_ssim_per_batch
                    }
            
            # All sCT predicted and real per image (per fold)
            all_preds_fold[fold] = all_preds   
            all_reals_fold[fold] = all_reals
            all_masks_fold[fold] = all_masks
            
            
    # --- SYNTHRAD2025
    
    # Across all folds (not the mean of each fold mean)
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
    
    # -----

    #print(all_preds_fold)
    #print(all_reals_fold)
    #print(all_masks_fold)
    
    # All x mean sCT
    if dataset_name is None:
        dataset_name = "Unknown"
    if result is None:
        result = "Unknown"
        
    all_psnr_ensemble, all_mae_ensemble, all_ssim_ensemble, all_ms_ssim_ensemble = mean_image_across_folds(all_preds_fold, all_reals_fold, all_masks_fold,  dataset_name, result, psnr, mae, ssim, ms_ssim, all_patients_path, file_format)
    print("\nMetrics from ensemble of models: ")
    print(all_psnr_ensemble)
    print(all_mae_ensemble)
    print(all_ssim_ensemble)
    print(all_ms_ssim_ensemble)

    ensemble_metrics_ttest = {
        'psnr': all_psnr_ensemble,
        'mae': all_mae_ensemble,
        'ssim':  all_ssim_ensemble,
        'msssim': all_ms_ssim_ensemble
    }
    
    # Save in JSON file
    os.makedirs('results', exist_ok=True)
    with open(f'results/ensemble_{result}_tested_{tested_in}.json', 'w') as f:
        json.dump(ensemble_metrics_ttest, f)
    
    # Across all mean images
    ensemble_metrics = defaultdict(list)
    
    for i in range(len(all_psnr_ensemble)):
                if i >= len(all_patients_path):
                    break
                patient_name = all_patients_path[i].split("/")[-1]
                
                if os.path.isdir(all_patients_path[i]):
                    for region in {"AB", "TH", "HN"}: # AB
                        if patient_name[1:3] == region: # 1ABA051
                            center = patient_name[3]
                            ensemble_metrics[(center, region)].append({
                                'patient': patient_name,
                                'psnr':all_psnr_ensemble[i],
                                'mae':all_mae_ensemble[i],
                                'ssim':all_ssim_ensemble[i],
                                'msssim': all_ms_ssim_ensemble[i]
                            })
   
    print(f"\n--- FINAL RESULTS FOR ENSEMBLE OF FOLDS IMAGES ---")
        
    for (center,region), values in ensemble_metrics.items():
                region_center_psnr = [entry['psnr'] for entry in values]
                region_center_mae = [entry['mae'] for entry in values]
                region_center_ssim = [entry['ssim'] for entry in values]
                region_center_msssim = [entry['msssim'] for entry in values]
        
                # Mean and std of metrics for the (center,region) 
                print(f"\nSummary for Center {center}, Region {region}:")
                print(f"- PSNR: {np.mean(region_center_psnr):.4f}±{np.std(region_center_psnr):.4f}")
                print(f"- MAE: {np.mean(region_center_mae):.4f}±{np.std(region_center_mae):.4f}")
                print(f"- SSIM: {np.mean(region_center_ssim):.4f}±{np.std(region_center_ssim):.4f}")
                print(f"- MR-SSIM: {np.mean(region_center_msssim):.4f}±{np.std(region_center_msssim):.4f}")
    

    metrics = ['psnr','mae','ssim','msssim']
    for metric in metrics:
        print(f"\n--- Results for {metric.upper()} ---")
        for fold, fold_data in all_results.items():
            if not fold_data:
                continue
            
            fold_values = np.array(fold_data[metric])
            ensemble_values = np.array(ensemble_metrics_ttest[metric])
        
            import scipy.stats
            # Use scipy instead of researchpy due to pandas version incompatibility with researchpy formatting
            ttest_res = scipy.stats.ttest_rel(fold_values, ensemble_values)
            
            print(f"\nFold {fold} vs Ensemble of Models")
            print(f"Paired T-test statistic: {ttest_res.statistic:.4f}, p-value: {ttest_res.pvalue:.4e}") 

                                

if __name__ == "__main__":
    main()
    
#python -m tests.test_ensemble
