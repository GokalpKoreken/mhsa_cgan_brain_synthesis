import os
from collections import defaultdict

# PyTorch
import torch

# MONAI
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ScaleIntensityRanged,
)
from monai.data import ITKReader
from utils.images_utils import RemoveEmptySlices, KeepSlicesWithAtLeast, ScaleIntensityWithoutEmptySlicesd, CropSlicesd


def load_images(directory_img: str, dataset23or25: int, is_test_set: bool=False):

    datasets = defaultdict(list)
    patient_folder_names = defaultdict(list)
    
    if dataset23or25==25:
        if "Task1" in os.listdir(directory_img) and os.path.isdir(os.path.join(directory_img, "Task1")):
            directory_img = os.path.join(directory_img, "Task1")
        patient_paths = []
        for folder_region in os.listdir(directory_img):
            region_path = os.path.join(directory_img, folder_region)
            if not os.path.isdir(region_path):
                continue
            #print(region_path)
            for folder_patient in os.listdir(region_path):
                #print(folder_patient)
                patient_path = os.path.join(region_path, folder_patient)
                #print(patient_path)
                patient_paths.append(patient_path)
    else:
        patient_paths = []
        for folder_patient in os.listdir(directory_img):
            #print(folder_patient)
            patient_path = os.path.join(directory_img, folder_patient)
            #print(patient_path)
            patient_paths.append(patient_path)
            
    for patient_path in patient_paths:
            folder_patient = os.path.basename(patient_path)
            
            if not os.path.isdir(patient_path) or folder_patient.lower() in [
                "overview",
                "overviews",
            ]:  # overview - SynthRAD2023; overviews - SynthRAD2025
                continue

            ext = ".nii.gz" if dataset23or25 == 23 else ".mha"
            mri_path = os.path.join(patient_path, f"mr{ext}")
            ct_path = os.path.join(patient_path, f"ct{ext}")
            mask_path = os.path.join(patient_path, f"mask{ext}")

            if (
                os.path.exists(mri_path)
                and os.path.exists(ct_path)
                and os.path.exists(mask_path)
            ):
                dataset = {"mr": mri_path, "ct": ct_path, "mask": mask_path}

                # SynthRAD2023
                if (
                    folder_patient[1] in {"P", "B"}
                    and folder_patient[2] in {"A", "B", "C"}
                    and dataset23or25 == 23
                ):
                    region = folder_patient[1]
                    # print(region)
                    center = folder_patient[2]
                    datasets[(center, region)].append(dataset)
                    patient_folder_names[(center, region)].append(patient_path)

                # SynthRAD2025
                if (
                    folder_patient[1:3] in {"AB", "TH", "HN"}
                    and folder_patient[3] in {"A", "B", "C"}
                    and dataset23or25 == 25
                ):
                    region = folder_patient[1:3]
                    center = folder_patient[3]
                    datasets[(center, region)].append(dataset)
                    patient_folder_names[(center, region)].append(patient_path)
                    # print(datasets)
                    
    if is_test_set:
        transforms = Compose(
            [
                LoadImaged(keys=["mr", "ct", "mask"], reader=ITKReader),
                EnsureChannelFirstd(keys=["mr", "ct", "mask"]),
                CropSlicesd(keys=["mr", "ct", "mask"], start_slice=80, end_slice=165),
                # remove empty slices (MRI copy), including the remaining slices with at least 10% of non-zeros pixels (MRI copy),
                # to obtain min and max intensities of MRI copy and with that normalise the real MRIs
                ScaleIntensityWithoutEmptySlicesd(keys=["mr"], reference_key="mr"),
                # normalize CT intensities (-1024 HU = air, 3000 HU = bone)
                ScaleIntensityRanged(
                    keys=["ct"],
                    a_min=-1024,
                    a_max=3000,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )
        
    else:
        transforms = Compose(
        [
            LoadImaged(keys=["mr", "ct", "mask"], reader=ITKReader),
            EnsureChannelFirstd(keys=["mr", "ct", "mask"]),
            CropSlicesd(keys=["mr", "ct", "mask"], start_slice=80, end_slice=165),
            RemoveEmptySlices(keys=["mr", "ct", "mask"], reference_key="mask"),
            KeepSlicesWithAtLeast(
                keys=["mr", "ct", "mask"], reference_key="mask", pct=0.1  
                # slices with at least 10% of non-zeros pixels are included
            ),
            # normalize MRI intensities
            ScaleIntensityd(
                keys=["mr"],
                minv=-1.0,
                maxv=1.0,
            ),
            # normalize CT intensities (-1024 HU = air, 3000 HU = bone)
            ScaleIntensityRanged(
                keys=["ct"],
                a_min=-1024,
                a_max=3000,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )


    from monai.data import Dataset
    monai_datasets = {
        key: Dataset(data=value, transform=transforms)
        for key, value in datasets.items()
    }

    train_val_datasets = {}
    tests_datasets = {}
    train_val_patient_folders = {}
    test_patient_folders = {}

    for key, dataset in monai_datasets.items():  # key is (center,region)
        folders = patient_folder_names[key]

        train_val, test = torch.utils.data.random_split(dataset, [0.90, 0.10])
        # print(type(train_val)) # Subset

        train_val_idx = train_val.indices
        test_idx = test.indices

        train_val_patient_folders[key] = [folders[i] for i in train_val_idx]
        test_patient_folders[key] = [folders[i] for i in test_idx]

        train_val_datasets[key] = train_val
        tests_datasets[key] = test

    return (
        train_val_datasets,
        tests_datasets,
        train_val_patient_folders,
        test_patient_folders,
    )


# Test set: center D (HN images) from SynthRAD2025
def load_hn_centerD_images(directory_img: str):
    
    datasets = []
    patient_folders = []
    
    for folder_region in os.listdir(directory_img):
        region_path = os.path.join(directory_img, folder_region)
        
        for folder_patient in os.listdir(region_path):
            patient_path = os.path.join(region_path, folder_patient)
            
            if not os.path.isdir(patient_path) or folder_patient.lower() == "overviews":
                continue

            if folder_patient[1:3] == "HN" and folder_patient[3] == "D":
                mri_path = os.path.join(patient_path, "mr.mha")
                ct_path = os.path.join(patient_path, "ct.mha")
                mask_path = os.path.join(patient_path, "mask.mha")
                
                if all(os.path.exists(p) for p in [mri_path, ct_path, mask_path]):
                    datasets.append({"mr": mri_path, "ct": ct_path, "mask": mask_path})
                    patient_folders.append(patient_path)

    transforms = Compose([
        LoadImaged(keys=["mr", "ct", "mask"], reader=ITKReader),
        EnsureChannelFirstd(keys=["mr", "ct", "mask"]),
        ScaleIntensityWithoutEmptySlicesd(keys=["mr"], reference_key="mr"),
        ScaleIntensityRanged(
            keys=["ct"],
            a_min=-1024, a_max=3000,
            b_min=-1.0, b_max=1.0,
            clip=True
        ),
    ])

    from monai.data import Dataset
    # Cria dataset
    dataset = Dataset(data=datasets, transform=transforms)
    
    return dataset, patient_folders

