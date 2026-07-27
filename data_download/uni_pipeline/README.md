# UNI Pipeline for TCGA-LUAD 9-Gene Prediction

Complete download extract patches extract UNI features pipeline.
Designed for TCGA-LUAD WSI from IDC. Works with DeepGEM MIL training.

## Features
- Download idc-index s5cmd from IDC public S3
- Patch extraction openslide gdcm backend streamed DICOM WSI reading
  Memory under 200MB even for 8B-pixel 22GB slices solves pydicom OOM
- Feature extraction UNI ViT-L 16 1024-dim auto Resize 224x224
- Auto space management delete dicom after patch extraction ~360MB net freed per case
- Skip logic cases with patches at least 900 auto-skipped supports resume

## Files
download_one_case.py                Download largest SM series for one case
extract_one_case_openslide.py        Streamed 1120x1120 patch extraction
pipeline_download_patch.py           Main download patches delete dicom
pipeline_uni_features.py             UNI feature extraction pipeline
redownload_specific_cases.py         Re-download and re-extract specific cases
cleanup_small_patches.py             Delete patches smaller than 50KB
cleanup_dicom.py                     Clean up dicom after pipeline
analyze_resize_info.py               Quantify 1120x1120 to 224x224 resize info loss

## Data Flow
IDC tcga_luad
  PatientID equals TCGA-XX-YYYY
    Multiple SM series each equals 1 slice
      Each SM has multiple dcm pyramid
        Download select largest SM series by series_size_MB
        Store at dicom_root case study SM_uid dcm_files
        Extract patches openslide streamed read_region 1120x1120 stride
        Store at patches_root case slice_uid x_y.jpg
        Filter skip white over 95 percent or tissue under 15 percent
        Limit max 1000 patches per case
        Extract features patch Resize 224 UNI ViT-L 1024-dim
        Store at features_root case.npz
        Content features N 1024 slice_ids slice_uids patch_paths
        Auto-clean delete dicom free space

## Usage

### 1 Setup Environment
conda create -n lung9gene python=3.11
conda activate lung9gene
pip install idc-index openslide-python openslide-bin python-gdcm pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg torch torchvision timm pillow numpy pandas

### 2 Download UNI ckpt
Download UNI ViT-L 16 from https://huggingface.co/MahmoodLab/UNI
Put at UNI assets ckpts vit_large_patch16_224.dinov2.uni_mass100k pytorch_model.bin

### 3 Main pipeline
python pipeline_download_patch.py
about 10 to 12 hours for 412 cases

### 4 Extract UNI features
python pipeline_uni_features.py
about 20 to 30 hours for 414 cases CPU mode

## Output Format
features_root TCGA-XX-YYYY.npz
  features N_patches 1024 float32
  slice_ids N_patches int32
  slice_uids n_slices str
  patch_paths N_patches str

Direct input to DeepGEM MIL training.

## Key Design Decisions
Why 1120x1120 patch UNI original paper uses 1120x1120 Large FOV 560um at 20x preserves tissue-level structure UNI internally Resize to 224x224

Why only 1 SM series 1 dcm per case D drive space limited 350GB User decision rolling strategy Auto space recovery

## Known Limitations
1. Only 1 slice per case (TCGA-LUAD has 2-4 slices, only largest downloaded)
2. Only 1 dcm per series (pyramid)
3. CPU extraction slow (~3-5 min/case)

## Performance Stats
Total cases: 421 (MC3 IDC)
Cases with patches: 422
Avg patches per case: 631 (max 1000)
Total patches: ~266,000

## Troubleshooting
openslide can not open .dcm: pip install openslide-bin (provides libopenslide-1.dll), pip install python-gdcm

pydicom decode OOM: Use extract_one_case_openslide.py, NOT pydicom pixel_array, openslide streamed, memory under 200MB

## References
UNI: Chen et al., Nature Medicine 2024
IDC: NCI public DICOM data
idc-index: IDC Python client
openslide: streamed WSI reader