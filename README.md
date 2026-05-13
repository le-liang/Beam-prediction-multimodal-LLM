# Beam-prediction-multimodal-LLM

This repository contains the data preprocessing pipeline for the Multimodal-Wireless dataset, targeting beam prediction tasks. The model implementation code will be open-sourced in the coming months.

## 0. Codes and Checkpoints

Users should download the .zip file in the **Latest Release**.

## 1. Environment Setup

First, please create and activate a new virtual environment with Python 3.8:

```bash
conda create -n preprocessor python=3.8
conda activate preprocessor
```

Next, install the required base dependencies using the provided requirements.txt:

```bash
pip install -r requirements.txt
```
**OpenCOOD Configuration:** This project relies on opencood for multimodal data
processing. Please note that you need to use the customized opencood source code
from the where2comm repository and have it available locally.

You can clone the where2comm repository and install it in development mode
within your current environment:

```bash
git clone https://github.com/YifanLu/where2comm.git
cd where2comm
python setup.py develop
```

(Alternatively, you can directly copy the opencood folder from the where2comm
repository into the root directory of this project, ensuring that Python can
import it correctly.)

## 2. Data Organization

Please organize and store the raw multimodal dataset strictly according to the
following directory tree structure:
```bash
|-- Town03
|   |
|   `-- Town03_CBDcrossroad_seed42
|       |
|       |-- cav_1
|       |   |-- 000001.yaml
|       |   |-- 000001.pcd
|       |   |-- 000001_paths.npy  
|       |   |-- 000001_camera0.png
|       |   |-- 000001_depth_camera0.png
|       |   `-- 000001.json
|       |
|       |-- cav_2
|       |   `-- ...
|       |
|       `-- cav_3
|           |-- 000001.yaml
|           |-- 000001.pcd
|           `-- ...
|
|-- Town05
|   `-- Town05_...  (Same structure)
|
`-- Town10
    `-- Town10_...  (Same structure)
```
⚠️ **Important Data Note:** In our downlink beamforming scenario, please pay special
attention: The multimodal data (such as images, point clouds, etc.) stored
within the cav_{i} folders are actually the global observation data collected by
the RSU (Road Side Unit) in that specific scenario. However, the channel data
stored in .npz (or .npy) format are distinct for each different cav_{i}. During
preprocessing, you must correctly combine the multimodal observation features
from the same RSU with the respective distinct channel data for each CAV.

## 3. Running the Code

Once the environment setup and data organization are complete, you can place the codes under the data paths, and run the
preprocessing script directly:
```bash
python Loader.py
```

