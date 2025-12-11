
<div align="center">

# Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge



[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2511.20726)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://yourname.github.io/project-page/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Authors -->
<br>

**Yuhang Wang**<sup>1</sup>, 
**Heye Huang**<sup>2,3*</sup>, 
**Zhenhua Xu**<sup>4</sup>, <br>
**Kailai Sun**<sup>2,3</sup>, 
**Baoshen Guo**<sup>2,3</sup>, 
**Jinhua Zhao**<sup>2,3</sup>

<br>

<!-- Affiliations -->
*<sup>1</sup>University of Chinese Academy of Sciences, China* <br>
*<sup>2</sup>Singapore-MIT Alliance for Research and Technology Centre (SMART), Singapore* <br>
*<sup>3</sup>Massachusetts Institute of Technology (MIT), USA* <br>
*<sup>4</sup>Tsinghua University, China* <br>

<br>

<img src="Fig_1.png" width="100%" alt="LFR Framework"/>

</div>

## 📖 Abstract

This repository contains the official implementation of "Learning from Risk". We propose a novel framework that utilizes **Large Language Models (LLMs)** to guide the generation of safety-critical driving scenarios. By incorporating prior knowledge, our method generates more realistic and challenging scenarios compared to rule-based approaches.

---

## 🏗️ Qualitative Results

<div align="center">

Visualizing the generation process across 4 diverse scenarios. 
<br>
**Left (Initial):** The benign traffic initialization.
**Right (Generated):** The safety-critical scenario guided by our LLM framework.
<br>
<i>Notice how the agents evolve to create more interactive and risky situations while maintaining realism.</i>

<table>
  <tr>
    <th width="50%" style="text-align:center"><strong>Initial Scene (Before)</strong></th>
    <th width="50%" style="text-align:center"><strong>Generated Risk Scenario (After)</strong></th>
  </tr>
  
  <!-- Group 1 -->
  <tr>
    <td><video src="https://github.com/user-attachments/assets/812f14f5-d84d-4704-aeaa-62877d35e693"></td>
    <td><video src="https://github.com/user-attachments/assets/239e4c14-bd6e-4867-8402-c6e6f204bd76"></td>
  </tr>
  
  <!-- Group 2 -->
  <tr>
    <td><video src="https://github.com/user-attachments/assets/b6538b6e-c4ea-4da0-801f-892ea5936db5"></td>
    <td><video src="https://github.com/user-attachments/assets/9f40b4d3-6b4f-455c-9012-88a334badb4c"></td>
  </tr>

  <!-- Group 3 -->
  <tr>
    <td><video src="https://github.com/user-attachments/assets/83d3cfe1-700d-4308-a607-5fed80eccea2"></td>
    <td><video src="https://github.com/user-attachments/assets/e720a592-7ebe-4e4c-b4ab-8d7ebb14e1a3"></td>
  </tr>

  <!-- Group 4 -->
  <tr>
    <td><video src="https://github.com/user-attachments/assets/b5efb473-2461-46c5-958f-e245112f60ea"></td>
    <td><video src="https://github.com/user-attachments/assets/de0ebb30-bef6-45eb-889c-9d77691ae0b0"></td>
  </tr>

</table>

</div>
---

## 🛠️ Installation

1. Create conda environment:
    ```bash
    # Fixed typo: assuming env name is related to paper acronym LFR
    conda create -n lrf python=3.8
    conda activate lrf
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Dataset Preparation

Please download the required datasets and organize them in the `data/` directory.

### 1. nuScenes Dataset
For training/testing the traffic model and scenario generation, the **nuScenes dataset** is required.
> **Note:** You only need to download the **Metadata** (e.g., v1.0-trainval_meta) and **Map expansion** (nuScenes-map-expansion-v1.3). The full dataset (images/lidar) is **NOT** required.

*   Download from: [nuScenes Official Website](https://www.nuscenes.org/download)

### 2. HighD Dataset
Please download the **HighD dataset** for highway scenario experiments.

*   Download from: [HighD Official Website](https://levelxdata.com/highd-dataset/)

### 📂 Directory Structure
After downloading, please organize your files as follows:

```text  
LFR/  
├── data/  
│   ├── nuscenes/  
│   │   ├── maps/                 # Map expansion files  
│   │   ├── v1.0-trainval/        # Metadata files  
│   │   └── ...  
│   ├── highd/  
│   │   ├── 01_tracks.csv  
│   │   ├── 01_tracksMeta.csv  
│   │   └── ...  
│   └── ...  
├── src/  
└── ...  

## 🚀 Usage

### 1. Initial Scene Generation

Generate traffic flow based on NuScenes dataset:

```bash
python src/train_traffic.py --config ./configs/train_traffic.cfg
```

### 2. Risk-Aware Scenario Generation

Generate adversarial scenarios with different risk levels using LLM guidance. 
> **Note:** You can replace `deepseek-reason` with other supported LLM models.

```bash
python src/adv_scenario_gen.py \
     --config configs/adv_gen_rule_based.cfg \
     --ckpt model_ckpt/traffic_model.pth \
     --use_llm \
     --llm_model deepseek-reason
```

---

## 📝 TODO List

- [ ] Releasing the code for generating highD scenarios.


## 📄 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{wang2025learning,
  title={Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge},
  author={Wang, Yuhang and Huang, Heye and Xu, Zhenhua and Sun, Kailai and Guo, Baoshen and Zhao, Jinhua},
  year={2025}
}
```

## 📝 License

This project is licensed under the [MIT License](LICENSE).

**Disclaimer:** This project is for academic research purposes only. Please conduct sufficient safety testing before deploying in real-world autonomous driving systems.

<div align="right">
Last update: 2025-11-10
</div>
