# Efficient Semi-supervised Tooth Instance Segmentation

This repository contains the implementation of our semi-supervised deep learning framework for tooth instance segmentation in 2D panoramic X-ray images, as presented in our MICCAI 2024 challenge paper:

**Efficient Semi-supervised Tooth Instance Segmentation in Panoramic X-Rays Using ResUnet50 and SAM Networks**  
📄 Published in *MICCAI Challenges 2024 – Lecture Notes in Computer Science*  
📚 DOI: [10.1007/978-3-031-88977-6_13](https://doi.org/10.1007/978-3-031-88977-6_13)

## Highlights

- 🔍 Combines **ResUnet50** with **SAM-Med2D** to improve instance segmentation of teeth on panoramic X-rays.
- ⚙️ Uses **semi-supervised learning** via pseudo-labeling to leverage unlabeled data.
- 🦷 Achieved **79.15% Dice (image-level)** and **45.58% Dice (instance-level)** on the STS 2024 Challenge validation set.
- 🧠 Trained on 2,380 panoramic X-rays and optimized for **clinical efficiency** (average inference time ~10s/image).
- 🧪 Includes ablation studies, reproducible training protocols, and detailed evaluation metrics.

## Code Access

We are currently preparing the final version of the code and will release it soon at:  
👉 [https://github.com/Guo777777](https://github.com/Guo777777)

The repository will include:

- 📁 Preprocessing scripts  
- 🧠 Model architecture (ResUnet50 + SAM-Med2D)  
- 🏷️ Pseudo-labeling strategy  
- 🧪 Evaluation scripts for STS metrics  
- 📊 Trained models and results  
