# 🔍 TorchVision Object Detection & Instance Segmentation Finetuning

A complete pipeline for finetuning a **Mask R-CNN** model on a custom pedestrian detection and
segmentation dataset using PyTorch and TorchVision.

---

## 📌 Overview

This project demonstrates how to finetune a pre-trained **Mask R-CNN** (ResNet-50 + FPN backbone)
model from the TorchVision Model Zoo on the
[Penn-Fudan Pedestrian Detection and Segmentation Dataset](https://www.cis.upenn.edu/~jshi/ped_html/).
It covers the full ML pipeline: custom dataset creation, model adaptation, training, evaluation,
and visual inference — making it an ideal reference for applying transfer learning to any custom
object detection or instance segmentation task.

> ⚠️ Requires `torchvision >= 0.16`. For older versions, refer to the legacy tutorial.

---

## 📂 Dataset

**Penn-Fudan Pedestrian Database**

| Property           | Details                                      |
|--------------------|----------------------------------------------|
| Total Images       | 170 high-resolution RGB images               |
| Pedestrian Instances | 345 labeled pedestrians                    |
| Sources            | 96 from UPenn campus, 74 from Fudan University |
| Pedestrian Heights | 180–390 pixels (upright posture)             |
| Annotation Format  | Per-instance color-coded segmentation masks  |

**Download & Extract:**
```bash
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P data
cd data && unzip PennFudanPed.zip
```

**Folder Structure:**
