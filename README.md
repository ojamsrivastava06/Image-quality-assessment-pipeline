# SIH26038 – Explainable AI for Diabetic Retinopathy Screening

## Module 1: Image Quality Assessment & Enhancement

This repository contains the implementation of **Module 1** for the Smart India Hackathon 2026 problem statement **SIH26038 – Explainable AI for Diabetic Retinopathy Screening in Rural India**.

The purpose of Module 1 is to determine whether an incoming retinal fundus photograph is suitable for downstream retinal analysis and diabetic retinopathy screening.

---

## 🎯 Objective

Before applying retinal analysis or AI-based screening, the input fundus image must first be checked for image quality.

Module 1 performs deterministic image quality assessment and classifies each image into three clinical-quality states:

- **NON-CRITICAL** → Image is acceptable → **OK TO GO**
- **BORDERLINE** → Image may benefit from enhancement → **ENHANCE & REASSESS**
- **CRITICAL** → Image is unsuitable → **RECAPTURE REQUIRED**

> `OK TO GO` is a clinical action directive, not a separate quality class.

---

## 🔍 Quality Assessment

The pipeline evaluates seven image-quality dimensions:

1. **Focus / Sharpness**
2. **Brightness / Exposure**
3. **Contrast**
4. **Noise**
5. **Field of View (FOV)**
6. **Illumination Uniformity**
7. **Artifacts / Glare**

The individual measurements are normalized to `[0, 1]` and combined using a weighted quality score.

---

## 🧠 Decision Pipeline

```text
Input Fundus Image
        │
        ▼
Basic Image Validation
        │
        ▼
Retinal FOV Detection
        │
        ▼
7-Dimension Quality Assessment
        │
        ├── Focus
        ├── Brightness
        ├── Contrast
        ├── Noise
        ├── FOV
        ├── Illumination
        └── Artifacts
        │
        ▼
Score Normalization
        │
        ▼
Hard-Failure Checks
        │
        ▼
Weighted Quality Score
        │
        ├── NON-CRITICAL
        │       └── OK TO GO
        │
        ├── BORDERLINE
        │       └── Enhancement
        │              └── Reassessment
        │
        └── CRITICAL
                └── RECAPTURE
