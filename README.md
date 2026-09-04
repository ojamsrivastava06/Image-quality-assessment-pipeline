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
```

---

## ✨ Module 1B: Borderline Image Enhancement

Module 1B is specifically designed for **BORDERLINE** fundus images identified by Module 1A. Images evaluated as `NON-CRITICAL` or `CRITICAL` bypass enhancement completely.

```text
Original Image
    ↓
Module 1A Quality Assessment
    ↓
NON-CRITICAL → OK TO GO
CRITICAL → RECAPTURE REQUIRED
BORDERLINE → Module 1B Enhancement
                    ↓
              Module 1A Reassessment
                    ↓
              Final Decision
```

### Core Architecture & Authority
- **Module 1A Authority**: Module 1A remains the sole authority for the final image-quality decision.
- **Controlled Repair**: Module 1B only attempts conservative, deterministic repair of borderline images.
- **Mandatory Reassessment**: Every enhanced image MUST be reassessed by Module 1A before any final decision is made.
- **No Bypass**: Module 1B never bypasses Module 1A quality assessment.
- **Diagnostic Decoupling**: Module 1B does not perform diabetic retinopathy diagnosis; its role is strictly pre-diagnostic image quality assurance.

### Targeted Enhancement Strategy
Module 1B uses deterministic classical image-processing methods selected according to the detected quality deficit:

- **CLAHE / contrast enhancement** → targeted to low-contrast images (applied on CIELAB $L$-channel inside retinal FOV)
- **Gamma correction** → targeted to mild underexposure / exposure deficit
- **Illumination normalization / flat-field correction** → targeted to uneven illumination and shading variation
- **Bilateral filtering** → targeted to moderate high-frequency sensor noise
- **Conservative sharpening** → targeted to mild blur
- **Limited glare/specular artifact handling** → small recoverable punctate glare regions via inpainting

**Operational Boundaries**:
- Enhancement is conservative, bounded, and strictly localized inside the detected retinal FOV mask.
- Severe blur, insufficient field of view, severe glare/clipping, and other unrecoverable capture failures are **not** artificially repaired.
- **No deep learning required**: No additional deep-learning or generative models are used; processing is entirely classical, reproducible, and non-hallucinating.

### Module 1B Safety Constraints
- **Maximum 2 operations per attempt**: Targeted strictly to detected clinical deficits.
- **Maximum 2 enhancement attempts**: Module 1B allows up to two controlled enhancement attempts for a borderline image. Each attempt is followed by Module 1A reassessment. If the image remains unacceptable after the allowed attempts, it is rejected/marked for recapture according to the final Module 1A decision.
- **Module 1A reassessment after every enhancement attempt**: No enhanced image can transition state without complete Module 1A re-evaluation.
- **Hard quality failures override weighted scoring**: Severe capture failures trigger immediate `RECAPTURE`.
- **Original image preserved**: The original image is always retained unmodified.
- **Enhanced image preserved separately**: Output artifacts maintain explicit before/after provenance.
- **Dataset/input image files are not modified**: Read-only input integrity is guaranteed and verified via cryptographic hashes.
- **Deterministic processing**: Pixel-for-pixel reproducible behavior across all executions.
- **No silent conversion**: Enhancement cannot silently convert a `CRITICAL` image into an accepted image without Module 1A validation.

---

## 🔄 Module 1A + Module 1B Final Decision Flow

```text
Input Fundus Image
        ↓
Module 1A
        ↓
Quality Assessment
        ↓
 ┌───────────────┬────────────────┬─────────────────┐
 │ NON-CRITICAL  │   BORDERLINE   │    CRITICAL     │
 │   OK TO GO    │       ↓        │    RECAPTURE    │
 │               │ Module 1B      │                 │
 │               │ Enhancement    │                 │
 │               │       ↓        │                 │
 │               │ Module 1A      │                 │
 │               │ Reassessment   │                 │
 └───────────────┴───────┬────────┴─────────────────┘
                         ↓
                  Final Decision
```

---

## 🛡️ Safety & Decision Rules

The pipeline follows a strict three-state decision contract:

| Status       | Action             |
| ------------ | ------------------ |
| NON-CRITICAL | OK TO GO           |
| BORDERLINE   | ENHANCE → REASSESS |
| CRITICAL     | RECAPTURE          |

Additional runtime checks verify:

- Normalized scores remain within [0, 1]
- Composite score remains within [0, 1]
- Quality weights sum to 1.0
- Clinical action flags remain consistent
- CRITICAL images are not enhanced
- NON-CRITICAL images bypass enhancement
- Input dataset files are not modified

---

## 📁 Project Structure

```text
.
├── src/
│   ├── config.py
│   ├── dataset_inspector.py
│   ├── fov_detector.py
│   ├── pipeline.py
│   ├── quality_classifier.py
│   ├── quality_enhancer.py
│   └── quality_metrics.py
│
├── scripts/
│   ├── smoke_test_module1.py
│   ├── test_module1b.py
│   ├── evaluate_borderline_cohort.py
│   ├── run_module1_full_production.py
│   └── validation/testing scripts
│
├── reports/
│   ├── module1b_evaluation.csv
│   ├── module1b_evaluation.md
│   ├── module1b_visual_validation/
│   └── Module 1 validation and analysis reports
│
├── .gitignore
└── README.md
```

---

## ⚙️ Requirements

- Python 3.13
- OpenCV
- NumPy
- SciPy
- scikit-image
- Pillow

Install dependencies as required by the project environment.

---

## ▶️ Running Module 1

From the project root:
```text
py -3.13 scripts\smoke_test_module1.py
```
The smoke test runs the complete Module 1 pipeline on a fundus image and reports the quality assessment and final decision.

Test a specific image:
```text
py -3.13 scripts\smoke_test_module1.py "dataset\your_image.png"
```
The script supports a specific image path and reports:
- Raw quality metrics
- Normalized quality scores
- Composite score
- Original status
- Enhancement requirement
- Enhancement operations
- Final status
- OK_TO_GO
- RECAPTURE_REQUIRED
- ENHANCEMENT_REQUIRED

---

## 📊 Full Dataset Evaluation

The complete dataset evaluation was performed separately during development and validation.

The production evaluation processed 4,178 fundus images.

Final provisional distribution:
```text
NON-CRITICAL: 3,891
BORDERLINE: 13
CRITICAL: 274
```
These results are included as development/validation artifacts and should not be interpreted as clinical performance claims.

---

## 📈 Module 1B Evaluation

Module 1B was evaluated across a representative cohort of borderline fundus images.

> **Notice**: The following metrics are development/validation results on a representative cohort and are not clinical performance claims.

- **Evaluated cohort**: 50 representative borderline fundus images
- **BORDERLINE → NON-CRITICAL**: 25 (50.0%)
- **Remaining BORDERLINE**: 8 (16.0%)
- **BORDERLINE → CRITICAL**: 17 (34.0%)
- **Average quality score**: 0.7551 → 0.7888
- **Net average score improvement**: +0.0337
- **Average improvement on converted images**: +0.0794
- **Average enhancement attempts**: 1.10 per image
- **Average processing time**: 2.015 seconds per image

### Primary-Deficit Evaluation Breakdown

| Primary Deficit | Evaluated | Converted to NON-CRITICAL | Success Rate |
|---|---:|---:|---:|
| Contrast Deficit (CLAHE) | 6 | 5 | 83.3% |
| Illumination Deficit (Flat-Fielding) | 7 | 7 | 100.0% |
| Exposure Deficit (Gamma Correction) | 21 | 15 | 71.4% |
| Sensor Noise Deficit (Bilateral) | 26 | 8 | 30.8% |
| Specular Glare (Inpainting) | 26 | 6 | 23.1% |

---

## 🧪 Verification

The Module 1 smoke test successfully verified:

- Core module imports
- End-to-end image processing
- FOV detection
- Seven-dimensional quality assessment
- Score normalization
- Composite scoring
- Enhancement routing
- Clinical decision flags
- Runtime invariants
- Dataset immutability

Example verification result:
```text
MODULE 1 SMOKE TEST: PASS
```

### Module 1B Testing & Validation

- **Automated Test Scenarios**: Eight comprehensive clinical test scenarios covering acceptable bypass, critical bypass, targeted CLAHE contrast improvement, illumination normalization, gamma exposure recovery, bilateral sensor noise suppression, multi-attempt exhaustion, and degradation rejection/escalation are implemented in `scripts/test_module1b.py`.
- **Validation Execution**:
  ```text
  py -3.13 scripts\test_module1b.py
  ```
  Expected verification result:
  ```text
  MODULE 1B VALIDATION: PASS
  ```
- **Cohort Evaluation Script**: Borderline cohort evaluation is available through `scripts/evaluate_borderline_cohort.py`.
- **Visual Validation Panels**: Annotated before/after side-by-side comparison panels demonstrating original vs. enhanced images, applied operations, score changes, and clinical decision rationales are stored in `reports/module1b_visual_validation/`.
- **Audit Documentation**: Detailed evaluation metrics, dimension deltas, and execution logs are documented in `reports/module1b_evaluation.md` and `reports/module1b_evaluation.csv`.

---

## ⚠️ Clinical Disclaimer

This repository represents a research/prototype implementation for the Smart India Hackathon problem statement.

- Module 1B is a research/prototype image-quality enhancement component.
- Enhancement results do not constitute a medical diagnosis.
- The quality thresholds, weights, and decision boundaries are provisional and require further empirical calibration and clinical validation using appropriately labelled fundus-image quality/gradability data.
- The Module 1 output is not a medical diagnosis and should not be used as a standalone clinical decision system.

---

## 🚀 Future Integration

Module 1 is designed as the image-quality gate before downstream retinal analysis and diabetic retinopathy screening.

The final system can integrate:
```text
Fundus Image
      ↓
Module 1
Image Quality Assessment
      ↓
Quality Gate
      ↓
Retinal Analysis / DR Screening
      ↓
Explainable Result
```

---

## 🏆 Smart India Hackathon 2026

Problem Statement: SIH26038
Domain: HealthTech
Focus: Explainable AI for Diabetic Retinopathy Screening in Rural India

Developed as part of the Smart India Hackathon 2026.
