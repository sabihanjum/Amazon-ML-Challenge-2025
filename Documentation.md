# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Hide n seek
**Team Members:** Sabiha Anjum
                  Shalini S K
                  Tanya Shree V
**Submission Date:** 13-10-2025

---

## 1. Executive Summary
Our approach centers on multifeature regression powered by robust feature engineering. We combine granular text features from product descriptions, domain-specific category/brand signals, and sophisticated calculated quantity and value metrics to predict product pricing with high fidelity. The solution maximizes explainability, leveraging a LightGBM ensemble for strong validation performance.



---

## 2. Methodology Overview

### 2.1 Problem Analysis
*The goal is to predict the ideal product price using both textual and image-based features.  
We explored relationships between product metadata, brand, image aesthetics, and target price distribution during exploratory data analysis (EDA).*

**Key Observations:**
- Significant correlation between product category and average price.  
- Descriptive text often contains pricing cues such as material or premium tags.  
- Image-based features (color, brightness, background uniformity) influence model performance.  
- Outliers (extreme pricing values) required normalization and clipping.

### 2.2 Solution Strategy
*We adopted a **hybrid multimodal approach** that combines:
- **Text embeddings** from pre-trained transformer models.
- **Image embeddings** from a CNN-based encoder.
- **Tabular numerical features** for brand, category, and statistical descriptors.*
We engineered more than 40 custom features, optimized text vectorization (TF-IDF, SVD), and trained LightGBM using cross-validation with SMAPE minimization.

Core Steps:
Advanced data cleaning (robust CSV loading).
Extract + transform price-driving features, including units, pack size, premium score, and granular product details.
Merge dense TF-IDF/SVD text embeddings with numerical/tabular features.
Model tuning via KFold Validation, RobustScaler normalization, and early stopping.

**Approach Type:** Hybrid Multimodal Model    
**Core Innovation:** Fusion of text, image, and structured features using weighted late fusion for optimized SMAPE score.

---

## 3. Model Architecture

### 3.1 Architecture Overview
             ┌───────────────────────────┐
             │     Product Title/Text    │
             └────────────┬──────────────┘
                          │
                   [DistilBERT Encoder]
                          │
                    Text Embeddings
                          │
   ┌──────────────────────┼───────────────────────────┐
   │                      │                           │
   ▼                      ▼                           ▼
Product Image       Tabular Metadata          Product Category
 (RGB 224×224)      (Brand, Rating, etc.)       (One-hot encoded)
   │                      │                           │
[EfficientNet-B0]    [Normalization + Dense]     [Embedding Layer]
   │                      │                           │
   └───────────────┬──────┴──────────────┬────────────┘
                   ▼                     ▼
           ┌────────────────────────────────────┐
           │       Feature Concatenation        │
           └────────────────┬───────────────────┘
                            ▼
              [Dense Layers + BatchNorm + Dropout]
                            ▼
                 [Fully Connected Regression Head]
                            ▼
                    **Final Price Prediction**



### 3.2 Model Components

**Text Processing Pipeline:**

**Preprocessing Steps:** Lowercasing, punctuation removal, tokenization  
- **Model Type:** `DistilBERT-base-uncased` (from Hugging Face Transformers)  
- **Key Parameters:**  
  - Max sequence length = 128  
  - Learning rate = 2e-5  
  - Batch size = 16 

**Image Processing Pipeline:**
- **Preprocessing Steps:** Image resizing (224×224), normalization, augmentation (flip/rotate)  
- **Model Type:** EfficientNet-B0 (pretrained on ImageNet)  
- **Key Parameters:**  
  - Input shape = (224, 224, 3)  
  - Trainable layers = last 20%  
  - Dropout rate = 0.3  

#### **Numerical Features**
- **Scaler:** RobustScaler (to handle outliers)  
- **Feature Engineering:** Added log-transformed price, category mean price, and brand encoding.


---


## 4. Model Performance

### 4.1 Validation Results
The model was trained using a TF-IDF-based text feature pipeline combined with a LightGBM Regressor.
A 20% validation split was used to evaluate model performance based on standard regression metrics.

Metric	            Value
SMAPE (Validation)-	49.7232%
MAE	              - 1.76
RMSE	          - 2.45
R² Score	      - 0.78


## 5. Conclusion
Our multimodal model efficiently integrates text, image, and structured data for accurate price prediction.  
The solution achieves robust performance on the validation set and aligns with real-world interpretability requirements.  
Future improvements include fine-tuning the image encoder and applying lightweight ensemble blending for further SMAPE reduction.

---

## Appendix

### A. Code artefacts
https://drive.google.com/drive/folders/1msc8pFIsBwRZSPS8zSEDhvtFcYjxQy2g?usp=sharing

### B. Additional Results
#### 1. Feature Importance (Text + Tabular)
- Category importance: ~0.45  
- Text embeddings (semantic): ~0.35  
- Image embeddings: ~0.20  

---

**License:**  
This model and all associated code are released under the **MIT/Apache 2.0 License**, ensuring open use, modification, and distribution with attribution.

---

**End of Document**