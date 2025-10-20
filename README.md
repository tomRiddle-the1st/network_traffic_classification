# Network Traffic Classification of 5G networks for Improvement of QoS


**Author:** Mohammed Aqeel Ismail  
**Institution:** School of Mathematics,   
Statistics and Computer Science,  
College of Agriculture, Engineering and Science,   
University of KwaZulu-Natal, Pietermaritzburg Campus,   
Pietermaritzburg 3201,  
RSA  
**Year:** 2025  

---

## Table of Contents  
⦁	[Overview](#overview)   
⦁	[Key Features](#key-features)   
⦁	[System Requirements](#system-requirements)  
⦁	[Installation](#installation)  
⦁	[Quick Start](#quick-start)  
⦁	[Dataset Preparation](#dataset-preparation)  
⦁	[Configuration](#configuration)  
⦁	[Usage Guide](#usage-guide)  
⦁	[Output Files](#output-files)  
⦁	[Performance Benchmarks](#performance-benchmarks)  
⦁	[Troubleshooting](#troubleshooting)  
⦁	[Project Structure](#project-structure)  
⦁	[Citation](#citation)  
⦁	[License](#license)  

---

## Overview

This system implements a **supervised machine learning pipeline** for classifying mobile application network traffic using the **MIRAGE-2019 dataset**. The primary goal is to enable **Quality of Service (QoS) optimization** in mobile networks through accurate traffic classification without payload inspection, ensuring user privacy while maintaining high performance.

PLease note dataset was too large to upload to GitHub but it can be downloaded from Kaggle: https://www.kaggle.com/datasets/programmerrdai/mirage-2019  
Should contain zip files from devices Google Nexus and Xiaomi Mi5 which was used for this project  
I had taken the files from each device folder and combined them into one folder named /data

Also the model.pkl is not available with the rest of the files. It was too large to upload. But if the script with code and dataset are downloaded and set up as outined by this document it will be able to train the model and you should have your own model.pkl locally as well as other output files in the output folder. The script creates output and chart folder if they are not already created.

### Key Capabilities

**Privacy-Preserving Classification**  
⦁	No packet payload inspection required  
⦁	Works with encrypted traffic (HTTPS, VPN, TLS)  
⦁	Compliant with data protection regulations (GDPR)  

 **High Accuracy Performance**  
⦁	**89.76% overall classification accuracy**  
⦁	Consistent performance across all QoS priority levels  
⦁	Robust handling of class imbalance  

 **Intelligent QoS Policy Generation**  
⦁	Automatic mapping from applications to QoS requirements  
⦁	Priority-based resource allocation recommendations  
⦁	Bandwidth, latency, and jitter tolerance specifications  

 **Comprehensive Evaluation**  
⦁	Detailed performance metrics (Accuracy, F1-Score, Precision, Recall)  
⦁	Visual analytics (Confusion Matrix, Distribution Charts)  
⦁	Per-application and per-priority-level analysis  

 **Flexible & Configurable**  
⦁	Toggle between SMOTE and Random Oversampling  
⦁	Optional Linear Discriminant Analysis (LDA)  
⦁	Adjustable Random Forest hyperparameters  

### Research Context  

This implementation is part of research on **"Network Traffic Classification of 5G networks for Improvement of QoS"**. The system addresses the challenge of identifying application types in modern encrypted networks to enable intelligent QoS policies.

**Problem Addressed:**
⦁	Traditional port-based classification: 30-70% accuracy (obsolete)  
⦁	Deep Packet Inspection (DPI): Ineffective with encryption, privacy concerns  
⦁	**Our Solution:** Flow-level statistical features + Random Forest = 89% accuracy  

---

## Key Features

### 1. **Traffic Classification**  
⦁	Classifies 20 mobile application types from MIRAGE-2019 dataset  
⦁	Uses flow-level statistical features (packet size, inter-arrival time, flow duration)  
⦁	Random Forest ensemble classifier with 300 trees  

### 2. **QoS Policy Framework**
Each classified application receives detailed QoS recommendations:  
⦁	**Priority Level:** High, Medium, Low  
⦁	**Bandwidth Requirements:** Low, Medium, High, Variable  
⦁	**Latency Sensitivity:** Low, Medium, High, Very High  
⦁	**Jitter Tolerance:** Very Low, Low, Medium, High  

### 3. **Supported Applications**
| Category | Applications | QoS Priority | 
|----------|-------------|--------------|
| **Navigation** | Waze | High |
| **VoIP/Messaging** | Viber, Messenger | High |
| **Gaming** | Slither.io | High |
| **Music Streaming** | Spotify | High |
| **Video Streaming** | YouTube | Medium |
| **Social Media** | Facebook, Twitter, Pinterest | Medium |
| **E-Commerce** | Wish, Subito, Groupon | Medium |
| **Travel** | TripAdvisor, Foursquare | Medium |
| **Weather** | AccuWeather | Medium |
| **Sports** | iLiga | Medium |
| **Cloud Storage** | Dropbox | Low |
| **Productivity** | Trello, Duolingo | Low |
| **Entertainment** | Comics Reader | Low |

### 4. **Advanced ML Techniques**
⦁	**Class Imbalance Handling:** Random Oversampling (preferred) or SMOTE  
⦁	**Feature Scaling:** StandardScaler normalization  
⦁	**Dimensionality Reduction:** Optional LDA  
⦁	**Model Persistence:** Save/load trained models for reuse  

### 5. **Visualization Suite**
⦁	Application traffic distribution bar chart (color-coded by priority)  
⦁	QoS priority distribution pie chart  
⦁	Latency sensitivity distribution pie chart  
⦁	Bandwidth requirements distribution pie chart  
⦁	Confusion matrix heatmap  
⦁	Feature importance analysis  

---

## System Requirements

### Minimum Requirements
⦁	**Operating System:** Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)  
⦁	**Python:** 3.8 or higher  
⦁	**RAM:** 8GB (16GB recommended for full dataset)  
⦁	**Storage:** 5GB free space  
⦁	**CPU:** Multi-core processor (4+ cores recommended)  

### Recommended Configuration
⦁	**RAM:** 16GB or higher  
⦁	**CPU:** Intel i5/i7 or AMD Ryzen 5/7 (8+ cores)  
⦁	**Storage:** SSD for faster data processing  
⦁	**GPU:** Not required (CPU-only implementation)  

### Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **pandas** | ≥ 1.3.0 | Data manipulation and analysis |
| **numpy** | ≥ 1.21.0 | Numerical computing |
| **scikit-learn** | ≥ 1.0.0 | Machine learning algorithms |
| **matplotlib** | ≥ 3.5.0 | Visualization and plotting |
| **seaborn** | ≥ 0.11.0 | Statistical data visualization |
| **imbalanced-learn** | ≥ 0.8.0 | Oversampling techniques |
| **joblib** | ≥ 1.1.0 | Model persistence |
| **scipy** | ≥ 1.7.0 | Scientific computing |

---

## Installation

### Method 1: Using pip (Recommended)

#### Step 1: Clone the Repository
```bash
git clone https://github.com/tomRiddle-the1st/network_traffic_classification.git
cd network-traffic-qos-classifier
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, imblearn; print(' All dependencies installed successfully!')"
```

#### Step 4: Verify Installation
```bash
python --version  # Should show Python 3.8+
pip list | grep scikit-learn  # Verify scikit-learn installation
```

### Method 2: Using Conda

```bash
# Create conda environment
conda create -n traffic-classifier python=3.9

# Activate environment
conda activate traffic-classifier

# Install dependencies
conda install pandas numpy scikit-learn matplotlib seaborn joblib scipy
pip install imbalanced-learn

# Verify installation
python -c "import pandas, sklearn, imblearn; print(' Installation successful!')"
```

### Method 3: Manual Installation

If `requirements.txt` is not available, install packages individually:
```bash
pip install pandas>=1.3.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install imbalanced-learn>=0.8.0
pip install joblib>=1.1.0
pip install scipy>=1.7.0
```

---

## Quick Start

### 1. Prepare Your Data Structure

Create the following directory structure:
```
project-root/
├── network_traffic_classification.py  # Main script
├── README.md                          # This file
└── data/                              # Your dataset folder
    ├── waze_traffic.json
    ├── youtube_traffic.json
    ├── spotify_traffic.json
    └── ... (other application JSON files)
```

### 2. Verify Dataset Format

Ensure your JSON files follow the MIRAGE-2019 format:
```json
{
  "flow_id_1": {
    "flow_features": {
      "packet_length": {
        "forward": {
          "mean": 125.5,
          "std": 45.2,
          "min": 60,
          "max": 1500
        },
        "backward": {
          "mean": 98.3,
          "std": 32.1,
          "min": 40,
          "max": 1200
        }
      }
    }
  }
}
```

### 3. Run the System

```bash
# Activate your virtual environment first
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Run the classification system
python network_traffic_classification.py
```

### 4. Expected Output

The system will execute the following pipeline:

```
NETWORK TRAFFIC CLASSIFICATION

Step 1: Aggregating flow data...
Processing waze with 1234 flows
Processing youtube with 5678 flows
...
Saved aggregate dataset to Mirage_flows.csv

Step 2: Preprocessing...
Dataset shape: (121955, 102)
Number of apps: 20
[Bar chart displayed]

Step 3: Train model...
Applied Random Oversampling: (121955, 102) ...flows sample to (237300, 102)
Train set: (97564, 102)
Test set: (24391, 102)
Training model...
Model training completed!
Total training time: 2.00 minutes and 15.43 seconds

Step 4: Evaluating model performance...
Accuracy: 0.8976
Macro F1-Score: 0.8971
Weighted F1-Score: 0.8971
[Confusion matrix displayed]
[Top features listed]
[QoS recommendations printed]

Step 5: QoS analysis (with charts)...
[Pie charts displayed]

Saving Model and Metadata
Model and metadata saved to model.pkl
Feature scaler saved to feature_scaler.pkl

Execution completed successfully
Total execution time: 5 minutes and 32.18 seconds
```

---

## Dataset Preparation

### MIRAGE-2019 Dataset

**Dataset Information:**  
⦁	**Name:** MIRAGE-2019  
⦁	**Source:** University of Naples Federico II  
⦁	**Applications:** 20 popular Android apps  
⦁	**Format:** JSON files with flow-level features  
⦁	**Size:** Variable (typically 100MB - 5GB depending on apps selected)  

**Download Instructions:**  
1. Visit: https://www.kaggle.com/datasets/programmerrdai/mirage-2019    
2. Download the MIRAGE-2019 dataset  
3. Extract JSON files to your `data/` directory  

### Supported Application Filenames

The system automatically extracts labels from filenames. Supported patterns:
⦁	`com.waze_*.json` → waze  
⦁	`com.google.android.youtube_*.json` → youtube  
⦁	`com.spotify.music_*.json` → spotify  
⦁	`com.facebook.katana_*.json` → facebook  
⦁	`com.facebook.orca_*.json` → messenger  
⦁	`air.com.hypah.io.slither_*.json` → slither  
⦁	And more...  

### Using Custom Datasets

To use your own dataset:

**1. Format Requirements:**
Your JSON files must contain flow-level features in this structure:
```json
{
  "flow_identifier": {
    "flow_features": {
      "feature_category": {
        "forward": {"stat_name": value},
        "backward": {"stat_name": value}
      }
    }
  }
}
```

**2. Update Label Extraction:**
Modify the `extract_label()` function to match your filename convention:
```python
def extract_label(filename):
    # Add your custom logic here
    if "myapp" in filename:
        return "myapp"
    # ... existing logic
```

**3. Add QoS Policies:**
Define QoS requirements for your applications:
```python
qos_policies = {
    'myapp': {
        'priority': 'high',
        'bandwidth': 'medium',
        'latency_sensitivity': 'high',
        'jitter_tolerance': 'low'
    }
}
```

---

## Configuration

### Basic Configuration

Edit configuration variables at the top of `network_traffic_classification.py`:

```python
# =========================
# CONFIGURATION SETTINGS
# =========================

# File paths
data_folder = "data"                     # Input data directory
processed_data = "Mirage_flows.csv"      # Cached processed dataset
Model = "model.pkl"                      # Output model file
scaler_file = "feature_scaler.pkl"       # Feature scaler file

# Machine Learning Techniques (Toggle on/off)
use_SMOTE = False                        # Use SMOTE oversampling
use_random_oversampling = True           # Use Random Oversampling (recommended)
use_lda = False                          # Apply dimensionality reduction

# Display Settings
top_features = 10                        # Number of top features to display
```

### Advanced Configuration

#### Random Forest Hyperparameters

Modify in the `train_model()` function:
```python
rf_model = RandomForestClassifier(
    n_estimators=300,          # Number of trees in forest
    max_depth=25,             # Maximum depth of each tree
    min_samples_split=5,      # Min samples required to split node
    min_samples_leaf=2,       # Min samples required at leaf node
    class_weight="balanced",  # Handle class imbalance
    random_state=42,          # Seed for reproducibility
    n_jobs=-1                 # Use all CPU cores
)
```

**Tuning Guidelines:**
- **Increase accuracy:** Increase `n_estimators` (200-500)  
- **Reduce overfitting:** Decrease `max_depth` (15-20)  
- **Speed up training:** Decrease `n_estimators`, increase `min_samples_split`  
- **Handle imbalance:** Keep `class_weight="balanced"`  

#### Train-Test Split

Adjust in `train_model()` function:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_resampled, 
    test_size=0.2,           # 20% for testing (adjust 0.1-0.3)
    stratify=y_resampled,    # Maintain class proportions
    random_state=42          # Reproducible split
)
```

#### Oversampling Configuration

**Random Oversampling (Current Default):**
```python
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)
```

**SMOTE (Alternative):**
```python
smote = SMOTE(
    random_state=42,
    k_neighbors=5,           # Number of nearest neighbors
    sampling_strategy='auto' # Balance all classes
)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
```

#### LDA Configuration

When `use_lda = True`:
```python
n_components = min(
    len(y.unique()) - 1,     # Maximum LDA components
    X_resampled.shape[1],    # Current feature count
    50                       # User-defined maximum
)
lda = LinearDiscriminantAnalysis(n_components=n_components)
```

### QoS Policy Customization

Define custom QoS policies for your applications:

```python
qos_policies = {
    'application_name': {
        'priority': 'high',              # Options: high, medium, low
        'bandwidth': 'medium',           # Options: low, medium, high, variable
        'latency_sensitivity': 'high',   # Options: low, medium, high, very_high
        'jitter_tolerance': 'low'        # Options: very_low, low, medium, high
    }
}
```

**QoS Policy Guidelines:**

| Application  | Priority | Bandwidth | Latency | Jitter |
|-----------------|----------|-----------|---------|--------|
| Waze | High | Medium | High | Low |
| Spotify | High | Medium | Very High | Very Low |
| Facebook | Medium | High | Medium | Medium |
| Messenger | Medium | Medium | Medium | High |
| Comic | Low | Variable | Low | High |
| iliga | Low | Low | Low | High |

---

## Usage Guide

### Command Line Execution

#### Basic Usage
```bash
python network_traffic_classification.py
```

#### With Configuration Changes
Edit the script first, then run:
```python
# In network_traffic_classification.py
use_SMOTE = True
use_random_oversampling = False
use_lda = True
top_features = 20
```

```bash
python network_traffic_classification.py
```

### Using the Trained Model

#### Loading and Using a Saved Model

```python
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model_info = joblib.load('model.pkl')
model = model_info['model']
scaler = model_info['scaler']
qos_policies = model_info['qos_policies']

# Load new data (ensure same preprocessing)
new_data = pd.read_csv('new_traffic_data.csv')
X_new = new_data.drop(columns=['flow_id', 'Label'])
X_new = X_new.fillna(0)

# Scale features
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_new_scaled)
confidence = model.predict_proba(X_new_scaled)

# Get QoS recommendations
for pred in predictions:
    policy = qos_policies.get(pred, qos_policies['default'])
    print(f"Application: {pred}")
    print(f"  Priority: {policy['priority']}")
    print(f"  Bandwidth: {policy['bandwidth']}")
```

### Batch Processing Multiple Datasets

```python
import os
from glob import glob

data_folders = ['dataset1', 'dataset2', 'dataset3']

for folder in data_folders:
    print(f"Processing {folder}...")
    data_folder = folder
    
    # Run the pipeline
    data = aggregate()
    X, y = preprocess(data)
    model, X_test, y_test, scaler = train_model(X, y)
    evaluate_model_with_qos(model, X_test, y_test, y)
    
    # Save model with folder-specific name
    joblib.dump(model, f"model_{folder}.pkl")
```

---

## Output Files

### Generated Files

| File | Description | Size | Format |
|------|-------------|------|--------|
| `Mirage_flows.csv` | Cached processed dataset | Variable | CSV |
| `model.pkl` | Trained Random Forest model | 150-200 MB | Pickle |
| `feature_scaler.pkl` | StandardScaler parameters | <1 MB | Pickle |
| `lda_transformer.pkl` | LDA transformer (if enabled) | <10 MB | Pickle |

### Model Information Structure

The saved `model.pkl` contains:
```python
{
    'model': RandomForestClassifier,      # Trained model
    'scaler': StandardScaler,             # Feature scaler
    'feature_names': list,                # List of feature names
    'classes': list,                      # Application labels
    'qos_policies': dict,                 # QoS policy mappings
    'configuration': {                    # Training configuration
        'use_SMOTE': bool,
        'use_random_oversampling': bool,
        'use_lda': bool,
        'top_features': int
    }
}
```

### Console Output

The system prints detailed progress information:

```
Step 1: Aggregating flow data...
├─ Processed waze with 1234 flows
├─ Processed youtube with 5678 flows
└─ Saved aggregate dataset

Step 2: Preprocessing...
├─ Dataset shape: (121955, 102)
└─ [Bar chart displayed]

Step 3: Train model...
├─ Applied Random Oversampling
├─ Train set: (97564, 102)
├─ Test set: (24391, 102)
└─ Training time: 2.15 minutes

Step 4: Evaluating model...
├─ Accuracy: 0.8976
├─ Macro F1: 0.8971
├─ Weighted F1: 0.8971
├─ [Confusion matrix displayed]
└─ [QoS recommendations printed]

Step 5: QoS analysis...
└─ [Distribution charts displayed]
```

### Visualizations

The system generates several plots:

1. **Application Traffic Distribution** (Bar Chart)  
⦁	   Shows number of flows per application  
⦁	   Color-coded by QoS priority level  
   
2. **QoS Priority Distribution** (Pie Chart)  
⦁	   High, medium, low priority breakdown  
⦁	   Percentage and flow count annotations  

3. **Latency Sensitivity Distribution** (Pie Chart)  
⦁	   Very high, high, medium, low sensitivity  
⦁	   Flow count for each category  

4. **Bandwidth Requirements** (Pie Chart)  
⦁	   Low, medium, high, variable bandwidth needs  
⦁	   Proportional representation  

5. **Confusion Matrix** (Heatmap)  
⦁	   Per-application classification accuracy  
⦁	   Identifies misclassification patterns  

---

## Performance Benchmarks

### Accuracy Metrics

| Configuration | Accuracy | Macro F1 | Weighted F1 | Training Time |
|--------------|----------|----------|-------------|---------------|
| **Random Oversampling** | **89.76%** | **0.8971** | **0.8971** | 2-3 min |
| SMOTE | 82,06% | 0.8206 | 0.8206 | 3-5 min |
| No Oversampling, No SMOTE, No LDA | 71.97% | 0.6706 | 0.7225 | 1-2 min |
| SMOTE With LDA | 75.64% | 0.7561 | 0.7561 | 2-3 min |
| Random Oversampling With LDA | 86.43% | 0.8631 | 0.8631 | 2-3 min |

### QoS Priority Performance

| Priority Level | Average F1-Score | Number of Apps |
|---------------|------------------|----------------|
| **High Priority** | 0.9323 | 5 apps |
| **Medium Priority** | 0.8845 | 11 apps |
| **Low Priority** | 0.8904 | 4 apps |

### System Performance vs Dataset Size

| Dataset Size | RAM Used | Processing Time | Recommended RAM |
|-------------|----------|-----------------|-----------------|
| < 10K flows | 2-4 GB | 30 sec - 2 min | 8 GB |
| 10K-50K flows | 4-6 GB | 2-5 min | 8 GB |
| 50K-100K flows | 6-8 GB | 5-10 min | 16 GB |
| 100K-500K flows | 8-12 GB | 10-30 min | 16 GB |
| > 500K flows | 12-16 GB | 30-60 min | 32 GB |

### Hardware Performance

**Test System Specifications:**  
⦁	 **CPU:** Intel Core i7-10700K (8 cores, 16 threads)  
⦁	 **RAM:** 16 GB DDR4  
⦁	 **Storage:** NVMe SSD  
⦁	 **OS:** Windows 10 / Ubuntu 20.04  

**Training Performance:**
⦁	Dataset Size: 237300 flows (after oversampling)  
⦁	Features: 102 flow-level statistics  
⦁	Training Time: 2 minutes 15 seconds  
⦁	Prediction Time: <1ms per flow  
⦁	Model Size: 180 MB  

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors

**Problem:** `MemoryError: Unable to allocate array`

**Solutions:**
```python
# Option 1: Reduce dataset size
sample_size = 50000
if len(data) > sample_size:
    data = data.sample(n=sample_size, random_state=42)

# Option 2: Reduce Random Forest complexity
rf_model = RandomForestClassifier(
    n_estimators=100,     # Reduced from 300
    max_depth=15,         # Reduced from 25
    n_jobs=2              # Limit parallel jobs
)

# Option 3: Disable LDA
use_lda = False

# Option 4: Use SMOTE instead of Random Oversampling (generates less data)
use_SMOTE = True
use_random_oversampling = False
```

#### 2. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'sklearn'`

**Solutions:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install individually
pip install scikit-learn
pip install imbalanced-learn

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

#### 3. JSON Parsing Errors

**Problem:** `json.decoder.JSONDecodeError`

**Solution:** The script now handles invalid JSON files automatically:
```python
# Already implemented in process_file()
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except json.JSONDecodeError:
    print(f"Warning: Skipping invalid JSON file {file_path}")
    return pd.DataFrame()
```

#### 4. File Not Found Errors

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory: 'data'`

**Solution:**
```bash
# Create data directory
mkdir data

# Verify JSON files exist
ls data/*.json

# Check current directory
pwd
```

#### 5. Low Accuracy

**Problem:** Accuracy below 80%

**Possible Causes and Solutions:**
```python
# 1. Insufficient data - Check dataset size
print(f"Total flows: {len(data)}")
# Need at least 1000+ flows per application

# 2. Class imbalance - Enable oversampling
use_random_oversampling = True

# 3. Poor feature quality - Check for missing values
print(f"Missing values: {X.isnull().sum().sum()}")

# 4. Overfitting - Reduce model complexity
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10
)
```

#### 6. Slow Performance

**Problem:** Training takes too long

**Optimizations:**
```python
# 1. Reduce number of trees
n_estimators=100  # Instead of 300

# 2. Limit tree depth
max_depth=15  # Instead of 25

# 3. Use fewer CPU cores
n_jobs=4  # Instead of -1 (all cores)

# 4. Sample large datasets
if len(data) > 100000:
    data = data.sample(n=100000, random_state=42)

```

#### 7. Visualization Issues

**Problem:** Plots not displaying

**Solutions:**
```python
# Add at end of script
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg'
plt.show(block=True)

# For Jupyter notebooks
%matplotlib inline
```

#### 8. Permission Errors

**Problem:** `PermissionError: [Errno 13] Permission denied`

**Solutions:**
```bash
# On Windows - Run as Administrator
# On Linux/macOS - Check permissions
chmod 755 network_traffic_classification.py

# Or save to user directory
processed_data = os.path.join(os.path.expanduser('~'), 'Mirage_flows.csv')
```

### Getting Help

If you encounter issues not covered here:

1. **Check the Error Message:** Read the full error traceback  
2. **Verify Installation:** Ensure all dependencies are installed  
3. **Check Dataset Format:** Verify JSON files match expected structure  
4. **Enable Debug Mode:** Add print statements to identify issues  
5. **Check System Resources:** Monitor RAM and CPU usage  

**Support Channels:**
⦁	GitHub Issues: https://github.com/tomRiddle-the1st/repo/issues  
⦁	Email: aqeelismail06@gmail.com  
⦁	Documentation: See USER_MANUAL.md for detailed guides  

---

## Project Structure

```
network_traffic_classification/
│
├── network_traffic_classification.py  # Main script
├── README.md                          # This file
├── USER_MANUAL.md                     # Detailed user manual
├── RESEARCH_PAPER.md                  # Full research article                           
│
├── data/                              # Input data directory
│   ├── waze_traffic.json
│   ├── youtube_traffic.json
│   ├── spotify_traffic.json
│   └── ... (other application JSON files)
│
├── output/                            # Generated outputs
    ├── Mirage_flows.csv               # Processed dataset
    ├── model.pkl                      # Trained model
    ├── feature_scaler.pkl             # Feature scaler
    ├── lda_transformer.pkl            # LDA transformer (optional)
    └── charts/                # Generated plots
        ├── traffic_distribution.png
        ├── priority_distribution.png
        ├── latency_distribution.png
        ├── bandwidth_distribution.png
        └── confusion_matrix.png

```

---

## Citation

If you use this software in your research, please cite:

```bibtex
@title={Network Traffic Classification of 5G networks for Improvement of QoS},
  author={Ismail, Mohammed Aqeel},
  year={2025},
  note={Software available at: https://github.com/tomRiddle-the1st/network_traffic_classification}
}
```

**Related Publications:**
⦁	MIRAGE-2019 Dataset: Aceto et al., "MIRAGE: Mobile-app Traffic Capture and Ground-truth Creation", IEEE ICCCS 2019

---

## License

This project is developed for educational purposes as part of COMP700 coursework.

Usage Terms:

⦁	Free for educational and personal use  
⦁	Modifications and improvements encouraged  
⦁	Attribution appreciated  

```
