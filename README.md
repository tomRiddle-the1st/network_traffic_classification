# Network Traffic Classification of 5G networks for Improvement of QoS

**Author:** Mohammed Aqeel Ismail  
**Institution:** School of Mathematics, Statistics and Computer Science,  
College of Agriculture, Engineering and Science,  
University of KwaZulu-Natal, Pietermaritzburg Campus,  
Pietermaritzburg 3201, RSA  
**Year:** 2025  

---

## Table of Contents  
⦁ [Overview](#overview)  
⦁ [Key Features](#key-features)  
⦁ [System Requirements](#system-requirements)  
⦁ [Installation](#installation)  
⦁ [Quick Start](#quick-start)  
⦁ [Dataset Preparation](#dataset-preparation)  
⦁ [Configuration](#configuration)  
⦁ [Usage Guide](#usage-guide)  
⦁ [Output Files](#output-files)  
⦁ [Performance Benchmarks](#performance-benchmarks)  
⦁ [Troubleshooting](#troubleshooting)  
⦁ [Project Structure](#project-structure)  
⦁ [Citation](#citation)  
⦁ [License](#license)  

---

## Overview

This system implements a **supervised machine learning pipeline** for classifying mobile application network traffic using the **MIRAGE-2019 dataset**. The primary goal is to enable **Quality of Service (QoS) optimization** in mobile networks through accurate traffic classification without payload inspection, ensuring user privacy while maintaining high performance.

**Please note:** The dataset was too large to upload to GitHub but can be downloaded from Kaggle: https://www.kaggle.com/datasets/programmerrdai/mirage-2019  

The downloaded dataset contains zip files from devices Google Nexus and Xiaomi Mi5. For the purpose of this project, JSON files from both devices should be combined into one folder named "data". The script will open the folder, load each JSON file, extract labels from filenames, extract flow-level statistical features from each file, and combine everything into a single CSV file (Mirage.csv).

**Also note:** The model.pkl file is not available with the rest of the files due to its large size. However, if you download the script with code and dataset and set them up as outlined in this document, the script will train the model and create your own model.pkl locally, along with other output files in the output folder. The script automatically creates the "output" and "charts" folders if they don't already exist.

### Key Capabilities

**Privacy-Preserving Classification**  
⦁ No packet payload inspection required  
⦁ Works with encrypted traffic (HTTPS, VPN, TLS)  
⦁ Compliant with data protection regulations (GDPR)  

**High Accuracy Performance**  
⦁ **89.76% overall classification accuracy**  
⦁ Consistent performance across all QoS priority levels  
⦁ Robust handling of class imbalance  

**Intelligent QoS Policy Generation**  
⦁ Automatic mapping from applications to QoS requirements  
⦁ Priority-based resource allocation recommendations  
⦁ Bandwidth, latency, and jitter tolerance specifications  

**Comprehensive Evaluation**  
⦁ Detailed performance metrics (Accuracy, F1-Score, Precision, Recall)  
⦁ Visual analytics (Confusion Matrix, Distribution Charts)  
⦁ Per-application and per-priority-level analysis  

**Flexible & Configurable**  
⦁ Toggle between SMOTE and Random Oversampling  
⦁ Optional Linear Discriminant Analysis (LDA)  
⦁ Adjustable Random Forest hyperparameters  

### Research Context  

This implementation is part of research on **"Network Traffic Classification of 5G networks for Improvement of QoS"**. The system addresses the challenge of identifying application types in modern encrypted networks to enable intelligent QoS policies.

**Problem Addressed:**
⦁ Traditional port-based classification: 30-70% accuracy (obsolete)  
⦁ Deep Packet Inspection (DPI): Ineffective with encryption, privacy concerns  
⦁ **Our Solution:** Flow-level statistical features + Random Forest = 89% accuracy  

---

## Key Features

### 1. **Traffic Classification**  
⦁ Classifies 20 mobile application types from MIRAGE-2019 dataset  
⦁ Uses flow-level statistical features (packet size, inter-arrival time, flow duration)  
⦁ Random Forest ensemble classifier with 300 trees  

### 2. **QoS Policy Framework**
Each classified application receives detailed QoS recommendations:  
⦁ **Priority Level:** High, Medium, Low  
⦁ **Bandwidth Requirements:** Low, Medium, High, Variable  
⦁ **Latency Sensitivity:** Low, Medium, High, Very High  
⦁ **Jitter Tolerance:** Very Low, Low, Medium, High  

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
| **Sports** | Motain (iLiga) | Medium |
| **Cloud Storage** | Dropbox | Low |
| **Productivity** | Trello, Duolingo | Low |
| **Entertainment** | Comics Reader | Low |

### 4. **Advanced ML Techniques**
⦁ **Class Imbalance Handling:** Random Oversampling (preferred) or SMOTE  
⦁ **Feature Scaling:** StandardScaler normalization  
⦁ **Dimensionality Reduction:** Optional LDA  
⦁ **Model Persistence:** Save/load trained models for reuse  

### 5. **Visualization Charts**
⦁ Application traffic distribution bar chart (color-coded by priority)  
⦁ QoS priority distribution pie chart  
⦁ Latency sensitivity distribution pie chart  
⦁ Bandwidth requirements distribution pie chart  
⦁ Confusion matrix heatmap  
 
---

## System Requirements

### Minimum Requirements
⦁ **Operating System:** Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)  
⦁ **Python:** 3.8 or higher  
⦁ **RAM:** 8GB (16GB recommended for full dataset)  
⦁ **Storage:** 5GB free space  
⦁ **CPU:** Multi-core processor (4+ cores recommended)  
⦁ **IDE (Optional):** Visual Studio Community 2022 or any Python IDE

### Recommended Configuration
⦁ **RAM:** 16GB or higher  
⦁ **CPU:** Intel i5/i7 or AMD Ryzen 5/7 (8+ cores)  
⦁ **Storage:** SSD for faster data processing  
⦁ **GPU:** Not required (CPU-only implementation)  

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

### Method 1: Using Visual Studio IDE Community 2022 (Recommended for Windows)

#### Step 1: Install Visual Studio 2022
1. Download from: https://visualstudio.microsoft.com/downloads/
2. During installation, select **Python development** workload
3. Ensure Python 3.8+ is included in the installation

#### Step 2: Clone or Download the Repository
```bash
# Option A: Clone with Git
git clone https://github.com/tomRiddle-the1st/network_traffic_classification.git

# Option B: Download ZIP from GitHub
# Extract to your desired location
```

#### Step 3: Open the Project in Visual Studio
1. Launch **Visual Studio Community 2022**
2. Select **File → Open → Folder**
3. Navigate to and select the `network_traffic_classification` folder
4. Visual Studio will automatically detect the Python project

#### Step 4: Set Up Python Environment
1. Go to **View → Other Windows → Python Environments**
2. Click **+ Add Environment** (or the gear icon)
3. Select **Virtual Environment**
4. Choose **Python 3.8** or higher as the base interpreter
5. Set environment location to `venv` folder in project directory
6. Click **Create**
7. Wait for environment creation (may take 1-2 minutes)

#### Step 5: Install Dependencies

**Option A: Using Package Manager (GUI)**
1. In **Python Environments** window, right-click your virtual environment
2. Select **Manage Python Packages**
3. For each package, search and click **Run command: pip install**:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
   - imbalanced-learn
   - joblib
   - scipy

**Option B: Using Terminal (Faster)**
1. Open **View → Terminal** (or press `Ctrl + ~`)
2. Ensure your virtual environment is activated (VS does this automatically)
3. Create `requirements
```requirements
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
joblib>=1.1.0
scipy>=1.7.0
```


#### Step 6: Verify Installation
In the Visual Studio terminal, run:
```bash
python --version  # Should show Python 3.8+
pip list  # Should show all installed packages
```

#### Step 7: Prepare Dataset
1. Create a `data/` folder in your project root
2. Download MIRAGE-2019 dataset from Kaggle
3. Extract and merge all JSON files from both device folders into `data/`

#### Step 8: Run the Script
**Option A: Using Debug/Run**
1. Open `network_traffic_classification.py` in the editor
2. Press **F5** (Start Debugging) or **Ctrl+F5** (Start Without Debugging)
3. View output in the integrated terminal

**Option B: Using Python Interactive**
1. Right-click anywhere in the editor
2. Select **Execute in Python Interactive**
3. View output and charts in the interactive window

**Option C: Using Terminal**
1. Open **View → Terminal**
2. Run:
```bash
python network_traffic_classification.py
```

---

### Method 2: Using Command Line (Cross-Platform)

#### Step 1: Clone the Repository
```bash
git clone https://github.com/tomRiddle-the1st/network_traffic_classification.git
cd network_traffic_classification
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

#### Step 3: Check requirements
```requirements
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
joblib>=1.1.0
scipy>=1.7.0
```

#### Step 4: Install Dependencies
```bash
# Install all requirements
pip install -r requirements #check above

# Verify installation
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, imblearn; print('All dependencies installed successfully!')"
```

#### Step 5: Verify Installation
```bash
python --version  # Should show Python 3.8+
pip list | grep scikit-learn  # Verify scikit-learn installation
```

---

### Method 3: Manual Installation
Install packages individually:
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

The script automatically creates `output/` and `charts/` folders, but you need to manually create and populate the `data/` folder:

```
project-root/
├── network_traffic_classification.py  # Main script
├── README.md                          # This file
├── requirements                      # Dependencies list
└── data/                              # CREATE THIS MANUALLY
    ├── com.waze_*.json
    ├── com.google.android.youtube_*.json
    ├── com.spotify.music_*.json
    └── ... (other application JSON files from BOTH devices)

# These folders are created automatically by the script:
├── output/                            #Auto-created via script or create the folder manually and the script will save the output files
│   ├── Mirage_flows.csv
│   ├── model.pkl
│   ├── feature_scaler.pkl
│   └── lda_transformer.pkl (if LDA enabled)
└── charts/                            # Auto-created via script or create the folder manually and the script will save the png files
    ├── traffic_distribution.png
    ├── priority_distribution.png    
    ├── latency_sensitivity.png
    ├── bandwidth_requirements.png
    └── confusion_matrix.png
```

### 2. Download and Prepare Dataset

#### Download Instructions:
1. Visit: https://www.kaggle.com/datasets/programmerrdai/mirage-2019
2. Download the MIRAGE-2019 dataset (zip files)
3. Extract the downloaded files - you'll see folders for **Google Nexus** and **Xiaomi Mi5**
4. **IMPORTANT:** Create a `data/` folder in your project root
5. **Merge all JSON files** from BOTH device folders into the single `data/` folder:
   ```
   data/
   ├── 1494419517_com.twitter.android_MIRAGE-2019_traffic_dataset_labeled_biflows.json
   ├── 1494434240_com.google.android.youtube_MIRAGE-2019_traffic_dataset_labeled_biflows.json
   ├── 1494508157_com.spotify.music_MIRAGE-2019_traffic_dataset_labeled_biflows.json
   ├── 1511195631_com.facebook.katana_MIRAGE-2019_traffic_dataset_labeled_biflows.json
   ├── 1511197686_com.facebook.orca_MIRAGE-2019_traffic_dataset_labeled_biflows.json
   ├── 1494596297_air.com.hypah.io.slither_MIRAGE-2019_traffic_dataset_labeled_biflows.json
   └── ... (all other JSON files from both devices)
   ```
6. The script will automatically process all JSON files in this folder regardless of device origin

### 3. Verify Dataset Format

Ensure your JSON files follow the MIRAGE-2019 format:
```json
{
  "192.168.20.101,51221,216.58.205.42,443,6": { #flow_id
    "packet_data": {
  "src_port": [ 51221, 443, 51221, ... ],
  "dst_port": [ 443, 51221,...],
  "packet_dir": [ 0, 1, 0,...],
  "L4_payload_bytes": [ 1368, 0, 1368, ...],
  "iat": [ 0, 0.05589914321899414, 0.2340989112854004,... ],
  "TCP_win_size": [ 1544, 725, ... ],
  "L4_raw_payload": [....}
    "flow_features": {
   "packet_length": {
     "biflow": {
       "min": 52.0,
       "max": 1420.0,
       "mean": 381.9183673469388,
       "std": 469.20914081665745,
       "var": 220157.2178259059,
       "mad": 77.0,
       "skew": 1.334384411480604,
       "kurtosis": 0.3833087760022593,
       "10_percentile": 52.0,
       "20_percentile": 52.0,
       "30_percentile": 52.0,
       "40_percentile": 52.0,
       "50_percentile": 129.0,
       "60_percentile": 168.19999999999993,
       "70_percentile": 500.3999999999984,
       "80_percentile": 685.2000000000011,
       "90_percentile": 1420.0
     },....
}
```

### 4. Run the System

**Using Visual Studio 2022:**
1. Open `network_traffic_classification.py`
2. Press **F5** or click the ** Start** button
3. View output in the integrated terminal

**Using Command Line:**
```bash
# Activate your virtual environment first
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Run the classification system
python network_traffic_classification.py
```

### 5. Expected Output

The system will execute the following pipeline:

```
NETWORK TRAFFIC CLASSIFICATION

Step 1: Aggregating flow data... #processes 1640 json files
Processing waze with 1234 flows
Processing youtube with 5678 flows
...
Saved aggregate dataset to output/Mirage_flows.csv

Step 2: Preprocessing...
Dataset shape: (121955, 102)
Number of apps: 20
Apps with number of flows:
Label
waze           11865
motain         11048
accuweather    10682
... (other applications)          
Saved bar chart plot 
[Bar chart displayed]

Step 3: Train model...
Applied Random Oversampling: (121955, 102) ...flows sampled to (237300, 102)
Class distribution after oversampling:
Label
twitter        11865
youtube        11865
spotify        11865
slither        11865
motain         11865
... (other applications) 
Train set: (189840, 102)
Test set: (47460, 102)
Training model...
Model training completed!
Total training time: 2.00 minutes and 4.03 seconds

Step 4: Evaluating model performance...
Model Evaluation
Accuracy: 0.8976
Macro F1-Score: 0.8971
Weighted F1-Score: 0.8971

Classification Report:
              precision  recall  f1-score     support
accuweather      0.9044  0.8491    0.8759   2373.0000
comics           0.9256  0.9284    0.9270   2373.0000
dropbox          0.8902  0.8883    0.8893   2373.0000
duolingo         0.8344  0.8453    0.8399   2373.0000
facebook         0.8679  0.8858    0.8767   2373.0000
foursquare       0.8988  0.8837    0.8912   2373.0000
... (other applications)

QoS Evaluation
Performance by QoS Priority Level:
High Priority Apps: F1=0.9323 (n=5 apps)
Medium Priority Apps: F1=0.8845 (n=11 apps)
Low Priority Apps: F1=0.8904 (n=4 apps)

[Confusion matrix displayed]
Saved plot: charts/confusion_matrix.png

Top 10 Most Important Features:
packet_length_upstream_flow_max
packet_length_upstream_flow_std
packet_length_upstream_flow_var
packet_length_upstream_flow_90_percentile
packet_length_downstream_flow_max
packet_length_biflow_max
iat_upstream_flow_max
iat_biflow_max
iat_downstream_flow_max
packet_length_upstream_flow_mean

QoS Policy Recommendations for each app
accuweather:
  Classification Confidence: 0.876
  Recommended Priority: medium
  Bandwidth Allocation: low
  Latency Sensitivity: medium
  Jitter Tolerance: high
comics:
  Classification Confidence: 0.927
  Recommended Priority: low
  Bandwidth Allocation: medium
  Latency Sensitivity: low
  Jitter Tolerance: high
(...other applications)

Step 5: QoS analysis (with charts)...
[Pie charts displayed]
Priority Distribution:
high  :  28920 flows ( 23.7%)
medium:  71205 flows ( 58.4%)
low   :  21830 flows ( 17.9%)
	
Latency Sensitivity Distribution:
very_high :   3189 flows (  2.6%)
high      :  19116 flows ( 15.7%)
medium    :  73567 flows ( 60.3%)
low       :  26083 flows ( 21.4%)
	
Bandwidth Distribution:
high      :   6493 flows (  5.3%)
medium    :  77930 flows ( 63.9%)
low       :  32327 flows ( 26.5%)
variable  :   5205 flows (  4.3%)
Saved plot: charts/priority_distribution.png
Saved plot: charts/latency_sensitivity.png
Saved plot: charts/bandwidth_requirements.png

Saving Model and Metadata
Model and metadata saved to output/model.pkl
Feature scaler saved to output/feature_scaler.pkl
Execution completed successfully
Total execution time: 3.0 minutes and 6.14 seconds
```

---

## Dataset Preparation

### MIRAGE-2019 Dataset

**Dataset Information:**  
⦁ **Name:** MIRAGE-2019  
⦁ **Source:** University of Naples Federico II  
⦁ **Applications:** 20 popular Android apps  
⦁ **Devices:** Google Nexus and Xiaomi Mi5
⦁ **Format:** JSON files with flow-level features  
⦁ **Size:** Variable (typically 100MB - 5GB depending on apps selected)  

**Download Instructions:**  
1. Visit: https://www.kaggle.com/datasets/programmerrdai/mirage-2019    
2. Download the MIRAGE-2019 dataset  
3. Extract JSON files from BOTH device folders to your `data/` directory  

### Supported Application Filenames

The system automatically extracts labels from filenames. Supported patterns:
⦁ 1494419517_com.twitter.android_MIRAGE-2019_traffic_dataset_labeled_biflows --> Twitter 
⦁ 1494434240_com.google.android.youtube_MIRAGE-2019_traffic_dataset_labeled_biflows --> youtube  
⦁ 1494508157_com.spotify.music_MIRAGE-2019_traffic_dataset_labeled_biflows --> spotify  
⦁ 1511195631_com.facebook.katana_MIRAGE-2019_traffic_dataset_labeled_biflows --> facebook  
⦁ 1511197686_com.facebook.orca_MIRAGE-2019_traffic_dataset_labeled_biflows --> messenger  
⦁ 1494596297_air.com.hypah.io.slither_MIRAGE-2019_traffic_dataset_labeled_biflows --> slither  
⦁ And more... 

### Using Custom Datasets

To use your own dataset:

**1. Format Requirements:**
Your JSON files must contain flow-level features in this structure:
```json
{
  " "192.168.20.101,51221,216.58.205.42,443,6"": { #flow_id
    "packet_data": {
  "src_port": [ 51221, 443, 51221, ... ],
  "dst_port": [ 443, 51221,...],
  "packet_dir": [ 0, 1, 0,...],
  "L4_payload_bytes": [ 1368, 0, 1368, ...],
  "iat": [ 0, 0.05589914321899414, 0.2340989112854004,... ],
  "TCP_win_size": [ 1544, 725, ... ],
  "L4_raw_payload": [....}
    "flow_features": {
   "packet_length": {
     "biflow": {
       "min": 52.0,
       "max": 1420.0,
       "mean": 381.9183673469388,
       "std": 469.20914081665745,
       "var": 220157.2178259059,
       "mad": 77.0,
       "skew": 1.334384411480604,
       "kurtosis": 0.3833087760022593,
       "10_percentile": 52.0,
       "20_percentile": 52.0,
       "30_percentile": 52.0,
       "40_percentile": 52.0,
       "50_percentile": 129.0,
       "60_percentile": 168.19999999999993,
       "70_percentile": 500.3999999999984,
       "80_percentile": 685.2000000000011,
       "90_percentile": 1420.0
     },....
}
```

**2. Update Label Extraction according to your dataset:**
Modify the `extract_label()` function in the script to match your filename convention:
```python
def extract_label(filename):
    # Add your custom logic here
    if "myapp" in filename:
        return "myapp"
    # ... existing logic
```

**3. Add QoS Policies:**
Define QoS  for your applications in the `qos_policies` dictionary:
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
# CONFIGURATION SETTINGS
# File paths
data_folder = "data"                     # Input data directory
output_folder = "output"                 # Output directory
charts_folder = "charts"                 # Charts directory
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

Define custom QoS policies for your applications in the script:

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

### Visual Studio 2022 Execution

#### Debug Mode (F5)
- Allows breakpoints and step-through debugging
- Useful for troubleshooting issues
- Slightly slower execution

#### Run Without Debugging (Ctrl+F5)
- Faster execution
- Better for production runs
- Shows complete output

#### Interactive Mode
- Right-click in editor → **Execute in Python Interactive**
- Keeps session alive for exploration
- Great for testing modifications

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
├─ Processing waze with 1234 flows
├─ Processing youtube with 5678 flows
└─ Saved aggregate dataset

Step 2: Preprocessing...
├─ Dataset shape: (121955, 102)
└─ [Bar chart displayed]

Step 3: Train model...
├─ Applied Random Oversampling
├─ Train set: (189840, 102)
├─ Test set: (47460, 102)
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

The system generates several plots automatically saved to the `charts/` folder:

1. **Traffic Distribution** (Bar Chart)  
   ⦁ Shows number of flows per application  
   ⦁ Color-coded by QoS priority level (blue=high, orange=medium, beige=low)  
   
2. **Priority Distribution** (Pie Chart)  
   ⦁ High, medium, low priority breakdown  
   ⦁ Percentage and flow count annotations  

3. **Latency Sensitivity Distribution** (Pie Chart)  
   ⦁ Very high, high, medium, low sensitivity  
   ⦁ Flow count for each category  

4. **Bandwidth Requirements** (Pie Chart)  
   ⦁ Low, medium, high, variable bandwidth needs  
   ⦁ Proportional representation  

5. **Confusion Matrix** (Heatmap)  
   ⦁ Per-application classification accuracy  
   ⦁ Identifies misclassification patterns  

---

## Output Files

### Generated Files

All output files are automatically saved to their respective folders:

**output/ folder:**
- `Mirage_flows.csv` - Aggregated and processed dataset
- `model.pkl` - Trained Random Forest model with metadata
- `feature_scaler.pkl` - StandardScaler for feature normalization
- `lda_transformer.pkl` - LDA transformer (only if `use_lda = True`)

**charts/ folder:**
- `traffic_distribution.png` - Application flow distribution bar chart
- `priority_distribution.png` - QoS priority pie chart
- `latency_sensitivity.png` - Latency sensitivity pie chart
- `bandwidth_requirements.png` - Bandwidth requirements pie chart
- `confusion_matrix.png` - Classification confusion matrix heatmap

### File Descriptions

#### Mirage_flows.csv
Contains aggregated flow-level features from all JSON files:
- Columns: flow_id, Label, and 100+ statistical features
- Used for caching to speed up subsequent runs
- Delete this file to force re-processing of raw JSON data

#### model.pkl
Complete model package including:
- Trained RandomForestClassifier
- Feature scaler (StandardScaler)
- Feature names list
- Application class labels
- QoS policy mappings
- Configuration settings used during training

#### feature_scaler.pkl
StandardScaler object for normalizing features:
- Fitted on training data statistics
- Required for preprocessing new data
- Use when making predictions on new flows

#### lda_transformer.pkl (Optional)
Linear Discriminant Analysis transformer:
- Only created when `use_lda = True`
- Reduces feature dimensionality
- Preserves class discriminative information

---

## Performance Benchmarks

### Accuracy Metrics

| Configuration | Accuracy | Macro F1 | Weighted F1 | Training Time |
|--------------|----------|----------|-------------|---------------|
| **Random Oversampling (Recommended)** | **89.76%** | **0.8971** | **0.8971** | 2-3 min |
| SMOTE | 82.06% | 0.8206 | 0.8206 | 3-5 min |
| Baseline (No Techniques) | 71.97% | 0.6706 | 0.7225 | 1-2 min |
| SMOTE + LDA | 75.64% | 0.7561 | 0.7561 | 2-3 min |
| Random Oversampling + LDA | 86.43% | 0.8631 | 0.8631 | 2-3 min |

**Key Findings:**
- Random Oversampling achieves best accuracy (89.76%)
- LDA reduces accuracy by ~3% but speeds up training
- Baseline without oversampling shows significant class bias

### QoS Priority Performance

| Priority Level | Average F1-Score | Number of Apps |
|---------------|------------------|----------------|
| **High Priority** | 0.9323 | 5 apps |
| **Medium Priority** | 0.8845 | 11 apps |
| **Low Priority** | 0.8904 | 4 apps |

**Analysis:**
- High-priority apps (Waze, Viber, Messenger, Slither, Spotify) have highest accuracy
- Critical for QoS - correctly identifies latency-sensitive applications
- Medium priority apps slightly lower due to diverse characteristics

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
⦁ **CPU:** Intel Core i7-10700K (8 cores, 16 threads)  
⦁ **RAM:** 16 GB DDR4  
⦁ **Storage:** NVMe SSD  
⦁ **OS:** Windows 10 / Ubuntu 20.04  

**Training Performance:**
⦁ Dataset Size: 237,300 flows (after oversampling)  
⦁ Features: 102 flow-level statistics  
⦁ Training Time: 2 minutes 15 seconds  
⦁ Prediction Time: <1ms per flow  
⦁ Model Size: ~180 MB  

**Optimization Tips:**
- Use SSD storage for 2-3x faster data loading
- Close unnecessary applications to free RAM
- Enable all CPU cores (`n_jobs=-1` in RandomForest)
- Process dataset in batches if memory is limited

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

**In Visual Studio 2022:**
1. Check **Python Environments** window
2. Verify correct environment is active
3. Right-click environment → **Manage Python Packages**
4. Search and install missing packages

**In Command Line:**
```bash
# Reinstall dependencies
pip install -r requirements #check requirements

# Or install individually
pip install scikit-learn
pip install imbalanced-learn

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

#### 3. JSON Parsing Errors

**Problem:** `json.decoder.JSONDecodeError`

**Solution:** The script already handles invalid JSON files automatically:
```python
# Already implemented in process_file()
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except json.JSONDecodeError:
    print(f"Warning: Skipping invalid JSON file {file_path}")
    return pd.DataFrame()
```

If you still encounter errors:
1. Check if JSON files are corrupted during download/extraction
2. Re-download the dataset
3. Verify file integrity with a JSON validator

#### 4. File Not Found Errors

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory: 'data'`

**Solution:**
```bash
# Create data directory
mkdir data

# Verify JSON files exist
# Windows PowerShell:
Get-ChildItem data\*.json

# Linux/macOS/Git Bash:
ls data/*.json

# Check current directory
pwd  # Linux/macOS
cd   # Windows
```

**In Visual Studio 2022:**
1. Right-click project in Solution Explorer
2. Add → New Folder → Name it "data"
3. Copy JSON files into this folder
4. Verify folder appears in Solution Explorer

#### 5. Low Accuracy

**Problem:** Accuracy below 80%

**Possible Causes and Solutions:**
```python
# 1. Insufficient data - Check dataset size
print(f"Total flows: {len(data)}")
# Need at least 500+ flows per application for good results

# 2. Class imbalance - Enable oversampling
use_random_oversampling = True

# 3. Poor feature quality - Check for missing values
print(f"Missing values: {X.isnull().sum().sum()}")
print(f"Features with >50% missing: {(X.isnull().sum() / len(X) > 0.5).sum()}")

# 4. Overfitting - Reduce model complexity
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10
)

# 5. Using wrong dataset - Verify JSON format matches MIRAGE-2019
```

#### 6. Slow Performance

**Problem:** Training takes too long (>10 minutes)

**Optimizations:**
```python
# 1. Reduce number of trees
n_estimators=100  # Instead of 300

# 2. Limit tree depth
max_depth=15  # Instead of 25

# 3. Use fewer CPU cores (if system is sluggish)
n_jobs=4  # Instead of -1 (all cores)

# 4. Sample large datasets
if len(data) > 100000:
    data = data.sample(n=100000, random_state=42)

# 5. Enable LDA for dimensionality reduction
use_lda = True  # Reduces features, speeds up training
```

**Additional Tips:**
- Close other applications to free system resources
- Use Task Manager (Windows) or Activity Monitor (macOS) to monitor resource usage
- Consider running overnight for very large datasets

#### 7. Visualization Issues

**Problem:** Plots not displaying or saving

**Solutions:**

**In Visual Studio 2022:**
```python
# Add at beginning of script
import matplotlib
matplotlib.use('TkAgg')  # Try different backends: 'Qt5Agg', 'Agg'

# Force plot display
plt.show(block=True)
```

**In Command Line:**
```python
# For headless servers (no display)
import matplotlib
matplotlib.use('Agg')  # Saves plots without displaying

# Verify charts folder exists
import os
if not os.path.exists('charts'):
    os.makedirs('charts')
```

**Common Issues:**
- Charts not appearing: Install `python-tk` package
- Charts not saving: Check folder permissions
- Charts blank: Update matplotlib to latest version



#### 8. Visual Studio Specific Issues

**Problem:** Python environment not detected

**Solution:**
1. Go to **Tools → Options → Python → Environments**
2. Click **+ Add Environment**
3. Manually specify Python interpreter path
4. Common locations:
   - `C:\Python38\python.exe`
   - `C:\Users\<YourName>\AppData\Local\Programs\Python\Python38\python.exe`

**Problem:** Script runs but no output visible

**Solution:**
1. Check **View → Output** window
2. Select "Python" from the dropdown
3. Or use **View → Terminal** for live output

**Problem:** Charts not displaying in VS

**Solution:**
1. Charts save to `charts/` folder automatically
2. View them from Windows Explorer / File Explorer
3. Or use **Python Interactive** window for inline display

#### 9. Dataset Issues

**Problem:** "No JSON files found in data folder"

**Solution:**
1. Verify `data/` folder exists in project root
2. Check JSON files have correct extension (`.json`, not `.txt` or `.json.txt`)
3. Verify files are not in subdirectories
4. In Visual Studio: Set **Show All Files** in Solution Explorer

**Problem:** "Unknown application labels"

**Solution:**
The script extracts labels from filenames. Ensure filenames match pattern:
```
com.app.package_device_capture.json
```

For custom apps, update `extract_label()` function in the script.

#### 10. Windows Path Issues

**Problem:** File paths not working on Windows

**Solutions:**
```python
# The script already uses os.path.join() for cross-platform compatibility
# But if you modify paths, always use:
data_folder = os.path.join("data")  # Not "data/"
output_file = os.path.join("output", "model.pkl")  # Not "output/model.pkl"

# Or use forward slashes (Python handles them on Windows)
data_folder = "data"  # Works on all platforms
```

### Getting Help

If you encounter issues not covered here:

**Step-by-Step Debugging:**
1. **Read the Full Error Message:** Check the complete error traceback
2. **Verify Installation:** Ensure all dependencies are installed correctly
3. **Check Dataset Format:** Verify JSON files match expected MIRAGE-2019 structure
4. **Enable Debug Output:** Add print statements to identify where issues occur
5. **Check System Resources:** Monitor RAM and CPU usage during execution
6. **Test with Small Sample:** Try with 2-3 JSON files first to isolate issues

**Support Channels:**
⦁ **GitHub Issues:** https://github.com/tomRiddle-the1st/network_traffic_classification/issues  
⦁ **Email:** aqeelismail06@gmail.com  
⦁ **Documentation:** See full research paper for technical details

**When Reporting Issues:**
Please include:
- Operating system and version
- Python version (`python --version`)
- Error message (full traceback)
- Dataset size and number of JSON files
- Configuration settings used
- Steps to reproduce the issue

---

## Project Structure

```
network_traffic_classification/
│
├── network_traffic_classification.py  # Main script (all-in-one)
├── README.md                          # This file
├── requirements                  # Python dependencies
├── USER_MANUAL.md                     # Detailed user guide (if available)
├── RESEARCH_PAPER.md                  # Full research article (if available)
│
├── data/                              # Input data directory (CREATE MANUALLY)
│   ├── com.waze_nexus_*.json
│   ├── com.waze_mi5_*.json
│   ├── com.google.android.youtube_nexus_*.json
│   ├── com.google.android.youtube_mi5_*.json
│   ├── com.spotify.music_*.json
│   └── ... (other application JSON files from both devices)
│
├── output/                            # Generated outputs (AUTO-CREATED)
│   ├── Mirage_flows.csv               # Processed dataset cache
│   ├── model.pkl                      # Trained model + metadata
│   ├── feature_scaler.pkl             # Feature normalization scaler
│   └── lda_transformer.pkl            # LDA transformer (if enabled)
│
└── charts/                            # Generated visualizations (AUTO-CREATED)
    ├── traffic_distribution.png       # Application flow distribution
    ├── priority_distribution.png      # QoS priority breakdown
    ├── latency_sensitivity.png        # Latency sensitivity distribution
    ├── bandwidth_requirements.png     # Bandwidth requirements
    └── confusion_matrix.png           # Classification confusion matrix
```

### Key Files Description

**network_traffic_classification.py**
- Main executable script
- Contains all functions: data loading, preprocessing, training, evaluation
- Configurable via variables at top of file
- No external dependencies (except libraries)

**data/ folder**
- Must contain MIRAGE-2019 JSON files
- Merge files from both Google Nexus and Xiaomi Mi5 devices
- Script processes all `.json` files in this directory

**output/ folder**
- Automatically created by script
- Stores trained models and processed data
- Reusable across multiple runs (caching)

**charts/ folder**
- Automatically created by script
- Stores all visualization outputs as PNG files
- High resolution (200 DPI) for publication quality

---

## Citation

If you use this software in your research, please cite:

```bibtex
@{title={Network Traffic Classification of 5G networks for Improvement of QoS},
  author={Ismail, Mohammed Aqeel},
  year={2025},
  school={University of KwaZulu-Natal},
  address={Pietermaritzburg, South Africa},
  note={Software available at: https://github.com/tomRiddle-the1st/network_traffic_classification}
}
```
---

## License

This project is developed for educational and research purposes as part of COMP700 coursework at the University of KwaZulu-Natal.

### Usage Terms

⦁ **Free for educational and personal use**  
⦁ **Free for academic research** (please cite)  
⦁ **Modifications and improvements encouraged**  
⦁ **Attribution appreciated but not required**  
⦁ **No warranty provided** - use at your own risk

### Disclaimer

This software is provided "as is" without warranty of any kind, express or implied. The authors and University of KwaZulu-Natal are not responsible for any damages or issues arising from the use of this software.

### Dataset License

The MIRAGE-2019 dataset is provided by the University of Naples Federico II. Please refer to their licensing terms when using the dataset.

---

## Frequently Asked Questions (FAQ)

### General Questions

**Q: What is the minimum dataset size needed?**  
A: At least 500 flows per application for reasonable accuracy. The full MIRAGE-2019 dataset contains 120K+ flows across 20 apps.

**Q: Can I use this for real-time traffic classification?**  
A: The model is trained for batch classification. For real-time use, you'll need to implement a flow extraction pipeline and load the trained model for predictions.

**Q: Does this work with encrypted traffic?**  
A: Yes! The system uses flow-level statistical features, not packet payloads, so it works with encrypted traffic (HTTPS, VPN, TLS).

**Q: Why Random Oversampling instead of SMOTE?**  
A: In our tests, Random Oversampling achieved 89.76% accuracy vs 82.06% for SMOTE. It's also faster and simpler.

### Technical Questions

**Q: How long does training take?**  
A: 2-3 minutes on a modern CPU (i7/Ryzen 7) with 16GB RAM for the full 237K flow dataset after oversampling.

**Q: Can I add new applications?**  
A: Yes! Add JSON files to `data/`, update `extract_label()` function, and add QoS policies to `qos_policies` dictionary.

**Q: What features does the model use?**  
A: 102 flow-level statistical features including packet length (mean, std, min, max, percentiles), inter-arrival time (IAT), and flow duration for forward, backward, and bidirectional flows.

**Q: Can I use this on a different dataset?**  
A: Yes, but you'll need to ensure your dataset has the same flow-level feature structure as MIRAGE-2019, or modify the `process_file()` function to extract features from your format.

### Troubleshooting Questions

**Q: Why is my accuracy so low (<70%)?**  
A: Common causes: insufficient data per app, disabled oversampling, corrupted JSON files, or wrong dataset format.

**Q: Charts are not displaying. What should I do?**  
A: Charts are automatically saved to `charts/` folder. Check there first. If they're not saving, verify folder permissions and matplotlib installation.

**Q: Script crashes with MemoryError. How to fix?**  
A: Reduce `n_estimators` to 100, limit dataset size with sampling, or disable Random Oversampling (though accuracy will drop).

---

## Acknowledgments

**Dataset:**  
This work uses the MIRAGE-2019 dataset provided by the University of Naples Federico II, Italy.

**Supervisor:**  
University of KwaZulu-Natal, School of Mathematics, Statistics and Computer Science

**Tools & Libraries:**  
- scikit-learn for machine learning algorithms
- imbalanced-learn for oversampling techniques
- matplotlib and seaborn for visualizations
- pandas and numpy for data manipulation

**Inspiration:**  
Modern network traffic classification research and the need for privacy-preserving QoS optimization in 5G networks.

---

## Version History

**v1.0 (2025)**
- Initial release
- Random Forest classifier with 89.76% accuracy
- Support for 20 MIRAGE-2019 applications
- QoS policy framework
- Comprehensive visualization suite
- Visual Studio 2022 compatibility

---

## Contact

**Author:** Mohammed Aqeel Ismail  
**Email:** aqeelismail06@gmail.com  
**Institution:** University of KwaZulu-Natal  
**GitHub:** https://github.com/tomRiddle-the1st/network_traffic_classification

For questions, bug reports, or collaboration opportunities, please open an issue on GitHub or contact via email.

---

**Thank you for using this network traffic classification system!** 

We hope this tool helps advance your research in QoS optimization and network traffic analysis. Contributions, feedback, and citations are greatly appreciated!
