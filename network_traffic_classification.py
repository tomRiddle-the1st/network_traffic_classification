import os
import json
from pickle import TRUE
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import bandwidth
import seaborn as sns

from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler

data_folder = "data" 

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

charts_folder = "charts"
os.makedirs(charts_folder, exist_ok=True) 

processed_data = os.path.join(output_folder, "Mirage_flows.csv")  
Model = os.path.join(output_folder, "model.pkl")                 
scaler_file = os.path.join(output_folder, "feature_scaler.pkl")   
lda_file = os.path.join(output_folder, "lda_transformer.pkl")     

#toggle
use_SMOTE = False
use_random_oversampling = True 
use_lda = True #dimension reduction
top_features = 10

#-------------------------------------------- QoS Policy Mapping----------------------------------------------------------
qos_policies = {
    'waze': {'priority': 'high', 'bandwidth': 'medium', 'latency_sensitivity': 'high', 'jitter_tolerance': 'low'},
    
    'accuweather': {'priority': 'medium', 'bandwidth': 'low', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'high'},
    
    'duolingo': {'priority': 'low', 'bandwidth': 'low', 'latency_sensitivity': 'low', 'jitter_tolerance': 'high'},
    
    'subito': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'high'},
    #contextlogic
    'wish': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'high'},
    'groupon': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'high'},
    
    'spotify': {'priority': 'high', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'medium'},
    #joelapenna/foursquare
    'foursquare': {'priority': 'medium', 'bandwidth': 'low', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'high'},
    
    'youtube': {'priority': 'medium', 'bandwidth': 'high', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'medium'},
    
    'twitter': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'high'},
    #facebook.katana
   'facebook': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'high'},
    #facebook.orca
    'messenger': {'priority': 'high', 'bandwidth': 'low', 'latency_sensitivity': 'high', 'jitter_tolerance': 'low'},  
    'pinterest': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'low', 'jitter_tolerance': 'high'},
    #iconology/comics
    'comics': {'priority': 'low', 'bandwidth': 'medium', 'latency_sensitivity': 'low', 'jitter_tolerance': 'high'},
    
    'dropbox': {'priority': 'low', 'bandwidth': 'variable', 'latency_sensitivity': 'low', 'jitter_tolerance': 'high'},
    
    'tripadvisor': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'high'},
    
    'slither': {'priority': 'high', 'bandwidth': 'medium', 'latency_sensitivity': 'very_high', 'jitter_tolerance': 'very_low'},
    
    'viber': {'priority': 'high', 'bandwidth': 'medium', 'latency_sensitivity': 'high', 'jitter_tolerance': 'low'},
    
    'trello': {'priority': 'low', 'bandwidth': 'low', 'latency_sensitivity': 'low', 'jitter_tolerance': 'high'},
    #motain/iliga
    'motain': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'high'},
    
    'unknown': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'medium'},
    'default': {'priority': 'medium', 'bandwidth': 'medium', 'latency_sensitivity': 'medium', 'jitter_tolerance': 'medium'}
}

#-----------------------------Extract application label from dataset-------------------------------------------
def extract_label(filename):
    try:
        name_of_file = os.path.splitext(filename)[0]
        parts = name_of_file.split("_")
        label = None

        for part in parts:
            if "." in part:
                label = part.lower()
                break

        if not label:
            return "unknown"

        subparts = label.split(".")
        prefixes = {"com", "org", "net", "air", "de", "it", "motain", "contextlogic", "joelapenna", "iconology"}
        remaining_parts = [p for p in subparts if p not in prefixes]

        if not remaining_parts:
            return "unknown"

        if "air.com.hypah.io.slither" in label:
            return "slither"
        if "com.contextlogic.wish" in label:
            return "wish"
        if "com.joelapenna.foursquared" in label:
            return "foursquare"
        if "com.facebook.katana" in label:
            return "facebook"
        if "com.facebook.orca" in label:
            return "messenger"
        if "com.iconology.comics" in label:
            return "comics"

        if len(remaining_parts) >= 2:
            if remaining_parts[0] in ["google", "facebook"]:
                if len(remaining_parts) >= 3 and remaining_parts[1] == "android":
                    app_name = remaining_parts[2]
                elif remaining_parts[1] in ["katana", "orca"]:
                    app_name = remaining_parts[1]
                else:
                    app_name = remaining_parts[1]
            else:
                app_name = remaining_parts[0]
        else:
            app_name = remaining_parts[0]

        if app_name in prefixes or app_name == "com":
            return "unknown"

        app_name = app_name.replace(",", "_").replace(" ", "_").lower()

        name_mapping = {
            "katana": "facebook",
            "orca": "messenger",
            "voip": "viber",
            "slither": "slither",
            "iliga": "iliga",
            "tripadvisor": "tripadvisor",
            "comics": "comics",
            "wish": "wish",
            "foursquared": "foursquare"
        }

        return name_mapping.get(app_name, app_name)

    except Exception as e:
        print(f"Error extracting label from {filename}: {e}")
        return "unknown"


#------------------------  Process JSON file and extract flow features ------------------------
def process_file(file_path, label):

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Skipping invalid JSON file {file_path}")
        return pd.DataFrame()
    rows = []
    for flow_id, flow in data.items():
        flow_id_clean = str(flow_id).replace(",", "_").replace(" ", "_")
        label_clean = label.replace(",", "_").replace(" ", "_")
        row = {"flow_id": flow_id_clean, "Label": label_clean}
        
        if 'flow_features' in flow:
            for feat_cat, stats in flow['flow_features'].items():
                for direction, vals in stats.items():
                    for stat_name, val in vals.items():
                        col_name = f"{feat_cat}_{direction}_{stat_name}"
                        row[col_name] = val
        
        rows.append(row)
    return pd.DataFrame(rows)

#-------------------------------------------------save plots-------------------------------------------------------
def save_plot(fname):
    try:
        path = os.path.join(charts_folder, fname)
        plt.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Saved plot: {path}")
    except Exception as e:
        print(f"Failed saving plot {fname}: {e}")


#---------------------- Aggregate all JSON files into Mirage.csv --------------
def aggregate():
    if os.path.exists(processed_data):
        print(f"Using cached {processed_data}")
        return pd.read_csv(processed_data)

    data_files = []
    for file_path in glob(os.path.join(data_folder, "*.json")):
        label = extract_label(os.path.basename(file_path))
        df = process_file(file_path, label)
        data_files.append(df)
        print(f"Processed {label} with {len(df)} flows")

    if not data_files:
        raise ValueError("No JSON files found in data folder")

    data = pd.concat(data_files, ignore_index=True)
    
    str_cols = data.select_dtypes(include="object")
    data[str_cols.columns] = str_cols.apply(lambda x: x.astype(str).str.replace(",", "_").str.replace(" ", "_"))

    data.to_csv(processed_data, index=False, sep=",", quotechar='"')
    print(f"Saved aggregate dataset to {processed_data}")
    return data

#----------------------------------------preprocessing -------------------------
def preprocess(data):
    # Separate features and labels
    X = data.drop(columns=["flow_id", "Label"], errors="ignore")
    y = data["Label"]

    # Handle missing values
    X = X.fillna(0)
    
    # Remove features with zero variance (constant features)
    constant_features = X.columns[X.var() == 0]
    if len(constant_features) > 0:
        print(f"Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)

    print(f"\nDataset shape: {X.shape}")
    print(f"Number of apps: {y.nunique()}")
    print(f"App with number of flows:")
    print(y.value_counts())

    # Application Traffic Bar Chart
    plt.figure(figsize=(16, 8))
    flows_table = y.value_counts().reset_index()
    flows_table.columns = ["App", "Flows"]
    flows_table = flows_table.sort_values("Flows", ascending=False)
    
    colors = []
    for app in flows_table["App"]:
        policy = map_to_qos_policy(app)
        if policy['priority'] == 'high':
            colors.append('#0881a3')  #   high priority
        elif policy['priority'] == 'medium':
            colors.append('#ffd6a4')  #  medium priority
        else:
            colors.append('#fde9df')  #   low priority
    
    bars = plt.bar(range(len(flows_table)), flows_table["Flows"], color=colors)
    plt.title("Application Traffic FLows", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Flows")
    plt.xlabel("Applications")
    plt.xticks(range(len(flows_table)), flows_table["App"], rotation=45, ha="right", fontsize=10)
    
    # bar graph legend
    import matplotlib.patches as patches
    high_patch = patches.Patch(color='#0881a3', label='High Priority')
    med_patch = patches.Patch(color='#ffd6a4', label='Medium Priority')
    low_patch = patches.Patch(color='#fde9df', label='Low Priority')
    plt.legend(handles=[high_patch, med_patch, low_patch], loc='upper right')
    plt.tight_layout()
    save_plot("traffic_distribution.png")
    plt.show()

    return X, y

#------------------------ training the model ------------------------------
def train_model(X, y):
    
    start_train  = time.time()
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Save scaler for future use
    joblib.dump(scaler, scaler_file)
    
    # Apply oversampling based on configuration
    oversampling_method = None
    if use_SMOTE and use_random_oversampling:
        print("Warning: Both SMOTE and Random Oversampling enabled. Using Random Oversampling only.")
        oversampling_method = "Random Oversampling"
    elif use_SMOTE:
        oversampling_method = "SMOTE"
    elif use_random_oversampling:
        oversampling_method = "Random Oversampling"
    
    if oversampling_method == "SMOTE":
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        print(f"Applied SMOTE: {X_scaled.shape} ... {X_resampled.shape}")
    elif oversampling_method == "Random Oversampling":
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_scaled, y)
        print(f"Applied Random Oversampling: {X_scaled.shape} ...flows sample to {X_resampled.shape}")
    else:
        X_resampled, y_resampled = X_scaled, y
        print("No oversampling applied")
    
    if oversampling_method:
        print("Class distribution after oversampling:")
        print(pd.Series(y_resampled).value_counts().head(10))
    
    # Apply dimension reduction if enabled
    if use_lda:
        n_components = min(len(y.unique()) - 1, X_resampled.shape[1], 50)
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X_resampled, y_resampled)
        print(f"Applied LDA: {X_resampled.shape} ...features reduce to {X_lda.shape}")
        
        # Convert back to DataFrame for consistency
        X_final = pd.DataFrame(X_lda, columns=[f'LDA_{i+1}' for i in range(X_lda.shape[1])])
        
        # Save LDA transformer
        joblib.dump(lda, lda_file)

    else:
        X_final = X_resampled
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    rf_model.fit(X_train, y_train)
    end_train = time.time()
    total_time = end_train - start_train
    minutes = total_time//60
    seconds = (total_time)-(minutes*60)
    print("Model training completed!")
    print(f"Total training time: {minutes:.2f} minutes and {seconds:.2f} seconds")

    
    return rf_model, X_test, y_test, scaler

#----------------------------------Map application to QoS policy ---------------------
def map_to_qos_policy(app_label):
    # Try exact match first
    if app_label in qos_policies:
        return qos_policies[app_label]
    
    # Try partial matching for  labels
    for policy_key in qos_policies.keys():
        if policy_key in app_label.lower():
            return qos_policies[policy_key]
    
    # Return default policy
    return qos_policies['default']

#---------------------------   Analyze QoS requirements distribution----------------------------
def analyze_qos_requirements(y):
    qos_analysis = {}
    priority_counts = {'high': 0, 'medium': 0, 'low': 0}
    latency_counts = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0}
    bandwidth_counts = {'high': 0, 'medium': 0, 'low': 0, 'variable':0}
    
    for label in y.unique():
        policy = map_to_qos_policy(label)
        qos_analysis[label] = policy
        
        priority_counts[policy['priority']] += y.value_counts()[label]
        latency_counts[policy['latency_sensitivity']] += y.value_counts()[label]
        bandwidth_counts[policy['bandwidth']] += y.value_counts()[label]
    
    return qos_analysis, priority_counts, latency_counts,bandwidth_counts

#---------------------------------Pie charts -----------------------------
def qos_chart_visualizations(y, priority_counts, latency_counts):
    
    # 1. Priority 
    plt.figure(figsize=(10, 8))
    priority_df = pd.DataFrame(list(priority_counts.items()), columns=['Priority', 'Flows'])
    colors_pie = ['#0881a3', '#ffd6a4', '#fde9df']  
    wedges, texts, autotexts = plt.pie(priority_df['Flows'], 
                                       labels=priority_df['Priority'], 
                                       autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(priority_df["Flows"]))} flows)',
                                       colors=colors_pie, 
                                       startangle=90)
    plt.title("Traffic Distribution by QoS Priority", fontsize=14, fontweight='bold')

    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    # Priority legend 
    import matplotlib.patches as patches
    legend_patches = []
    for priority, color in zip(priority_df['Priority'], colors_pie):
        legend_patches.append(patches.Patch(color=color, label=f'{priority.title()} Priority'))
    plt.legend(handles=legend_patches, loc='upper right')
    plt.tight_layout()
    save_plot("priority_distribution.png")
    plt.show()
    
    # 2. Latency 
    plt.figure(figsize=(10, 8))
    latency_df_pie = pd.DataFrame(list(latency_counts.items()), columns=['Latency', 'Flows'])
    latency_df_pie = latency_df_pie[latency_df_pie['Flows'] > 0]  # Filter zero flows
    colors_latency = ['#FF0000','#0881a3', '#ffd6a4', '#fde9df']  
    wedges, texts, autotexts = plt.pie(latency_df_pie['Flows'], 
                                       labels=latency_df_pie['Latency'], 
                                       autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(latency_df_pie["Flows"]))} flows)', 
                                       colors=colors_latency[:len(latency_df_pie)], 
                                       startangle=90)
    plt.title("Traffic Distribution by Latency Sensitivity", fontsize=14, fontweight='bold')

    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    # Latency legend 
    import matplotlib.patches as patches
    legend_patches = []
    for latency, color in zip(latency_df_pie['Latency'], colors_latency[:len(latency_df_pie)]):
        legend_patches.append(patches.Patch(color=color, label=f'{latency.title()} Sensitivity'))
    plt.legend(handles=legend_patches, loc='upper right')
    plt.tight_layout()
    save_plot("latency_sensitivity.png")
    plt.show()

    # 3. Bandwidth 
    plt.figure(figsize=(10, 8))
    bandwidth_counts = {}
    for label in y.unique():
        policy = map_to_qos_policy(label)
        bandwidth = policy['bandwidth']
        bandwidth_counts[bandwidth] = bandwidth_counts.get(bandwidth, 0) + y.value_counts()[label]
    
    bandwidth_df = pd.DataFrame(list(bandwidth_counts.items()), columns=['Bandwidth', 'Flows'])
    colors_bandwidth = ['#ffd6a4', '#0881a3', '#fde9df', '#faffb8']    

    wedges, texts, autotexts = plt.pie(bandwidth_df['Flows'], 
                                       labels=bandwidth_df['Bandwidth'], 
                                       autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(bandwidth_df["Flows"]))} flows)', 
                                       colors=colors_bandwidth[:len(bandwidth_df)], 
                                       startangle=90)
    plt.title("Traffic Bandwidth Requirements", fontsize=14, fontweight='bold')
    
    # to make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    # Bandwidth legend
    import matplotlib.patches as patches
    legend_patches = []
    for bandwidth, color in zip(bandwidth_df['Bandwidth'], colors_bandwidth[:len(bandwidth_df)]):
        legend_patches.append(patches.Patch(color=color, label=f'{bandwidth.title()} Bandwidth'))
    plt.legend(handles=legend_patches, loc='upper right')
    plt.tight_layout()
    save_plot("bandwidth_requirements.png")
    plt.show()
    

#-------------------------------------- evaluation with QoS-focused metrics--------------
def evaluate_model_with_qos(model, X_test, y_test, y):
    print("Model Evaluation")
    
    # Basic predictions
    y_pred = model.predict(X_test)
    
    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    
    print("\n Classification Report:")
    print(report_df.round(4))
    
    # QoS-specific evaluation
    print("\n")
    print("QoS Evaluation")
    
    # Group performance by QoS priority levels
    qos_performance = {'high': [], 'medium': [], 'low': []}
    
    unique_labels = list(set(y_test.unique()) | set(y_pred))
    for label in unique_labels:
        if label in report and 'f1-score' in report[label]:
            policy = map_to_qos_policy(label)
            qos_performance[policy['priority']].append(report[label]['f1-score'])
    
    print("Performance by QoS Priority Level:")
    for priority, scores in qos_performance.items():
        if scores:
            avg_f1 = np.mean(scores)
            print(f"{priority.capitalize()} Priority Apps: F1={avg_f1:.4f} (n={len(scores)} apps)")
        else:
            print(f"{priority.capitalize()} Priority Apps: No apps in test set")
    
    # Confusion Matrix
    print("\n Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="coolwarm", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_plot("confusion_matrix.png")
    plt.show()

    # Top features 
    if use_lda == False:
        if hasattr(model, 'feature_importances_') and len(X_test.columns) == len(model.feature_importances_):
            top_features_list = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False).head(top_features)
            print(f"\nTop {top_features} Most Important Features:")
            for feature_name in top_features_list.index:
                print(feature_name)

     #QoS Policy Recommendations
    print("\nQoS Policy Recommendations for each app")
    
    for label in model.classes_:
        policy = map_to_qos_policy(label)
        if label in report and 'f1-score' in report[label]:
            confidence = report[label]['f1-score']
            print(f"{label}:")
            print(f"  Classification Confidence: {confidence:.3f}")
            print(f"  Recommended Priority: {policy['priority']}")
            print(f"  Bandwidth Allocation: {policy['bandwidth']}")
            print(f"  Latency Sensitivity: {policy['latency_sensitivity']}")
            print(f"  Jitter Tolerance: {policy['jitter_tolerance']}")

 #----------------------------Show QoS analysis, policies, and charts-------------------------------
def qos_analysis_and_charts(y):    
  
    
    # QoS Analysis
    qos_analysis, priority_counts, latency_counts, bandwidth_counts = analyze_qos_requirements(y)
    
    print(f"\nPriority Distribution:")
    total_flows = len(y)
    for priority, count in priority_counts.items():
        percentage = (count / total_flows) * 100
        print(f"{priority:6}: {count:6d} flows ({percentage:5.1f}%)")
    
    print(f"\nLatency Sensitivity Distribution:")
    for latency, count in latency_counts.items():
        percentage = (count / total_flows) * 100
        print(f"{latency:10}: {count:6d} flows ({percentage:5.1f}%)")

    print(f"\nBandwidth Distribution:")
    for bandwidth, count in bandwidth_counts.items():
        percentage = (count / total_flows) * 100
        print(f"{bandwidth:10}: {count:6d} flows ({percentage:5.1f}%)")

    #  pie charts and heatmap
    qos_chart_visualizations(y, priority_counts, latency_counts)
    
    # Group applications by priority for recommendations
    high_priority_apps = []
    medium_priority_apps = []
    low_priority_apps = []
    
    for app in y.unique():
        policy = map_to_qos_policy(app)
        flow_count = y.value_counts()[app]
        app_info = {
            'name': app,
            'flows': flow_count,
            'policy': policy
        }
        
        if policy['priority'] == 'high':
            high_priority_apps.append(app_info)
        elif policy['priority'] == 'medium':
            medium_priority_apps.append(app_info)
        else:
            low_priority_apps.append(app_info)
    
    # flow count (descending)
    high_priority_apps.sort(key=lambda x: x['flows'], reverse=True)
    medium_priority_apps.sort(key=lambda x: x['flows'], reverse=True)
    low_priority_apps.sort(key=lambda x: x['flows'], reverse=True)

# Main 
start_time = time.time()
print("NETWORK TRAFFIC CLASSIFICATION")
try:
    # Step 1: Data aggregation
    print("\nStep 1: Aggregating flow data...")
    data = aggregate()
        
    # Step 2: Preprocessing
    print("\nStep 2: Preprocessing...")
    X, y = preprocess(data)
        
    # Step 3: Model training
    print("\nStep 3: Train model...")
    model, X_test, y_test, scaler = train_model(X, y)
        
    # Step 4: Model evaluation 
    print("\nStep 4: Evaluating model performance...")
    evaluate_model_with_qos(model, X_test, y_test, y)
        
    # Step 5: shows pie charts 
    print("\nStep 5: QoS analysis (with charts)...")
    qos_analysis_and_charts(y)
        
    # Step 6: Save model
    print("Saving Model and Metadata")
        
    model_info = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'classes': model.classes_.tolist(),
        'qos_policies': qos_policies,
        'configuration': {
            'use_SMOTE': use_SMOTE,
            'use_random_oversampling': use_random_oversampling,
            'use_lda': use_lda,
            'top_features': top_features
        }
    }
        
    if use_lda:
        lda_transformer = joblib.load(lda_file)
        model_info['lda_transformer'] = lda_transformer
        
    joblib.dump(model_info, Model)
    print(f"Model and metadata saved to {Model}")
    print(f"Feature scaler saved to {scaler_file}")
        
    end_time = time.time()
    total_time = end_time - start_time
    minutes = total_time//60
    seconds = (total_time)-(minutes*60)
    print("Execution completed successfully")
    print(f"Total execution time: {minutes} minutes and {seconds:.2f} seconds")
    
except Exception as e:
    print(f" Error during execution: {str(e)}")
    raise



