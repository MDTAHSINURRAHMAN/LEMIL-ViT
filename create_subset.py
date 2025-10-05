import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import jensenshannon
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Constants
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

TRAIN_MIN_IMAGES_PER_CLASS = 40
VAL_MIN_IMAGES_PER_CLASS = 8
PREVALENCE_TOLERANCE = 0.20  # ±20% relative tolerance

def load_data():
    # Load the main data file
    df = pd.read_csv('data/chestXray14/Data_Entry_2017.csv')
    # Load test set patient list
    with open('data/chestXray14/test_list.txt', 'r') as f:
        test_images = set(line.strip() for line in f)
    
    # Split into test and non-test
    df['is_test'] = df['Image Index'].isin(test_images)
    test_df = df[df['is_test']].copy()
    nontest_df = df[~df['is_test']].copy()
    
    return df, test_df, nontest_df

def get_patient_label_sets(df):
    """Get unique label sets per patient."""
    patient_labels = defaultdict(set)
    for _, row in df.iterrows():
        labels = row['Finding Labels'].split('|')
        if 'No Finding' in labels:
            labels.remove('No Finding')
        patient_labels[row['Patient ID']].update(labels)
    return patient_labels

def compute_class_prevalences(df):
    """Compute per-class prevalences at the image level."""
    total_images = len(df)
    prevalences = {}
    for disease in DISEASE_CLASSES:
        count = df['Finding Labels'].str.contains(disease).sum()
        prevalences[disease] = count / total_images
    return prevalences

def check_constraints(subset_df, source_prevalences, min_images):
    """Check if subset meets prevalence and minimum image constraints."""
    violations = []
    subset_prevalences = compute_class_prevalences(subset_df)
    
    for disease in DISEASE_CLASSES:
        p_source = source_prevalences[disease]
        p_subset = subset_prevalences[disease]
        count = subset_df['Finding Labels'].str.contains(disease).sum()
        
        # Check minimum images constraint
        if count < min_images:
            violations.append(f"{disease}: Only {count} images (min {min_images} required)")
        
        # Check prevalence tolerance
        rel_diff = abs(p_subset - p_source) / p_source
        if rel_diff > PREVALENCE_TOLERANCE:
            violations.append(f"{disease}: Prevalence drift {rel_diff:.2%} exceeds ±{PREVALENCE_TOLERANCE:.0%}")
    
    return violations, subset_prevalences

def compute_cooccurrence_matrix(df, mlb):
    """Compute disease co-occurrence matrix."""
    label_lists = df['Finding Labels'].str.split('|').apply(lambda x: [l for l in x if l != 'No Finding'])
    label_matrix = mlb.fit_transform(label_lists)
    return (label_matrix.T @ label_matrix) / len(df)

def compute_js_divergence(source_df, subset_df):
    """Compute Jensen-Shannon divergence between source and subset co-occurrences."""
    mlb = MultiLabelBinarizer(classes=DISEASE_CLASSES)
    source_matrix = compute_cooccurrence_matrix(source_df, mlb)
    subset_matrix = compute_cooccurrence_matrix(subset_df, mlb)
    
    # Compute JS divergence for each pair of distributions
    js_div = 0
    n_pairs = 0
    for i in range(len(DISEASE_CLASSES)):
        for j in range(i+1, len(DISEASE_CLASSES)):
            js = jensenshannon(source_matrix[i], subset_matrix[i])
            if not np.isnan(js):
                js_div += js
                n_pairs += 1
    
    return js_div / n_pairs if n_pairs > 0 else 0

def iterative_stratified_sampling(df, patient_labels, target_size, source_prevalences, min_images_per_class):
    """Perform iterative stratified sampling at patient level."""
    all_patients = list(patient_labels.keys())
    selected_patients = set()
    current_images = 0
    
    # Create a priority queue of diseases based on rarity
    disease_priority = sorted(
        DISEASE_CLASSES,
        key=lambda d: (source_prevalences[d], -df['Finding Labels'].str.contains(d).sum())
    )
    
    # First pass: ensure minimum representation for each disease
    for disease in disease_priority:
        if len(selected_patients) == 0:
            # Start with patients that have the rarest disease
            candidates = [p for p in all_patients if disease in patient_labels[p]]
            if candidates:
                patient = random.choice(candidates)
                selected_patients.add(patient)
                current_images += len(df[df['Patient ID'] == patient])
        
        disease_images = sum(
            len(df[df['Patient ID'] == p])
            for p in selected_patients
            if disease in patient_labels[p]
        )
        
        while disease_images < min_images_per_class and current_images < target_size:
            # Find patients with this disease that aren't selected yet
            candidates = [p for p in all_patients 
                        if p not in selected_patients and disease in patient_labels[p]]
            
            if not candidates:
                break
                
            # Choose the patient that helps balance other diseases too
            best_patient = None
            best_score = float('-inf')
            
            for patient in candidates:
                patient_diseases = patient_labels[patient]
                score = 0
                # Reward having under-represented diseases
                for d in patient_diseases:
                    d_images = sum(
                        len(df[df['Patient ID'] == p])
                        for p in selected_patients
                        if d in patient_labels[p]
                    )
                    if d_images < min_images_per_class:
                        score += 1
                    
                if score > best_score:
                    best_score = score
                    best_patient = patient
            
            if best_patient:
                selected_patients.add(best_patient)
                patient_image_count = len(df[df['Patient ID'] == best_patient])
                current_images += patient_image_count
                disease_images += patient_image_count
            else:
                break
    
    # Second pass: fill up to target size while maintaining balance
    while current_images < target_size:
        best_patient = None
        best_score = float('-inf')
        
        subset_df = df[df['Patient ID'].isin(selected_patients)]
        current_prevalences = compute_class_prevalences(subset_df)
        
        for patient in all_patients:
            if patient in selected_patients:
                continue
            
            # Calculate how this patient would affect class balance
            patient_diseases = patient_labels[patient]
            score = 0
            
            # For each disease this patient has
            for disease in patient_diseases:
                target_prev = source_prevalences[disease]
                current_prev = current_prevalences.get(disease, 0)
                
                # Reward moving closer to target prevalence
                if current_prev < target_prev:
                    score += 1
                elif current_prev > target_prev * (1 + PREVALENCE_TOLERANCE):
                    score -= 1
            
            if score > best_score:
                best_score = score
                best_patient = patient
        
        if best_patient is None or best_score < 0:
            break
            
        selected_patients.add(best_patient)
        current_images += len(df[df['Patient ID'] == best_patient])
    
    return df[df['Patient ID'].isin(selected_patients)]

def main():
    print("Loading data...")
    full_df, test_df, nontest_df = load_data()
    
    # Compute source prevalences from non-test pool
    source_prevalences = compute_class_prevalences(nontest_df)
    patient_labels = get_patient_label_sets(nontest_df)
    
    print("\nCreating validation set (~10-12% of non-test patients)...")
    val_target = int(len(nontest_df) * 0.11)  # Target ~11% of images
    val_df = iterative_stratified_sampling(
        nontest_df, 
        patient_labels,
        val_target,
        source_prevalences,
        VAL_MIN_IMAGES_PER_CLASS
    )
    
    print("Creating training set...")
    remaining_df = nontest_df[~nontest_df['Patient ID'].isin(val_df['Patient ID'])]
    remaining_patient_labels = {k: v for k, v in patient_labels.items() 
                              if k not in set(val_df['Patient ID'])}
    
    train_df = iterative_stratified_sampling(
        remaining_df,
        remaining_patient_labels,
        3750,  # Target ~3.75k images
        source_prevalences,
        TRAIN_MIN_IMAGES_PER_CLASS
    )
    
    # Save the splits
    required_columns = ['Image Index', 'Patient ID', 'Finding Labels']
    
    test_df[required_columns].to_csv('test_official.csv', index=False)
    val_df[required_columns].to_csv('val_small.csv', index=False)
    train_df[required_columns].to_csv('train_small.csv', index=False)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Test set: {len(test_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Training set: {len(train_df)} images")
    
    print("\nClass distribution (count, prevalence, relative difference):")
    print("Disease      | Train (n={})\t| Val (n={})\t| Source".format(
        len(train_df), len(val_df)))
    print("-" * 80)
    
    for disease in DISEASE_CLASSES:
        train_count = train_df['Finding Labels'].str.contains(disease).sum()
        val_count = val_df['Finding Labels'].str.contains(disease).sum()
        source_prev = source_prevalences[disease]
        
        train_prev = train_count / len(train_df)
        val_prev = val_count / len(val_df)
        
        train_diff = (train_prev - source_prev) / source_prev
        val_diff = (val_prev - source_prev) / source_prev
        
        print(f"{disease:12} | {train_count:4d} ({train_prev:.3f}) [{train_diff:+.1%}] | "
              f"{val_count:4d} ({val_prev:.3f}) [{val_diff:+.1%}]")
    
    # Compute co-occurrence drift
    train_js = compute_js_divergence(nontest_df, train_df)
    val_js = compute_js_divergence(nontest_df, val_df)
    print(f"\nCo-occurrence drift (JS-divergence):")
    print(f"Training set: {train_js:.4f}")
    print(f"Validation set: {val_js:.4f}")

if __name__ == '__main__':
    main()
