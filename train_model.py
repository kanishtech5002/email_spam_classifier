import os
import urllib.request
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Ignore warnings for clean beginner-friendly console output
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import our custom preprocessing function
from preprocess import preprocess_text

def main():
    print("="*60)
    print("Email Spam Classifier: Model Training & Evaluation Pipeline")
    print("="*60)

    # Define absolute paths based on the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "spam.csv")
    images_dir = os.path.join(base_dir, "static", "images")
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")

    # 1. Dataset Download & Loading
    dataset_url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

    if not os.path.exists(dataset_path):
        print(f"Dataset '{dataset_path}' not found. Downloading...")
        try:
            # Download dataset using urllib
            urllib.request.urlretrieve(dataset_url, dataset_path)
            print("Download completed successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return
    else:
        print(f"Dataset '{dataset_path}' already exists. Skipping download.")

    # Load dataset with latin-1 encoding as it contains special characters
    print("\n[Step 1] Loading dataset...")
    df = pd.read_csv(dataset_path, encoding='latin-1')
    
    # Clean dataset columns (SMS Spam Collection raw csv contains extra blank columns)
    df = df.iloc[:, [0, 1]] # Keep only the first two columns
    df.columns = ['label', 'message']
    
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())

    # Check for missing/NULL values
    print("\n[Step 2] Checking for missing (NULL) values...")
    null_counts = df.isnull().sum()
    print("Null values per column:")
    print(null_counts)
    
    # Drop rows with null values if present
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"Found {total_nulls} null values. Dropping them to clean dataset...")
        df = df.dropna()
        print(f"New dataset shape: {df.shape}")
    else:
        print("No null values found. Proceeding with clean dataset.")

    # Map labels: ham -> 0, spam -> 1 for standardized ML processing
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # 2. Preprocessing
    print("\n[Step 3] Preprocessing text data (lowercase, remove punctuation, numbers, stopwords, and stemming)...")
    # Apply our modular preprocessing function to the entire text
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Ensure there are no empty messages after preprocessing (e.g. messages containing only numbers/punctuation)
    empty_messages_count = (df['processed_message'] == '').sum()
    if empty_messages_count > 0:
        print(f"Found {empty_messages_count} empty messages after preprocessing. Removing them...")
        df = df[df['processed_message'] != '']
        print(f"Dataset shape after removing empty processed messages: {df.shape}")

    # Create static/images directory for visualizations if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)

    # 3. Visualizations Part 1: Dataset Exploration
    print("\n[Step 4] Generating dataset visualizations...")
    
    # A. Spam vs Ham count graph
    plt.figure(figsize=(6, 4))
    # Use distinct colors: Green for Ham, Red/Coral for Spam
    sns.countplot(x='label', data=df, hue='label', palette=['#4CAF50', '#F44336'], legend=False)
    plt.title('Spam vs Ham Count Distribution', fontsize=12, pad=15)
    plt.xlabel('Category (0 = Ham, 1 = Spam)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Ham', 'Spam'])
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'spam_vs_ham_count.png'), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(images_dir, 'spam_vs_ham_count.png')}")

    # B. Word Cloud for Spam Messages
    spam_text = ' '.join(df[df['label'] == 1]['processed_message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds', max_words=100).generate(spam_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Spam Messages', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'wordcloud_spam.png'), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(images_dir, 'wordcloud_spam.png')}")

    # C. Top Spam Keywords Bar Chart
    all_spam_words = []
    for msg in df[df['label'] == 1]['processed_message']:
        all_spam_words.extend(msg.split())
    
    common_spam = Counter(all_spam_words).most_common(15)
    if common_spam:
        words, counts = zip(*common_spam)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(counts), y=list(words), hue=list(words), palette='Reds_r', legend=False)
        plt.title('Top 15 Most Common Keywords in Spam Messages', fontsize=12, pad=15)
        plt.xlabel('Frequency')
        plt.ylabel('Keywords')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'top_spam_keywords.png'), dpi=300)
        plt.close()
        print(f"Saved: {os.path.join(images_dir, 'top_spam_keywords.png')}")

    # 4. Strict Featurization Ordering: Train/Test Split BEFORE Vectorizer fitting
    print("\n[Step 5] Performing Train/Test Split...")
    X = df['processed_message']
    y = df['label']

    # 80/20 train-test split with fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    print("\n[Step 6] Fitting TF-IDF Vectorizer on training data...")
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # FIT and TRANSFORM only on training data
    X_train_vec = vectorizer.fit_transform(X_train)
    # TRANSFORM test data using parameters learned from training data (prevents data leakage)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Model Training & Comparison
    print("\n[Step 7] Training and evaluating Machine Learning models...")
    
    # Multinomial Naive Bayes is our production model
    nb_model = MultinomialNB()
    
    models = {
        "Multinomial Naive Bayes": nb_model,
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }

    results = {}
    nb_confusion_matrix = None
    nb_accuracy = 0.0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_vec)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm
        }
        
        # Capture Naive Bayes specifically
        if name == "Multinomial Naive Bayes":
            nb_accuracy = acc
            nb_confusion_matrix = cm

    print("\n" + "="*50)
    print(f"Production Model Selected: Multinomial Naive Bayes")
    print(f"Accuracy: {nb_accuracy:.4f}")
    print("="*50)

    # 6. Visualizations Part 2: Model Comparison & Results
    print("\n[Step 8] Generating model comparison and evaluation plots...")
    
    # D. Accuracy Comparison Graph
    model_names_list = list(results.keys())
    accuracies_list = [results[m]["accuracy"] for m in model_names_list]
    
    plt.figure(figsize=(8, 5))
    bars = sns.barplot(x=model_names_list, y=accuracies_list, hue=model_names_list, palette='viridis', legend=False)
    plt.title('Model Accuracy Comparison', fontsize=12, pad=15)
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.ylim(0.8, 1.05) # Focus on high range
    
    # Label accuracy values on top of bars
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.005, f"{height:.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'model_accuracy_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(images_dir, 'model_accuracy_comparison.png')}")

    # E. Confusion Matrix Heatmap for the production model (Multinomial Naive Bayes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(nb_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix (Multinomial Naive Bayes)', fontsize=12, pad=15)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(images_dir, 'confusion_matrix.png')}")

    # 7. Save Multinomial Naive Bayes Model & TF-IDF Vectorizer using Pickle
    print("\n[Step 9] Saving Multinomial Naive Bayes model and TF-IDF vectorizer...")
    
    # Save the Naive Bayes model
    with open(model_path, "wb") as f:
        pickle.dump(nb_model, f)
    print(f"Saved Multinomial Naive Bayes model to '{model_path}'")
    
    # Save the TF-IDF vectorizer
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved TF-IDF vectorizer to '{vectorizer_path}'")

    print("\nTraining and pipeline execution completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
