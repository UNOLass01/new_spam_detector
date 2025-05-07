import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('model/final_spam_detection_model', custom_objects={
    'KerasLayer': hub.KerasLayer
})

# Prediction function
def predict_spam(text_samples):
    """
    Predict whether a message is spam (1) or not (0).
    Args:
        text_samples: List or Series of strings
    Returns:
        Numpy array of predictions (0 or 1)
    """
    predictions = model.predict(text_samples, batch_size=32)
    return (predictions > 0.5).astype(int)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["HAM", "SPAM"], columns=["HAM", "SPAM"])
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Display classification report
def display_classification_report(y_true, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["HAM", "SPAM"]))

# Main evaluation using CSV dataset
if __name__ == "__main__":
    # Load dataset from CSV
    df = pd.read_csv("data/combined_data.csv")
    
    # Basic check
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    
    # Drop missing values
    df = df.dropna(subset=['text', 'label'])

    # Extract inputs and labels
    texts = df['text'].astype(str)
    true_labels = df['label'].astype(int).values

    # Predict
    predictions = predict_spam(texts)

    # Print a sample of predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(texts))):
        print(f"Message: {texts.iloc[i]}\nPrediction: {'SPAM' if predictions[i][0] == 1 else 'HAM'}\n")

    # Evaluate
    print("\nModel Evaluation:")
    plot_confusion_matrix(true_labels, predictions)
    display_classification_report(true_labels, predictions)
