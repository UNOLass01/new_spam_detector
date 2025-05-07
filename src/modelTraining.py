import pandas as pd
import re
import nltk
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow_hub import KerasLayer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# ==================== 1. Load and Prepare Dataset ====================
def load_data():
    df = pd.read_csv("./data/emails.csv")
    df.drop_duplicates(subset="text", inplace=True)
    return df

# ==================== 2. Enhanced Text Preprocessing ====================
def clean_text(text):
    # Lowercase for uncased BERT
    text = text.lower()

    # Replace URLs and emails with tags
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ==================== 3. Build Enhanced BERT Model ====================
def create_enhanced_model():
    # Using smaller BERT for efficiency (can upgrade to larger if needed)
    bert_preprocessor = KerasLayer(
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', 
        name='preprocessing'
    )
    bert_encoder = KerasLayer(
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2',
        trainable=True, 
        name='BERT_encoder'
    )
    
    text_input = Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocessor(text_input)
    outputs = bert_encoder(preprocessed_text)

    # Enhanced classification head
    x = Dropout(0.3)(outputs['pooled_output'])
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=text_input, outputs=output)

    optimizer = Adam(learning_rate=2e-5, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

# ==================== 4. Enhanced Training Process ====================
def train_model(model, X_train, y_train, X_test, y_test):
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    callbacks = [
        ModelCheckpoint(
            'model/best_spam_model',
            monitor='val_precision',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_precision',
            patience=5,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=30,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save('model/spam_detection_model')
    return history

# ==================== 5. Main Execution ====================
def main():
    print("Loading and preprocessing data...")
    df = load_data()
    df["clean_text"] = df["text"].apply(clean_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], 
        test_size=0.15, 
        stratify=df["label"], 
        random_state=42
    )

    # Create and train model
    print("Building model...")
    model = create_enhanced_model()
    model.summary()
    #plot_model(model, to_file='model/model_architecture.png', show_shapes=True)

    print("Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Save final model in specified format
    print("Saving model...")
    model.save('model/final_spam_detection_model')  # SavedModel format
    
    # Save training history
    with open('model/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print("Training complete. Model saved to 'model/final_spam_detection_model'")

if __name__ == "__main__":
    main()