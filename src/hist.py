import matplotlib.pyplot as plt
import pickle

def plot_training_history(history):
    # Set the style for better aesthetics
    plt.style.use('ggplot')

    # Create a figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    
    # Plotting loss
    axs[0, 0].plot(history['loss'], label='Training Loss', color='blue')
    axs[0, 0].plot(history['val_loss'], label='Validation Loss', color='orange')
    axs[0, 0].set_title('Loss Evolution', fontsize=16)
    axs[0, 0].set_xlabel('Epochs', fontsize=14)
    axs[0, 0].set_ylabel('Loss', fontsize=14)
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Plotting accuracy
    axs[0, 1].plot(history['accuracy'], label='Training Accuracy', color='green')
    axs[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
    axs[0, 1].set_title('Accuracy Evolution', fontsize=16)
    axs[0, 1].set_xlabel('Epochs', fontsize=14)
    axs[0, 1].set_ylabel('Accuracy', fontsize=14)
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Plotting precision
    axs[1, 0].plot(history['precision'], label='Training Precision', color='purple')
    axs[1, 0].plot(history['val_precision'], label='Validation Precision', color='cyan')
    axs[1, 0].set_title('Precision Evolution', fontsize=16)
    axs[1, 0].set_xlabel('Epochs', fontsize=14)
    axs[1, 0].set_ylabel('Precision', fontsize=14)
    axs[1, 0].legend()
    axs[1, 0].grid()

    # Plotting recall
    axs[1, 1].plot(history['recall'], label='Training Recall', color='magenta')
    axs[1, 1].plot(history['val_recall'], label='Validation Recall', color='yellow')
    axs[1, 1].set_title('Recall Evolution', fontsize=16)
    axs[1, 1].set_xlabel('Epochs', fontsize=14)
    axs[1, 1].set_ylabel('Recall', fontsize=14)
    axs[1, 1].legend()
    axs[1, 1].grid()

    # Plotting AUC
    axs[2, 0].plot(history['auc'], label='Training AUC', color='brown')
    axs[2, 0].plot(history['val_auc'], label='Validation AUC', color='olive')
    axs[2, 0].set_title('AUC Evolution', fontsize=16)
    axs[2, 0].set_xlabel('Epochs', fontsize=14)
    axs[2, 0].set_ylabel('AUC', fontsize=14)
    axs[2, 0].legend()
    axs[2, 0].grid()

    # Hide the last subplot (2, 1) since we have only 5 metrics
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig('model/training_history.png')
    plt.show()

# After training, call the function
with open('model/training_history.pkl', 'rb') as f:
    history = pickle.load(f)

plot_training_history(history)
