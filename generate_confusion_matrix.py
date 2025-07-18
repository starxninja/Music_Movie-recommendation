#!/usr/bin/env python3
"""
Script to generate accurate confusion matrix for the emotion detection model
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os
import seaborn as sns

def load_model():
    """Load the trained emotion detection model"""
    emotion_model = Sequential()
    
    # Build the model architecture (same as in train.py)
    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))
    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(7, activation='softmax'))
    
    # Load the trained weights
    if os.path.exists('model.h5'):
        emotion_model.load_weights('model.h5')
        print("Model loaded successfully!")
        return emotion_model
    else:
        print("Error: model.h5 not found!")
        return None

def evaluate_model():
    """Evaluate the model and generate confusion matrix"""
    print("Loading model...")
    model = load_model()
    
    if model is None:
        return
    
    print("Setting up data generators...")
    # Setup test data generator
    test_dir = 'data/test'
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False  # Important: don't shuffle for evaluation
    )
    
    print(f"Found {test_generator.samples} test samples")
    print(f"Class indices: {test_generator.class_indices}")
    
    # Get class labels in correct order
    class_labels = list(test_generator.class_indices.keys())
    print(f"Class labels: {class_labels}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(test_generator, steps=test_generator.samples // 64 + 1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Ensure we have the same number of predictions and true labels
    min_length = min(len(predicted_classes), len(true_classes))
    predicted_classes = predicted_classes[:min_length]
    true_classes = true_classes[:min_length]
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predicted classes: {len(predicted_classes)}")
    print(f"True classes: {len(true_classes)}")
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Generate confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))
    
    # Create and save confusion matrix plot
    create_confusion_matrix_plot(cm, class_labels)
    
    return cm, accuracy, class_labels

def create_confusion_matrix_plot(cm, class_labels):
    """Create and save a beautiful confusion matrix plot"""
    plt.figure(figsize=(12, 10))
    
    # Create a more detailed confusion matrix plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Number of Predictions'})
    
    plt.title('Emotion Detection Model - Confusion Matrix\n(Actual vs Predicted)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Emotion', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Emotion', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy text
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    plt.text(0.02, 0.98, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Also create a normalized version
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Normalized Predictions'})
    
    plt.title('Emotion Detection Model - Normalized Confusion Matrix\n(Actual vs Predicted)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Emotion', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Emotion', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    print("Normalized confusion matrix saved as 'confusion_matrix_normalized.png'")
    
    plt.show()

def print_detailed_analysis(cm, class_labels):
    """Print detailed analysis of the confusion matrix"""
    print("\n" + "="*60)
    print("DETAILED MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Calculate per-class metrics
    for i, emotion in enumerate(class_labels):
        tp = cm[i, i]  # True positives
        fp = np.sum(cm[:, i]) - tp  # False positives
        fn = np.sum(cm[i, :]) - tp  # False negatives
        tn = np.sum(cm) - tp - fp - fn  # True negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{emotion.upper()}:")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score: {f1_score:.4f}")
        print(f"  Total samples: {np.sum(cm[i, :])}")
        print(f"  Correctly classified: {tp}")
        print(f"  Misclassified: {np.sum(cm[i, :]) - tp}")
    
    # Overall statistics
    print(f"\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total test samples: {np.sum(cm)}")
    print(f"Correctly classified: {np.sum(np.diag(cm))}")
    print(f"Misclassified: {np.sum(cm) - np.sum(np.diag(cm))}")
    print(f"Overall accuracy: {np.sum(np.diag(cm)) / np.sum(cm):.4f} ({np.sum(np.diag(cm)) / np.sum(cm) * 100:.2f}%)")

if __name__ == "__main__":
    print("Starting model evaluation and confusion matrix generation...")
    
    # Check if test data exists
    if not os.path.exists('data/test'):
        print("Error: Test data directory 'data/test' not found!")
        print("Please ensure you have the test dataset in the correct location.")
        exit(1)
    
    # Check if model exists
    if not os.path.exists('model.h5'):
        print("Error: Model file 'model.h5' not found!")
        print("Please ensure you have trained the model first.")
        exit(1)
    
    # Evaluate model and generate confusion matrix
    cm, accuracy, class_labels = evaluate_model()
    
    if cm is not None:
        # Print detailed analysis
        print_detailed_analysis(cm, class_labels)
        
        print(f"\n✅ Confusion matrix generation completed!")
        print(f"📊 Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📁 Files saved:")
        print(f"   - confusion_matrix.png (raw counts)")
        print(f"   - confusion_matrix_normalized.png (normalized)")
    else:
        print("❌ Failed to generate confusion matrix!") 