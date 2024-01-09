import torch
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def create_card_mapping_files():
    suits = ['E', 'H', 'S', 'K']
    values = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    mapping = {}
    class_id = 0
    for suit in suits:
        for value in values:
            mapping[class_id] = f'{suit}_{value}'
            class_id += 1
    return mapping


def create_card_mapping():
    suits = ['E', 'H', 'S', 'K']
    values = ['A', 'K', 'D', 'B', '10', '9', '8', '7', '6']
    mapping = {}
    class_id = 0
    for suit in suits:
        for value in values:
            mapping[class_id] = f'{suit}_{value}'
            class_id += 1
    return mapping


card_mapping = create_card_mapping()

def preprocess(frame, input_size=(224, 224)):
    # Define the transformation steps
    transform = transforms.Compose([
        transforms.Resize(input_size),       # Resize to the input size expected by the model
        transforms.ToTensor(),               # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization (adjust if needed)
                             std=[0.229, 0.224, 0.225])
    ])

    # Convert the frame to PIL image for processing
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply the transformations
    processed_frame = transform(pil_image)

    # Add an extra batch dimension since PyTorch treats all inputs as batches
    processed_frame = processed_frame.unsqueeze(0)

    return processed_frame

def postprocess(predictions):
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    _, predicted_class = torch.max(probabilities, 1)
    return predicted_class, probabilities


def display(frame, card_info, class_mapping):
    predicted_class, probabilities = card_info

    # Convert class_mapping keys to integers if they are strings
    if isinstance(next(iter(class_mapping)), str):
        class_mapping = {int(k.split('_')[1]): v for k, v in class_mapping.items()}

    for i, class_idx in enumerate(predicted_class):
        class_idx_item = class_idx.item()
        # Accessing the probability of the predicted class
        probability = probabilities[i, class_idx_item].item()

        if class_idx_item not in class_mapping:
            print(f"Warning: Class index {class_idx_item} not found in class_mapping.")
            continue

        class_name = class_mapping[class_idx_item]

        text = f"{class_name}: {probability:.2f}"
        cv2.putText(frame, text, (10, 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def plot_confusion_matrix(all_targets_np, all_predictions_np, card_mapping):
    """
    Plot the confusion matrix for model predictions.
    
    Args:
        all_targets_np: Numpy array of target values.
        all_predictions_np: Numpy array of predicted values.
        card_mapping: Dictionary mapping card names to their numerical labels.
    """
    cm = confusion_matrix(all_targets_np, all_predictions_np)
    df_cm = pd.DataFrame(cm, index=[i for i in card_mapping.keys()], columns=[i for i in card_mapping.keys()])
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_training_statistics(epochs, training_losses, validation_losses, accuracies):
    """
    Plot the training loss, validation loss, and accuracy over epochs.
    
    Args:
        epochs: Range of epochs.
        training_losses: List of training losses per epoch.
        validation_losses: List of validation losses per epoch.
        accuracies: List of accuracies per epoch.
    """
    plt.figure(figsize=(10, 3))

    # Plot Training Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, training_losses, 'r-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Validation Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, validation_losses, 'b-', label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, accuracies, 'g-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


