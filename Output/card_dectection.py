import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

def create_card_mapping():
    suits = ['E', 'H', 'S', 'K']
    values = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    mapping = {}
    class_id = 0
    for suit in suits:
        for value in values:
            # Use integer class_id as the key instead of the string
            mapping[class_id] = f'{suit}_{value}'
            class_id += 1
    return mapping

card_mapping = create_card_mapping()

print(card_mapping)
print("Number of classes:", len(card_mapping))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

        # After conv1 and pool: 64x64 -> 32x32
        # After conv2 and pool: 32x32 -> 16x16
        # After conv3 and pool: 16x16 -> 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, len(card_mapping))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def preprocess(frame, input_size=(64, 64)):
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


# Assuming `CNN` is your model class
model = CNN()
model = torch.load('Models\TrainedModels\jass_card_classifier_model_20.pth',map_location=torch.device('cpu'))
model.eval()

cap = cv2.VideoCapture(1)  # '0' is typically the default camera

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    # Preprocess the frame for your model
    processed_frame = preprocess(frame)  # Implement this function

    # Perform inference (ensure the tensor is on the same device as the model)
    with torch.no_grad():
        predictions = model(processed_frame)


    # Post-process the predictions to extract card information
    # Implement postprocess function based on your model's output
    card_info = postprocess(predictions)  

    # Display the results on the frame
    display(frame, card_info, card_mapping)

    # Display the frame
    cv2.imshow('Card Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
