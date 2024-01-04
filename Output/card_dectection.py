import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        #self.resnet = models.resnet34(pretrained=True)
        # Update the model initialization with the new 'weights' parameter
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Replace the last fully connected layer
        # ResNet34 uses 512 for fc layers
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)

num_classes = len(card_mapping)  # Assuming 'card_mapping' is your class mapping

# After initializing your model
model = ResNet34(num_classes).to(device)

# Assuming `CNN` is your model class#
#model = CNN()
model = torch.load('Models\TrainedModels\jass_card_classifier_model_v1.pth',map_location=torch.device('cpu'))
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
