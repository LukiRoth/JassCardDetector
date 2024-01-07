import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from Utils.model_definition import ResNet34
from Utils.helper import *
import matplotlib.pyplot as plt 

MIN_WIDTH = 50  # minimum width of the contour to be considered a card
MIN_HEIGHT = 70  # minimum height of the contour to be considered a card

def calculate_rotation_angle(approx):
    # Extract points from the approx contour
    pt1, pt2, pt3, pt4 = approx.squeeze()

    # Calculate distances between each pair of points
    distances = [np.linalg.norm(pt1-pt2), np.linalg.norm(pt2-pt3), 
                 np.linalg.norm(pt3-pt4), np.linalg.norm(pt4-pt1)]

    # Find the longest side
    longest_side_index = np.argmax(distances)
    ptA, ptB = approx.squeeze()[longest_side_index], approx.squeeze()[(longest_side_index + 1) % 4]

    # Calculate the angle of the longest side with respect to the vertical axis
    angle = np.arctan2(ptB[1] - ptA[1], ptB[0] - ptA[0])
    angle_degrees = np.degrees(angle)

    # Adjust the angle to ensure the card is upright
    if angle_degrees < -45:
        rotation_angle = -(90 + angle_degrees)
    elif angle_degrees > 45:
        rotation_angle = 90 - angle_degrees
    else:
        rotation_angle = -angle_degrees

    return rotation_angle


def rotate_image(image, angle):
    # Rotate the image by the given angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

card_mapping = create_card_mapping()

# Load your trained ResNet34 model
model = torch.load('Models\TrainedModels\jass_card_classifier_model_35.pth',map_location=torch.device('cpu'))
model.eval()

# Define the transformation needed for the ResNet34 input
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize directly to 224x224, maintaining aspect ratio
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize video capture
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale just for thresholding and contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for contour in contours:
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Check if the contour has 4 vertices (potential card)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            # Process only if contour meets size thresholds
            if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                # Extract the ROI using the original RGB frame
                roi = frame[y:y+h, x:x+w]
                # Convert to RGB if your camera captures in BGR
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                # Determine card orientation and calculate rotation angle
                rotation_angle = calculate_rotation_angle(approx)

                # Rotate the ROI
                roi_rotated = rotate_image(roi, rotation_angle)

                # Convert rotated ROI to PIL image and transform
                roi_pil = Image.fromarray(roi_rotated)
                roi_tensor = transform(roi_pil).unsqueeze(0)


                # Unnormalize the tensor
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                roi_unnorm = torch.squeeze(roi_tensor)  # Remove batch dimension
                roi_unnorm = roi_unnorm * torch.tensor(std[:, None, None]) + torch.tensor(mean[:, None, None])
                roi_unnorm = roi_unnorm.numpy()

                # Transpose from (C, H, W) to (H, W, C)
                roi_unnorm = np.transpose(roi_unnorm, (1, 2, 0))

                # Plot using Matplotlib
                plt.imshow(roi_unnorm)
                plt.title("ROI Tensor Visualized")
                plt.show()

                # Make a prediction
                with torch.no_grad():
                    prediction = model(roi_tensor)
                max_index = torch.argmax(prediction).item()

                # Get the corresponding card label
                card_label = card_mapping[max_index]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, str(card_label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Card Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()