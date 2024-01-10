# JassCardDetector
JassCardDetector is a project for recognizing french Jass playing cards using image processing and neural networks. It focuses on edge detection to identify card boundaries and a trained neural network to recognize the card's value and suit.

***Project structure:***

1. **Data**: This folder will contain the datasets you use for training and testing your neural network. It can be further subdivided into:
   - **Raw**: Original, unprocessed images of Jass cards.
   - **Processed**: Images after preprocessing (e.g., converted to grayscale, cropped, etc.)
   - **data_background_remove.py**: Script for removing the background from the raw images.
   - **data_generation.py**: Script for generating augmented data from the images with removed background.
   - **dataset_downloader.py**: Script for downloading the dataset from Kaggle.

2. **Docs**: Documentation for the project.

3. **Models**: This directory should contains the neural network models which were trained on the data in the `Data` folder.

4. **Output**: The Output contains the results of your project, such as images, videos, and log files.

5. **Src** (Source): Contains all the source code of the project.
   - **CardRecognition**: Code specific to the card recognition neural network.
      - **Archiv**: Contains old versions of files
      - **card_classification_train.py**: Script for training the card recognition neural network.
   - **Utils**: Any utility scripts or helper functions.
      - **helper.py**: Helper functions for the project.
      - **model_definition.py**: Contains the model definition of the card recognition neural network.
   - **card_detection_v1.py**: First version of the card detection algorithm. It detects 1 card per image. It takes frames from a video and with the help of the card recognition neural network it detects the card's value and suit.
   - **card_detection_v2.py**: Second version of the card detection algorithm. It detects multiple cards per image. It takes frames from a video first detects the card itself with the opencv contour detection algorithm. It then cuts out the card and with the help of the card recognition neural network it detects the card's value and suit.

6. **Requirements**: A requirements.txt file for listing all the Python dependencies.

7. **License**: A file containing the chosen license for your project.

8. **README.md**: A detailed README file explaining the project's purpose, how to set it up, how to use it, and any other necessary information for users and contributors.
