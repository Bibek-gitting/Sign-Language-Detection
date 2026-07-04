# 🌟 Sign-Language-Detection

[![GitHub Stars](https://img.shields.io/github/stars/Bibek-gitting/Sign-Language-Detection?style=social)](https://github.com/Bibek-gitting/Sign-Language-Detection/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Bibek-gitting/Sign-Language-Detection?style=social)](https://github.com/Bibek-gitting/Sign-Language-Detection/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [🌟 Sign-Language-Detection](#-sign-language-detection)
  - [Table of Contents](#table-of-contents)
  - [✨ Project Description](#-project-description)
  - [🚀 Key Features & Benefits](#-key-features--benefits)
  - [🛠️ Technologies Used](#-technologies-used)
  - [⚙️ Prerequisites & Dependencies](#-prerequisites--dependencies)
  - [Installation & Setup Instructions](#installation--setup-instructions)
  - [💡 Usage](#-usage)
    - [Real-time Sign Language Detection](#real-time-sign-language-detection)
    - [Building the Dataset](#building-the-dataset)
    - [Testing Data Capture](#testing-data-capture)
  - [📂 Project Structure](#-project-structure)
  - [🔧 Configuration Options](#-configuration-options)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

## ✨ Project Description
This project aims to create a robust and user-friendly Sign Language Detection system using computer vision and machine learning. It translates dynamic hand gestures captured from a webcam into text and spoken alphabets, serving as a communication aid. The system includes modules for building a custom dataset of gestures, training a deep learning model (likely an LSTM-based network), and deploying it for real-time recognition with speech output.

## 🚀 Key Features & Benefits
*   **Real-time Gesture Recognition**: Detects and interprets sign language gestures live from a webcam feed with high accuracy.
*   **Speech Output Integration**: Converts recognized gestures into spoken words using a text-to-speech by (`Windows built-in speech API: System.Speech)`, enhancing accessibility.
*   **Customizable Dataset Builder**: Provides a script (`build_dataset.py`) to easily create and expand a personalized dataset of sign language gestures.
*   **Deep Learning Powered**: Leverages a Keras/TensorFlow-based artificial neural network (ANN/MLP) for gesture classification and recognition.
*   **Modular Architecture**: Designed with separate components for landmark extraction, dataset creation, model training (implied by `display.py` loading), and real-time inference.
*   **Accessibility Enhancement**: Aims to bridge communication gaps for the deaf and hard-of-hearing community.

## 🛠️ Technologies Used
### Languages
*   **Python**

### Frameworks & Libraries
*   **MediaPipe**: For efficient and accurate real-time hand landmark detection.
*   **TensorFlow / Keras**: For building, training, and deploying deep learning models (specifically, LSTM networks).
*   **OpenCV (`cv2`)**: For camera integration, image processing, and visualization.
*   **NumPy**: For numerical operations and data handling.
*   **System.Speech**: For text-to-speech functionality.

## ⚙️ Prerequisites & Dependencies
Before you can run this project, ensure you have the following installed:

*   **Python**: A compatible version of Python (typically Python 3.10). The exact version might be specified in a `pyversion.py` file if available.
*   **Git**: For cloning the repository.

All required Python libraries are listed in `requirements.txt`.

## Installation & Setup Instructions
Follow these steps to get your development environment set up and run the project:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Bibek-gitting/Sign-Language-Detection.git
    cd Sign-Language-Detection
    ```

2.  **Check Python Version:**
    If a `pyversion.py` file exists, consult it for the recommended Python version.

3.  **Create a Virtual Environment (Recommended):**
    It's good practice to create a virtual environment to manage project dependencies isolation.
    ```bash
    python -m venv venv
    ```

4.  **Activate the Virtual Environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

5.  **Install Required Libraries:**
    Install all the necessary Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## 💡 Usage

### Real-time Sign Language Detection
After setting up the environment and potentially building your dataset (see below), you can run the main application for real-time sign language detection:

```bash
python display.py
```
This script will:
*   Load a pre-trained Keras model (or expect one to be trained).
*   Initialize your webcam.
*   Display the webcam feed, detect hand landmarks, and predict gestures.
*   Provide speech output for recognized gestures.

### Building the Dataset
To train a new model or expand the existing one, you'll need to build a dataset of hand landmark sequences. This process captures data for each defined sign language action.

```bash
python build_dataset.py
```
*   **Important**: This script will guide you through capturing sequences for various actions defined in `features.py`. It requires user interaction to perform gestures in front of the camera. This process can be time-consuming, as indicated in the script, and only needs to be done once per dataset.

### Testing Data Capture
You can use `datatest.py` to test the hand landmark detection and ensure data capture works correctly without necessarily building the full dataset.

```bash
python datatest.py
```
This script will open the webcam and demonstrate the hand detection, potentially showing the capture process for a specific gesture (e.g., 'H' as seen in the snippet).

## 📂 Project Structure

```
├── .gitignore            # Specifies intentionally untracked files to ignore
├── Alphabet/             # Directory containing example images for different letters/gestures (e.g., A.png, B.png)
│   ├── A.png
│   ├── B.png
│   ├── C.png
│   ├── D.png
│   ├── E.png
│   ├── F.png
│   ├── G.png
│   ├── H.png
│   ├── I.png
│   ├── J.png
│   ├── K.png
│   ├── L.png
│   ├── M.png
│   ├── N.png
│   ├── O.png
│   ├── P.png
│   ├── Q.png
│   └── R.png
├── build_dataset.py      # Script responsible for collecting and preprocessing hand landmark data to build the training dataset
├── datatest.py           # Script for testing real-time hand detection and data recording capabilities
├── display.py            # The main application script for real-time sign language detection, model inference, and speech output
├── features.py           # Core utilities for MediaPipe hand detection, landmark extraction, and defining project constants (e.g., DATA_PATH, actions)
└── README.md             # This comprehensive README file
└── (Other potential files: requirements.txt, pyversion.py, model_weights.h5, model_architecture.json, etc.)
```

## 🔧 Configuration Options
Several key parameters can be configured to customize the behavior of the system. These are typically found within `features.py` and `display.py`:

*   **`features.py`**:
    *   `DATA_PATH`: The base directory where captured gesture data (`.npy` files) will be stored.
    *   `actions`: A Python list defining the specific sign language characters the system should recognize (e.g., `[A', 'B', 'Q']`).
    *   `no_sequences`: The number of video sequences to capture for each defined action during dataset building.
    *   `sequence_length`: The number of frames (or landmark sets) that constitute a single sequence for model input.
    *   MediaPipe Hand Detection Parameters: In `mp_hands.Hands` initializations (e.g., in `datatest.py` and `features.py`), you can adjust `model_complexity`, `min_detection_confidence`, and `min_tracking_confidence` for performance vs. accuracy trade-offs.

*   **`display.py`**:
    *   `timeout`: The duration (in seconds) after which the system stops processing if no hand is detected, potentially saving resources.
    *   `threshold`: The confidence score required for a prediction to be considered valid. Predictions below this threshold might be ignored.
    *   `predictions` (deque `maxlen`): Defines the sliding window size for recent predictions, used to achieve stable gesture recognition over multiple frames.

To modify these, you will need to edit the respective Python source files.

## 🤝 Contributing
Contributions are highly welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these guidelines:

1.  **Fork** the repository.
2.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3.  **Implement your changes** and ensure they are well-documented.
4.  **Commit your changes** with clear, descriptive messages.
5.  **Push your branch** to your forked repository.
6.  **Open a Pull Request** to the `main` branch of this repository, describing your changes in detail.

## 📄 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

## 🙏 Acknowledgments
*   **Google MediaPipe**: For providing robust and efficient hand landmark detection solutions.
*   **TensorFlow & Keras**: For the powerful deep learning framework enabling model development.
*   **OpenCV**: For indispensable computer vision functionalities.
*   To the open-source community for inspiring tools and resources.
