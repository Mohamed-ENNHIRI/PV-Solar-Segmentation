
# PV-Solar-Segmentation: A Deep Learning Model for Detecting Photovoltaic Panels

This project provides a solution for detecting photovoltaic (PV) panels on rooftops using deep learning techniques. Specifically, it employs a U-Net architecture to perform semantic segmentation on rooftop imagery and identify whether a rooftop contains PV panels. This model is designed for high accuracy even in images with varying quality and environmental conditions.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Dataset](#dataset)
- [License](#license)

## Overview

The goal of this project is to provide an easy-to-use tool for detecting PV panels on rooftops using satellite or aerial imagery. The core of the tool is a U-Net model, which is well-suited for segmentation tasks. The model outputs a binary mask indicating the locations of PV panels within the image. The repository includes a script that can process both individual images and entire folders of rooftop imagery.

### Key Features

- **U-Net-based Segmentation Model**: Trained to detect photovoltaic panels on rooftops.
- **Support for Multiple Image Formats**: Supports `.jpg`, `.jpeg`, `.png`, and `.bmp` formats.
- **Batch Processing**: Can process an entire folder of images.
- **Model Inference**: After training, the model can be used to segment new rooftop images.

---

## Installation

### 1. Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/mohamed-ennhiri/PV-Solar-Segmentation.git
cd PV-Solar-Segmentation
```

### 2. Set Up a Virtual Environment (Optional)
It’s recommended to use a virtual environment for managing dependencies:

```bash
# For Linux/macOS
python3 -m venv pvseg_env
source pvseg_env/bin/activate

# For Windows
python -m venv pvseg_env
pvseg_env\Scriptsactivate
```

### 3. Install Dependencies
Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

## Dependencies

This project requires the following Python packages:

- `torch` (for deep learning)
- `torchvision` (for image transformations)
- `PIL` (for image loading and saving)
- `numpy` (for numerical operations)
- `matplotlib` (for visualization)
- `tqdm` (for progress bars)

The required dependencies are listed in the `requirements.txt` file.

---

## File Structure

The project has the following file structure:

```
PV-Solar-Segmentation/
├── data/                    # Contains the training and validation data
│   ├── train/               # Training images
│   ├── val/                 # Validation images
├── src/                     # Contains the source code
│   ├── model.py             # Defines the U-Net model architecture
│   ├── utils.py             # Utility functions (e.g., dataset handling)
│   ├── train.py             # Training script (if you wish to train the model)
│   ├── evaluate.py          # Script for evaluating model performance
├── model.pth                # Trained model weights (if available)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── __init__ .py             # Module initialization
```

- `data/` contains the image datasets for training and validation.
- `src/` contains the source code for training, evaluation, and segmentation.
- `model.pth` is the trained model checkpoint.
- `requirements.txt` lists all necessary Python packages.

---

## Usage

### 1. Prepare the Model for Inference

Once you have trained the model or downloaded a pre-trained model (`model.pth`), you can use the `evaluate.py` script to apply segmentation to images.

### 2. Configure the Script

Modify the `evaluate.py` script to specify the following paths directly in the code:

- **Model Path**: `model_path = "model.pth"`
- **Input Image/Folder Path**: `input_path = "path/to/your/input"` (Can be a single image or folder of images)
- **Output Folder**: `output_folder = "predicted_masks"` (Directory where masks will be saved)
- **Threshold**: `threshold = 0.5` (Threshold for segmentation results)
- **Show Images**: `show = True` (Set to `False` to disable image display)

### 3. Running the Script

Once you have configured the paths and options, run the script:

```bash
python src/segment.py
```

This will process all images in the specified `input_path`, generate segmentation masks, and save them in the `output_folder`.

---

## Model Training

If you'd like to train the model from scratch or fine-tune it on your own dataset, use the `train.py` script. Ensure that your dataset is prepared in the correct format, with images and masks located in the respective directories (`train/` and `val/`).

### Training Example

```bash
python src/train.py --train_data_path path/to/train --val_data_path path/to/val --output_model_path model.pth
```

This will start the training process and save the trained model weights in the specified `output_model_path`.

---

## Evaluation

To evaluate the model's performance on a validation set, you can use the `evaluate.py` script. This script computes metrics such as accuracy, precision, recall, and F1 score based on the predictions of the trained model.

```bash
python src/evaluate.py --model model.pth --val_data_path path/to/val
```

This will evaluate the model on the validation set and print the performance metrics.

---

## Dataset

The dataset used in this project comes from [this source](https://zenodo.org/records/5171712). Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery along with labeled masks to train the U-Net model for detecting photovoltaic (PV) panels on rooftops.

You can download the dataset  and follow the directory structure as shown in the repository.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
The dataset used in this project is from Kaggle and is subject to the licensing terms provided by the dataset owner.

---

Feel free to modify the script and model for your specific use case. If you encounter any issues or have suggestions, open an issue in the repository. Happy coding!
