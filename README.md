# Blood Cell Detection with YOLOv11n

## Overview
This project employs the YOLOv11n object detection architecture to identify and classify blood cells from microscopic images. The system can effectively detect three types of blood cells:

- **Red Blood Cells (RBC)**
- **White Blood Cells (WBC)**
- **Platelets**

A Gradio-based web application is included for easy visualization of results and performance metrics. Link: https://huggingface.co/spaces/shubhamprakash108/yolo

## Features
- Accurate detection and classification of blood cells.
- Utilizes a custom dataset annotated for YOLO compatibility.
- Interactive web interface for image upload and result visualization.
- Performance metrics including precision-recall evaluations.

## Dataset
- **Total Images:** 366  
- **Classes:** Red Blood Cells (RBC), White Blood Cells (WBC), Platelets  
- **Format:** Converted annotations to YOLO-compatible format.

## Requirements
- **Python:** 3.8+

### Libraries:
- PyTorch
- Ultralytics
- OpenCV
- NumPy
- Pandas
- Gradio
- Optional: CUDA-enabled GPU for faster training and inference.

### Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Installation

### Clone the repository:
```bash
git clone https://github.com/your-username/blood-cell-detection.git
cd blood-cell-detection
```

### Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training

### The YOLOv11n model is trained with the following configuration:
- **Model:** YOLOv11n  
- **Epochs:** 100  
- **Batch Size:** 64  
- **Image Size:** 640x640  
- **Device:** CUDA  

### Run the training script:
```python
from ultralytics import YOLO  

model = YOLO("yolo11n.pt")  
model.train(  
    data="dataset/dataset.yaml",  
    epochs=100,  
    batch=64,  
    imgsz=640,  
    device="cuda",  
    name="yolov11_medical",  
    amp=True  
)
```

## Running the Application

### Launch the Gradio-based web application using the provided script:
```bash
python app.py
```

## Key Features of the Application
- **Image Upload:** Upload microscopic images for analysis.
- **Annotated Image:** View images with bounding boxes and class labels.
- **Detection Results:** Tabular display of detection classes, confidence scores, and bounding boxes.
- **Precision-Recall Metrics:** Evaluate the performance of the model.

## Project Structure
```
blood-cell-detection/  
│  
├── dataset/  
│   ├── train/  
│   │   ├── images/  
│   │   └── labels/  
│   └── val/  
│       ├── images/  
│       └── labels/  
│  
├── yolo11n.pt           # Pre-trained model weights  
├── best.pt              # Best trained model weights  
├── app.py               # Gradio application script  
├── requirements.txt     # Python dependencies  
└── README.md            # Project documentation  
```

## Future Improvements
- Expand the dataset to improve generalization.
- Experiment with advanced YOLO variants.
- Implement additional data augmentation techniques.
- Fine-tune hyperparameters for improved accuracy.

