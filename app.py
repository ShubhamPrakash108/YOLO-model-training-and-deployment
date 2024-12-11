import torch
import gradio as gr
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO('best.pt')

def predict_image(image):
    """
    Perform object detection on the input image
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        tuple: Annotated image, results dataframe, precision-recall table
    """
    # Perform prediction
    results = model(image)[0]
    
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    
    # Prepare results for display
    detection_results = []
    
    # Draw bounding boxes and collect detection info
    for box in results.boxes:
        # Extract box coordinates
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        
        # Get class and confidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put text on the image
        label = f'{class_name} {conf:.2f}'
        cv2.putText(annotated_image, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Collect detection details
        detection_results.append({
            'Class': class_name,
            'Confidence': f'{conf:.4f}',
            'Bounding Box': f'[{x1},{y1},{x2},{y2}]'
        })
    
    # Create DataFrame for detections
    detections_df = pd.DataFrame(detection_results)
    
    # Generate precision-recall metrics (placeholder - you'll need to replace with actual metrics)
    pr_metrics = generate_pr_metrics()
    
    return annotated_image, detections_df, pr_metrics

def generate_pr_metrics():
    """
    Generate placeholder precision and recall metrics
    
    Note: Replace this with actual calculations from your training results
    """
    # This is a placeholder - you should replace with actual metrics from your training
    metrics_data = {
        'Class': model.names.values(),
        'Precision': [0.85, 0.90, 0.88],  # Example values
        'Recall': [0.80, 0.85, 0.82]      # Example values
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df

# Create Gradio interface
def launch_interface():
    # Input image upload
    image_input = gr.Image(type="numpy", label="Upload Image")
    
    # Outputs
    annotated_image_output = gr.Image(label="Annotated Image")
    detections_output = gr.Dataframe(label="Detection Results")
    metrics_output = gr.Dataframe(label="Precision-Recall Metrics")
    
    # Gradio interface
    demo = gr.Interface(
        fn=predict_image,
        inputs=image_input,
        outputs=[
            annotated_image_output, 
            detections_output, 
            metrics_output
        ],
        title="YOLO Object Detection",
        description="Upload an image for object detection"
    )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = launch_interface()
    demo.launch()