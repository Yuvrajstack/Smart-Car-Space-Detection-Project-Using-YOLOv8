# Smart Car Space Detection Project

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)](https://github.com/ultralytics/ultralytics)

This project demonstrates a smart parking space detection system using the state-of-the-art YOLOv8 object detection model. It can analyze images or video feeds to identify and locate available and occupied parking spaces in real-time.

## Overview

The increasing density of urban environments has made finding parking a significant challenge. This project aims to address this issue by providing an intelligent system that can:

* *Detect parking spaces:* Accurately identify individual parking slots in an image or video frame.
* *Classify space occupancy:* Determine whether each detected parking space is currently occupied or vacant.
* *Enable real-time monitoring:* Process live video streams for continuous parking availability updates.

This technology has the potential to be integrated into various applications, such as:

* *Mobile parking apps:* Guiding users directly to available spots.
* *Digital signage in parking lots:* Displaying real-time availability information.
* *Parking management systems:* Optimizing space utilization and enforcement.

* *Powered by YOLOv8:* Utilizes the latest advancements in object detection for high accuracy and speed.
* *Real-time processing:* Capable of analyzing video streams for dynamic parking updates.
* *Clear visualization:* Provides visual output highlighting detected spaces and their occupancy status.
* *Modular design:* Allows for potential integration with other systems and data sources.
* *Easy to use:* Provides a straightforward setup and execution process.

## Getting Started

### Prerequisites

* *Python 3.8+*
* *pip* (Python package installer)
* *CUDA enabled GPU* (recommended for faster processing, but CPU inference is also possible)
* *Basic knowledge of Python and command-line interface.*

### Installation

1.  *Clone the repository:*
    bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    
2.  *Install the required dependencies:*
    bash
    pip install -r requirements.txt
    
    This will install ultralytics (for YOLOv8) and other necessary libraries like opencv-python.

### Usage

1.  *Prepare your input data:*
    * You can use a single image or a video file for testing.
    * Ensure the parking spaces are clearly visible in your input.

2.  *Run the detection script:*

    * *For image processing:*
        bash
        python detect_spaces.py --source path/to/your/image.jpg
        

    * *For video processing:*
        bash
        python detect_spaces.py --source path/to/your/video.mp4
        

    * *To use your webcam (if configured):*
        bash
        python detect_spaces.py --source 0
        
3.  *View the results:*
    * The script will process the input and display a window showing the detected parking spaces with bounding boxes and occupancy status (e.g., "Vacant", "Occupied").
    * Processed images or videos might also be saved in a runs/detect/ directory.

### Script Arguments

The detect_spaces.py script accepts the following arguments:

* --source: Path to the input image, video file, or camera index (e.g., 0 for default webcam).
* --weights: Path to the YOLOv8 model weights file (default is a pre-trained model). You can specify a custom trained model if you have one.
* --conf: Confidence threshold for object detection (default: 0.5).
* --iou: Intersection over Union (IoU) threshold for non-maximum suppression (default: 0.45).
* --save-txt: Save detection results to a text file (default: False).
* --save-conf: Save object confidence scores in the saved text files (default: False).
* --save-crop: Save the cropped bounding boxes of detected objects (default: False).
* --show: Display the detection results in a window (default: True).
* --save: Save the processed image or video with detections (default: False).
* --name: Name of the experiment directory to save results (default: 'exp').
* --exist-ok: Overwrite existing experiment directories (default: False).

### Demo


https://github.com/user-attachments/assets/58d87664-3c63-46e2-b581-01287a9e528d

