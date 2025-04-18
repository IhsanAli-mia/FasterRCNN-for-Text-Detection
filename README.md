# Faster R-CNN for Text Detection

This project implements Faster R-CNN for detecting text regions in natural scene images. It adapts the powerful object detection capabilities of Faster R-CNN to locate and highlight areas containing textual content, which can then be passed on to OCR engines for text recognition.
## Overview

Text detection in images is a crucial step in many computer vision applications, such as:

    Scene text recognition

    Automated document processing

    License plate recognition

    Assistive technology

## This project:

    Uses a modified Faster R-CNN architecture tailored for detecting text-like objects

    Trains and evaluates on datasets such as ICDAR, SynthText, or custom datasets

    Outputs bounding boxes around detected text areas

    Works for both axis aligned and general oriented bounding boxes

## Features

    Deep learning-based text region detection

    Easy-to-modify configuration for different datasets

    High accuracy in complex scene images

    Visualizations of detection results

    Compatible with common OCR pipelines

## Architecture

The model architecture is based on:

    Backbone: ResNet-50/101 or VGG16 (configurable)

    Region Proposal Network (RPN): Proposes candidate text regions

    ROI Pooling + Classifier: Detects text/non-text and regresses bounding boxes

Input Image → CNN Backbone → RPN → ROI Pooling → Text Classification & Box Regression

## Getting Started
### Prerequisites

    Python 3.7+

    PyTorch

    torchvision

    OpenCV

    matplotlib

### Install dependencies:

pip install -r requirements.txt

### Dataset

Prepare your dataset and configure paths in config.py.
Training

python train.py --config config.yaml

### Inference

python infer.py

## Project Structure

.
├── datasets/
├── models/
├── utils/
├── train.py
├── detect.py
├── evaluate.py
├── config.yaml
├── requirements.txt
└── README.md
