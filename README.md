# Sentinel Labeler

A web-based labeling tool for **Sentinel-2 satellite imagery**, providing multiple pre-labeling methods (KMeans RGB, Otsu, NDVI-based) and support for **custom AI model inference**.

## Features
- **Polygon labeling** and **mask correction** tools.
- Multiple **pre-labeling** methods:
  - KMeans clustering on RGB
  - Otsu thresholding (grayscale)
  - NDVI-based thresholding (auto & fixed)
- Adjustable image overlay transparency.
- Real-time **progress tracking** for long-running tasks.
- Upload and run your **own trained model**.
- Fully **Flask-based backend** with modular services.

## Technology Stack
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Processing**: NumPy, scikit-learn, Pillow
- **Geospatial**: Sentinel-2 imagery, NDVI computation

## Installation
```bash
git clone https://github.com/alibardesstani/sentinel-labeler.git
cd sentinel-labeler
pip install -r requirements.txt
