# AEye - Eye Tracking Application

## Overview
AEye is an eye tracking application that utilizes computer vision techniques to estimate gaze direction based on facial landmarks. It leverages the Dlib library for face detection and landmark prediction, and OpenCV for image processing.

## Requirements
- Python 3.x
- OpenCV
- Dlib
- NumPy

You can install the required packages using pip:

```
pip install opencv-python dlib numpy
```

## Usage
1. Clone the repository:
```
git clone https://github.com/yourusername/aeye.git
```

## Setup
1. Download the Dlib shape predictor model from [Dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract it to your project directory.
2. Ensure your camera is connected and accessible.

## Usage
Run the application using the following command:
```
python aeye.py
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request.

## License
This project is licensed under the MIT License.
