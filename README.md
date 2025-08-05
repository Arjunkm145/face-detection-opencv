# Face Detection Using OpenCV and DNN

This project performs real-time face detection using a Deep Neural Network (DNN) with OpenCV.

## Features
- Real-time webcam feed
- Face detection using a pretrained DNN model
- Press `q` to exit the application

## Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/face-detection-opencv.git
   cd face-detection-opencv
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python
   ```
3. Run the script:
   ```bash
   python face_capture.py
   ```

## Model Files
Place the following two files in the `models` folder:
- `res10_300x300_ssd_iter_140000.caffemodel`
- `deploy.prototxt`

You can download them from the OpenCV GitHub or use the ones provided.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author
Arjun K M
