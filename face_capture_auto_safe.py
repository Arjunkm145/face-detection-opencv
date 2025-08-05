
import cv2
import datetime
import os
import time

# Load the DNN face detector
model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_file = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Create subfolder for screenshots
output_dir = "screenshots"
os.makedirs(output_dir, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot open webcam. Please check your camera and try again.")
    input("Press Enter to exit...")
    exit()

# Delay time (in seconds) between automatic captures
capture_delay = 3
last_capture_time = 0

print("✅ Webcam started. Looking for faces...")
print("📸 Photos will be taken automatically every 3 seconds when a face is detected.")
print("❌ Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error reading frame from webcam.")
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            face_detected = True
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    current_time = time.time()
    if face_detected and (current_time - last_capture_time >= capture_delay):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"face_ml_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")
        last_capture_time = current_time

    cv2.imshow("ML Face Detector (Auto Mode)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
