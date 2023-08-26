from flask import Flask, Response, render_template, request
import cv2
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to get frames from webcam
def get_frame():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = model(frame)  # Perform object detection
            frame_with_boxes = results.render()[0]
            ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
