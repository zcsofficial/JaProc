import cv2
import dlib
import numpy as np
import time
from datetime import datetime
from flask import Flask, Response, render_template, jsonify
import threading

# Initialize Flask app
app = Flask(__name__)

# Initialize webcam and models
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Variables for tracking malpractice
warnings = 0
max_warnings = 10
last_warning_time = 5
cooldown = 5
debounce_count = 0
debounce_threshold = 5
exam_active = True
latest_message = "All good"

# Log file
log_file = open("malpractice_log.txt", "a")

def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"{timestamp} - {message}\n")
    log_file.flush()

def get_gaze_direction(landmarks):
    left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
    right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)
    eye_center = (left_eye + right_eye) / 2
    nose = (landmarks.part(30).x, landmarks.part(30).y)
    if abs(eye_center[1] - nose[1]) > 20:
        return "Looking away"
    return "Looking at screen"

def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    suspicious_objects = ["cell phone", "book"]
    detected = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and classes[class_id] in suspicious_objects:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detected.append((classes[class_id], (x, y, w, h)))
    return detected

def detect_malpractice(frame):
    global warnings, last_warning_time, debounce_count, exam_active, latest_message
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(50, 50))
    current_time = time.time()
    message = "All good"
    malpractice_detected = False

    if len(faces) == 0:
        malpractice_detected = True
        message = "No face detected!"
    elif len(faces) > 1:
        malpractice_detected = True
        message = "Multiple faces detected!"
    else:
        dlib_faces = detector(gray)
        if len(dlib_faces) > 0:
            landmarks = predictor(gray, dlib_faces[0])
            gaze = get_gaze_direction(landmarks)
            if gaze == "Looking away":
                malpractice_detected = True
                message = "Looking away from screen!"

    objects = detect_objects(frame)
    if objects:
        malpractice_detected = True
        message = f"Suspicious object detected: {', '.join([obj[0] for obj in objects])}"

    if malpractice_detected:
        debounce_count += 1
        if debounce_count >= debounce_threshold and (current_time - last_warning_time > cooldown):
            warnings += 1
            last_warning_time = current_time
            print(f"Warning {warnings}/{max_warnings}: {message}")
            log_event(f"Warning {warnings}: {message}")
            debounce_count = 0
            if warnings >= max_warnings:
                exam_active = False
                message = "Max warnings reached. Logging out..."
                log_event("User logged out due to max warnings")
    else:
        debounce_count = 0

    latest_message = message
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for _, (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Suspicious Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, f"Warnings: {warnings}/{max_warnings}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def generate_frames():
    while exam_active:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_malpractice(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    log_file.close()

@app.route('/')
def index():
    global exam_active
    if not exam_active:
        return render_template('logout.html')
    return render_template('exam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({'warnings': warnings, 'max_warnings': max_warnings, 'message': latest_message, 'exam_active': exam_active})

if __name__ == "__main__":
    # Create templates folder and HTML files
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')

    with open('templates/exam.html', 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Online Exam</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; }
                #video { border: 2px solid #000; }
                #status { margin-top: 20px; font-size: 20px; }
                #instructions { margin: 20px; text-align: left; display: inline-block; }
            </style>
        </head>
        <body>
            <h1>Online Exam</h1>
            <div id="instructions">
                <h3>Instructions</h3>
                <ul>
                    <li>Keep your face visible in the camera.</li>
                    <li>Do not look away from the screen.</li>
                    <li>No additional people or suspicious objects allowed.</li>
                    <li>3 warnings result in automatic logout.</li>
                </ul>
            </div>
            <img id="video" src="{{ url_for('video_feed') }}" width="640" height="480">
            <div id="status"></div>
            <script>
                function updateStatus() {
                    fetch('/status')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('status').innerHTML = 
                                `Warnings: ${data.warnings}/${data.max_warnings} | ${data.message}`;
                            if (!data.exam_active) {
                                window.location.reload();
                            }
                        });
                }
                setInterval(updateStatus, 1000);
            </script>
        </body>
        </html>
        ''')

    with open('templates/logout.html', 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Exam Ended</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                h1 { color: #ff0000; }
            </style>
        </head>
        <body>
            <h1>Exam Session Ended</h1>
            <p>You have been logged out due to exceeding the maximum number of warnings.</p>
        </body>
        </html>
        ''')

    # Run Flask app
    app.run(debug=True, threaded=True)