import cv2
import dlib
import numpy as np
import time
from datetime import datetime
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify, session
import os
import threading
import secrets
import pyaudio
import wave
from ultralytics import YOLO
import math

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # For session management

# Dummy user database
users = {}
exams = ["Math 101", "Physics 102", "Chemistry 103"]

# Initialize webcam and models
cap = None
face_cascade = None
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 nano model

# Malpractice tracking
warnings = 0
max_warnings = 10
last_warning_time = 3
cooldown = 5
debounce_count = 0
debounce_threshold = 5
exam_active = False
latest_message = "All good"
recording = False
out = None
audio_out = None

# Log file
log_file = open("malpractice_log.txt", "a")

def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"{timestamp} - {message}\n")
    log_file.flush()

def get_gaze_direction(landmarks):
    # Enhanced gaze detection using eye aspect ratio and direction
    left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
    left_center = np.mean(left_eye_pts, axis=0)
    right_center = np.mean(right_eye_pts, axis=0)
    nose = (landmarks.part(30).x, landmarks.part(30).y)
    
    eye_vec = right_center - left_center
    gaze_angle = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))
    if abs(gaze_angle) > 30 or abs(left_center[1] - nose[1]) > 20:
        return "Looking away"
    return "Looking at screen"

def get_head_pose(landmarks):
    # Simple head pose estimation using nose and eye positions
    nose = (landmarks.part(30).x, landmarks.part(30).y)
    left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
    right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)
    
    eye_mid = (left_eye + right_eye) / 2
    angle = math.degrees(math.atan2(nose[1] - eye_mid[1], nose[0] - eye_mid[0]))
    if abs(angle) > 45:  # Head tilted too far
        return "Head turned away"
    return "Head forward"

def detect_objects(frame):
    # Advanced object detection with YOLOv8
    results = yolo_model(frame, conf=0.5)
    suspicious_objects = ["cell phone", "book", "calculator", "laptop"]
    detected = []
    
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            if label in suspicious_objects:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected.append((label, (x1, y1, x2 - x1, y2 - y1)))
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
            head_pose = get_head_pose(landmarks)
            if gaze == "Looking away":
                malpractice_detected = True
                message = "Looking away from screen!"
            elif head_pose != "Head forward":
                malpractice_detected = True
                message = "Head turned away!"

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
    return frame

def record_session(exam_name):
    global out, audio_out, recording, exam_active
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"recordings/{exam_name}_{date_str}.avi"
    audio_filename = f"recordings/{exam_name}_{date_str}.wav"
    
    # Video setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
    
    # Audio setup (with basic anomaly detection)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    audio_frames = []
    
    log_event(f"Started recording for {exam_name}")
    while recording and exam_active:
        # Video recording with malpractice detection
        ret, frame = cap.read()
        if not ret:
            log_event("Camera access lost during recording")
            break
        frame = detect_malpractice(frame)
        out.write(frame)
        
        # Audio recording and basic anomaly check
        audio_data = stream.read(1024, exception_on_overflow=False)
        audio_frames.append(audio_data)
        audio_level = np.frombuffer(audio_data, dtype=np.int16).max()
        if audio_level > 10000:  # Threshold for loud sounds
            log_event("Suspicious audio detected (possible talking)")
    
    # Cleanup
    out.release()
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    log_event(f"Recording saved: {video_filename}, {audio_filename}")

@app.route('/', methods=['GET', 'POST'])
def login():
    global cap, face_cascade
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not cap.isOpened():
                return "Camera not accessible. Please check your device."
            return redirect(url_for('select_exam'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username not in users:
            users[username] = password
            return redirect(url_for('login'))
        return "Username already exists"
    return render_template('register.html')

@app.route('/select_exam', methods=['GET', 'POST'])
def select_exam():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        session['exam_name'] = request.form['exam']
        return redirect(url_for('instructions'))
    return render_template('select_exam.html', exams=exams)

@app.route('/instructions')
def instructions():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('instructions.html', exam_name=session['exam_name'])

@app.route('/pre_check')
def pre_check():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('pre_check.html')

@app.route('/start_exam', methods=['POST'])
def start_exam():
    global exam_active, warnings, recording
    if 'username' not in session:
        return redirect(url_for('login'))
    exam_name = session.get('exam_name', 'Unknown')
    exam_active = True
    warnings = 0
    recording = True
    threading.Thread(target=record_session, args=(exam_name,), daemon=True).start()
    return redirect(url_for('exam'))

@app.route('/exam')
def exam():
    if 'username' not in session or not exam_active:
        return redirect(url_for('logout'))
    return render_template('exam.html', exam_name=session['exam_name'])

@app.route('/status')
def status():
    return jsonify({'warnings': warnings, 'max_warnings': max_warnings, 'message': latest_message, 'exam_active': exam_active})

@app.route('/logout')
def logout():
    global exam_active, recording
    exam_active = False
    recording = False
    if cap is not None:
        cap.release()
    session.pop('username', None)
    session.pop('exam_name', None)
    return render_template('logout.html')

if __name__ == "__main__":
   

    app.run(debug=True, threaded=True)