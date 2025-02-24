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
import hashlib
from functools import wraps
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# MySQL Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',  # Replace with your MySQL username
    'password': 'Adnan@66202',  # Replace with your MySQL password
    'database': 'jazz'  # Create this database first
}

# Model and webcam initialization
cap = None
face_cascade = None
detector = dlib.get_frontal_face_detector()
predictor = None
yolo_model = None

# Malpractice tracking
warnings = 0
max_warnings = 10
last_warning_time = 0
cooldown = 5
debounce_count = 0
debounce_threshold = 5
exam_active = False
latest_message = "All good"
recording = False
out = None
lock = threading.Lock()
current_session_id = None

# Directory setup
os.makedirs("recordings", exist_ok=True)

# Database connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def log_event(message, session_id):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO logs (session_id, message, timestamp) VALUES (%s, %s, %s)", 
                       (session_id, message, timestamp))
        conn.commit()
        cursor.close()
        conn.close()

def initialize_models():
    global cap, face_cascade, predictor, yolo_model
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade")
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        raise RuntimeError(f"{predictor_path} not found")
    predictor = dlib.shape_predictor(predictor_path)
    yolo_model = YOLO("yolov8n.pt")

def get_eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def get_gaze_direction(landmarks):
    left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
    left_ear = get_eye_aspect_ratio(left_eye_pts)
    right_ear = get_eye_aspect_ratio(right_eye_pts)
    nose = (landmarks.part(30).x, landmarks.part(30).y)
    
    if left_ear < 0.2 or right_ear < 0.2:
        return "Eyes closed"
    eye_vec = np.mean(right_eye_pts, axis=0) - np.mean(left_eye_pts, axis=0)
    gaze_angle = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))
    if abs(gaze_angle) > 30 or abs(np.mean(left_eye_pts, axis=0)[1] - nose[1]) > 20:
        return "Looking away"
    return "Looking at screen"

def get_head_pose(landmarks):
    nose = (landmarks.part(30).x, landmarks.part(30).y)
    left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
    right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)
    eye_mid = (left_eye + right_eye) / 2
    angle = math.degrees(math.atan2(nose[1] - eye_mid[1], nose[0] - eye_mid[0]))
    return "Head turned away" if abs(angle) > 45 else "Head forward"

def detect_objects(frame, exam_name):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT suspicious_objects FROM exams WHERE name = %s", (exam_name,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        suspicious_objects = result[0].split(",") if result else ["cell phone", "book", "calculator", "laptop"]
    else:
        suspicious_objects = ["cell phone", "book", "calculator", "laptop"]
    
    results = yolo_model(frame, conf=0.5)
    detected = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            if label in suspicious_objects:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected.append((label, (x1, y1, x2 - x1, y2 - y1)))
    return detected

def detect_malpractice(frame, exam_name):
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
        if dlib_faces:
            landmarks = predictor(gray, dlib_faces[0])
            gaze = get_gaze_direction(landmarks)
            head_pose = get_head_pose(landmarks)
            if gaze in ["Looking away", "Eyes closed"]:
                malpractice_detected = True
                message = gaze
            elif head_pose != "Head forward":
                malpractice_detected = True
                message = head_pose

    objects = detect_objects(frame, exam_name)
    if objects:
        malpractice_detected = True
        message = f"Suspicious object: {', '.join([obj[0] for obj in objects])}"

    with lock:
        if malpractice_detected:
            debounce_count += 1
            if debounce_count >= debounce_threshold and (current_time - last_warning_time > cooldown):
                warnings += 1
                last_warning_time = current_time
                log_event(f"Warning {warnings}: {message}", current_session_id)
                update_session_warnings(warnings)
                debounce_count = 0
                if warnings >= max_warnings:
                    exam_active = False
                    message = "Max warnings reached. Logging out..."
                    log_event("User logged out due to max warnings", current_session_id)
                    update_session_status("terminated")
        else:
            debounce_count = 0
        latest_message = message
    return frame

def update_session_warnings(warnings):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE exam_sessions SET warnings = %s WHERE id = %s", (warnings, current_session_id))
        conn.commit()
        cursor.close()
        conn.close()

def update_session_status(status):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE exam_sessions SET status = %s, end_time = %s WHERE id = %s", 
                       (status, datetime.now(), current_session_id))
        conn.commit()
        cursor.close()
        conn.close()

def record_session(exam_name):
    global out, recording, exam_active, current_session_id
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"recordings/{exam_name}_{date_str}.avi"
    audio_filename = f"recordings/{exam_name}_{date_str}.wav"
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    audio_frames = []
    
    log_event(f"Started recording for {exam_name}", current_session_id)
    frame_count = 0
    while recording and exam_active:
        ret, frame = cap.read()
        if not ret:
            log_event("Camera access lost", current_session_id)
            break
        frame_count += 1
        if frame_count % 5 == 0:
            frame = detect_malpractice(frame, exam_name)
        out.write(frame)
        audio_data = stream.read(1024, exception_on_overflow=False)
        audio_frames.append(audio_data)
        audio_level = np.frombuffer(audio_data, dtype=np.int16).max()
        if audio_level > 10000:
            log_event("Suspicious audio detected", current_session_id)
    
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
    log_event(f"Recording saved: {video_filename}, {audio_filename}", current_session_id)
    if not exam_active:
        update_session_status("terminated" if warnings >= max_warnings else "completed")

def generate_frames():
    while exam_active:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_malpractice(frame, session.get('exam_name', 'Unknown'))
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = %s AND password = %s", (username, password))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            if user:
                session['username'] = username
                session['user_id'] = user[0]
                try:
                    initialize_models()
                    return redirect(url_for('select_exam'))
                except RuntimeError as e:
                    return str(e)
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
                conn.commit()
                cursor.close()
                conn.close()
                return redirect(url_for('login'))
            except Error as e:
                cursor.close()
                conn.close()
                return "Username already exists"
    return render_template('register.html')

@app.route('/select_exam', methods=['GET', 'POST'])
@login_required
def select_exam():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM exams")
        exams = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
    else:
        exams = ["Math 101", "Physics 102", "Chemistry 103"]
    
    if request.method == 'POST':
        session['exam_name'] = request.form['exam']
        return redirect(url_for('instructions'))
    return render_template('select_exam.html', exams=exams)

@app.route('/instructions')
@login_required
def instructions():
    return render_template('instructions.html', exam_name=session['exam_name'])

@app.route('/pre_check')
@login_required
def pre_check():
    return render_template('pre_check.html')

@app.route('/start_exam', methods=['POST'])
@login_required
def start_exam():
    global exam_active, warnings, recording, current_session_id
    exam_name = session.get('exam_name', 'Unknown')
    user_id = session.get('user_id')
    
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM exams WHERE name = %s", (exam_name,))
        exam_id = cursor.fetchone()[0]
        cursor.execute("INSERT INTO exam_sessions (user_id, exam_id) VALUES (%s, %s)", (user_id, exam_id))
        conn.commit()
        current_session_id = cursor.lastrowid
        cursor.close()
        conn.close()
    
    with lock:
        exam_active = True
        warnings = 0
        recording = True
    threading.Thread(target=record_session, args=(exam_name,), daemon=True).start()
    return redirect(url_for('exam'))

@app.route('/exam')
@login_required
def exam():
    if not exam_active:
        return redirect(url_for('logout'))
    return render_template('exam.html', exam_name=session['exam_name'])

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
@login_required
def status():
    with lock:
        return jsonify({'warnings': warnings, 'max_warnings': max_warnings, 'message': latest_message, 'exam_active': exam_active})

@app.route('/logout')
def logout():
    global exam_active, recording
    with lock:
        exam_active = False
        recording = False
    if cap is not None:
        cap.release()
    if current_session_id:
        update_session_status("completed")
    session.pop('username', None)
    session.pop('exam_name', None)
    session.pop('user_id', None)
    return render_template('logout.html')

if __name__ == "__main__":
    try:
        app.run(debug=True, threaded=True)
    finally:
        if cap is not None:
            cap.release()