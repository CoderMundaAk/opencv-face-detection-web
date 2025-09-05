from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import threading
import os
from datetime import datetime

app = Flask(__name__)
camera = cv2.VideoCapture(0)

current_detections = {'faces': 0, 'eyes': 0, 'timestamp': None}
detection_lock = threading.Lock()
os.makedirs('uploads', exist_ok=True)

def generate_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 7)

        face_count = len(faces)
        total_eyes = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            total_eyes += len(eyes)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        with detection_lock:
            current_detections['faces'] = face_count
            current_detections['eyes'] = total_eyes
            current_detections['timestamp'] = datetime.now().isoformat()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    with detection_lock:
        return jsonify(current_detections)

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if not success:
        return jsonify({'success': False, 'error': 'Camera error'})

    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join('uploads', filename)
    cv2.imwrite(filepath, frame)
    return jsonify({'success': True, 'filename': filename})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
