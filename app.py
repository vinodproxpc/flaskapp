from flask import Flask, request, jsonify, Response, render_template, flash, redirect, url_for
import requests
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from PIL import Image
import io
import logging
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'test1234nkksdvkksv'
db = SQLAlchemy(app)

# Camera Model
class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    rtsp_url = db.Column(db.String(300), nullable=False)

AI_SERVER_URL = "http://10.10.18.232:5053"

# Add logging configuration at the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    """ Home page list of all camera"""
    cameras = Camera.query.all()
    return render_template('index.html', cameras=cameras)


@app.route('/add', methods=['GET', 'POST'])
def add_camera():
    """Add camera"""
    if request.method == 'POST':
        name = request.form['name']
        rtsp_url = request.form['rtsp_url']
        new_camera = Camera(name=name, rtsp_url=rtsp_url)
        db.session.add(new_camera)
        db.session.commit()
        flash('Camera Added Successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('add_camera.html')


@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_camera(id):
    """Edit camera """
    camera = Camera.query.get_or_404(id)
    if request.method == 'POST':
        camera.name = request.form['name']
        camera.rtsp_url = request.form['rtsp_url']
        db.session.commit()
        flash('Camera Updated Successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('edit_camera.html', camera=camera)


@app.route('/delete/<int:id>')
def delete_camera(id):
    """Delete Camera"""
    camera = Camera.query.get_or_404(id)
    db.session.delete(camera)
    db.session.commit()
    flash('Camera Deleted Successfully!', 'danger')
    return redirect(url_for('index'))

@app.route('/health', methods=['GET'])
def health():
    response = requests.get(f"{AI_SERVER_URL}/health")
    return jsonify(response.json())

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if request.method == 'GET':
        response = requests.get(f"{AI_SERVER_URL}/setup")
    else:
        response = requests.post(f"{AI_SERVER_URL}/setup", json=request.json)
    return jsonify(response.json())

@app.route('/predict', methods=['POST'])
def predict():
    response = requests.post(f"{AI_SERVER_URL}/predict", json=request.json)
    return jsonify(response.json())

@app.route('/webhook', methods=['POST'])
def webhook():
    response = requests.post(f"{AI_SERVER_URL}/webhook", json=request.json)
    return jsonify(response.json())

@app.route('/image_exists', methods=['POST'])
def image_exists():
    response = requests.post(f"{AI_SERVER_URL}/image_exists", json=request.json)
    return jsonify(response.json())

def generate_frames(rtsp_url):
    # Configure RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    
    # Set buffer size
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    # Set video codec
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    
    # Configure RTSP transport - using environment variable
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

    frame_count = 0
    error_count = 0
    max_errors = 5  # Maximum consecutive errors before reconnecting

    while True:
        try:
            success, frame = cap.read()
            
            if not success:
                error_count += 1
                logger.warning(f"Failed to read frame. Error count: {error_count}")
                
                if error_count >= max_errors:
                    logger.info("Attempting to reconnect to RTSP stream...")
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_url)
                    error_count = 0
                continue
            
            # Reset error count on successful frame read
            error_count = 0
            frame_count += 1

            # Process every 3rd frame to reduce load
            if frame_count % 3 != 0:
                continue

            # Convert frame to bytes
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Create files dictionary with image bytes
            files = {
                'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')
            }
            
            # Send frame to AI server for prediction with timeout
            response = requests.post(
                f"{AI_SERVER_URL}/predict", 
                files=files, 
                timeout=1.0  # 1 second timeout
            )
            
            if response.status_code == 200:
                predictions = response.json()
                
                # Draw predictions on frame
                for pred in predictions.get('predictions', []):
                    bbox = pred.get('bbox', [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        label = pred.get('label', 'unknown')
                        score = pred.get('score', 0)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add label and confidence
                        text = f"{label}: {score:.2f}"
                        cv2.putText(frame, text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  (0, 255, 0), 2)

            # Convert frame to JPEG format for streaming
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except requests.exceptions.Timeout:
            logger.warning("AI server prediction timeout")
            continue
            
        except Exception as e:
            logger.error(f"Error in frame processing: {str(e)}")
            continue
    
    cap.release()

# Update the video_feed route to include error handling
# @app.route('/video_feed')
# def video_feed():
#     rtsp_url = request.args.get('rtsp_url')
#     if not rtsp_url:
#         return "Error: No RTSP URL provided", 400
        
#     return Response(
#         generate_frames(rtsp_url),
#         mimetype='multipart/x-mixed-replace; boundary=frame'
#     )

# Video Feed Route
@app.route('/video_feed/<int:id>')
def video_feed(id):
    camera = Camera.query.get_or_404(id)
    return Response(generate_frames(camera.rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/multistream')
def multistream():
    return render_template('index.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)