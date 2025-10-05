import cv2
from deepface import DeepFace
import sys
import os
import time

# --- CONFIGURATION (Based on successful testing) ---

# The camera index that successfully showed your video feed (likely 1)
CAMERA_INDEX = 1 

# Dummy folder required by DeepFace
DB_PATH = "./face_database"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Load the fastest face detector (Haar Cascade)
try:
    # Uses the fast Haar Cascade XML file included with OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    print("🔴 FATAL ERROR: Could not load Haar Cascade file. Check OpenCV installation.")
    sys.exit()

# --- CAMERA INITIALIZATION ---
cap = cv2.VideoCapture(CAMERA_INDEX) 

# Force stable settings for maximum compatibility
# These settings resolved previous issues on your system
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


if not cap.isOpened():
    print(f"🔴 FATAL ERROR: Cannot open video source at index {CAMERA_INDEX}. Check camera availability.")
    sys.exit()

print("✅ Real-Time Emotion Tracker Active (Hybrid System)")
print("Press 'q' in the video window to quit.")

# --- LIVE ANALYSIS LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. FAST DETECTION (Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # 2. ACCURATE EMOTION ANALYSIS (DeepFace)
        try:
            # We skip DeepFace's internal, slow detector and give it the face region (ROI)
            analysis = DeepFace.analyze(
                face_roi, 
                actions=['emotion'], 
                detector_backend='skip', 
                enforce_detection=False 
            )

            if analysis and len(analysis) > 0:
                dominant_emotion = analysis[0]['dominant_emotion']
                emotion_score = analysis[0]['emotion'][dominant_emotion]
                
                # Draw the bounding box (Blue)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Draw label text (Green for high contrast)
                cv2.putText(
                    frame, 
                    f"{dominant_emotion} ({emotion_score:.1f}%)",
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (0, 255, 0), 
                    2
                )
        
        except Exception:
            # Show red box if DeepFace analysis fails inside a detected face area
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Analyzing...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            pass

    else:
        # Message when no face is detected
        cv2.putText(frame, "Waiting for face...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Emotion Tracking for Devpost', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("\nEmotion tracking session ended. Resources released.")