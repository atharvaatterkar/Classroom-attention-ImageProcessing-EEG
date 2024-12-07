from flask import Flask, render_template, request, redirect, url_for, send_file
import numpy as np
import cv2
from detectfaces import get_faces
from keras.models import load_model
import face_recognition
import time
import pandas as pd
from datetime import datetime

# Flask App Initialization
name = "Attention"
app = Flask(__name__)

@app.route("/Analytics", methods=['GET','POST'])
def Analytics():
    return render_template('analytics.html')

@app.route("/Messages", methods=['GET','POST'])
def Messages():
    return render_template('messages.html')

@app.route("/Settings", methods=['GET','POST'])
def Settings():
    return render_template('setting.html')

@app.route("/", methods=['GET','POST'])
def home():
    return render_template('main.html')
# Routes
@app.route('/live', methods=['GET', 'POST'])
def live():
    return render_template('live.html')

    # if request.method == "POST":
    #     return render_template('live.html')
    # else:
    #     return "chill!!"
    

@app.route('/download-report')
def download_report():
    try:
        csv_path = 'Custom/Evaluation.csv'
        return send_file(csv_path, as_attachment=True, download_name='Evaluation_Report.csv')
    except Exception as e:
        print(f"Error while downloading the file: {e}")
        return "Failed to download the report. pls try again later", 500
    
@app.route('/done', methods=['GET', 'POST'])
def done():
    if request.method == "POST":
        name1 = request.form["name1"]

        # Current Date and Time
        now = datetime.now()
        date = str(now.strftime("%d-%m-%Y %H:%M")).split(' ')[0].replace('-', '/').encode()

        # Load known face encodings and names
        face_data = [
            ("prasad", "images/prasad.jpg"),
            ("atharva", "images/atharva.jpg"),
            ("vaibhav", "images/vaibhav.jpg"),
            ("mrinmayee", "images/mrinmayee.jpg"),
            ("deepali", "images/deepali.jpg"),
        ]

        known_face_encodings = []
        known_face_names = []
        for name, path in face_data:
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)

        # Initialize tracking and attendance data
        t_students = {name: {'focus': 0, 'distract': 0, 'attendance': 0} for name in known_face_names}
        df = pd.read_csv('Custom/Evaluation.csv')

        # Add today's date column if not exists
        today_date = datetime.now().strftime('%d/%m/%Y')
        if today_date not in df.columns:
            df[today_date] = 'Absent'

        # Variables
        face_locations = []
        face_encodings = []
        process_this_frame = True
        attendance = []
        img_rows, img_cols = 48, 48
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        box_color = (255, 245, 152)

        # Load Models
        model = []
        print('Loading Models...')
        for i in range(2):
            m = load_model(f'saved_model/cnn{i}.h5')
            model.append(m)
            print(f'Model {i + 1}/3 loaded')
        m = load_model('saved_model/ensemble.h5')
        model.append(m)
        print('Ensemble model loaded\nLoading complete!')

        # Prediction Function
        def predict(x):
            x_rev = np.flip(x, 1)
            x = x.astype('float32') / 255
            x_rev = x_rev.astype('float32') / 255
            p = np.zeros((1, 14))
            p[:, :7] = model[0].predict(x.reshape(1, 48, 48, 1))
            p[:, 7:] = model[1].predict(x_rev.reshape(1, 48, 48, 1))
            return model[2].predict(p)

        # Tracking states
        t_states = {name: {'focus_start': None, 'distract_start': None} for name in known_face_names}

        # Video Capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.open()

        start_session_time = time.time()

        while True:
            ret, img = cap.read()
            curTime = time.time()

            # Get Faces
            faces = get_faces(img, method='haar')
            for i, (face, x, y, w, h) in enumerate(faces):
                pre = predict(face)
                emotion_index = np.argmax(pre)
                emotion_label = emotion_labels[emotion_index]
                emotion_confidence = int(pre[0, emotion_index] * 100)

                name = ''
                try:
                    small_frame = cv2.resize(img[y-20:y+h+20, x-20:x+w+20], (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = small_frame[:, :, ::-1]

                    if process_this_frame:
                     face_locations = face_recognition.face_locations(small_frame)
                     if face_locations:
                         face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                         for face_encoding in face_encodings:
                             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                             name = "Unknown"

                             if True in matches:
                                 # Use face_distance to find the best match
                                 face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                 best_match_index = np.argmin(face_distances)

                                 # Ensure the match is correct
                                 if matches[best_match_index]:
                                     name = known_face_names[best_match_index]
                                     confidence_score = 1 - face_distances[best_match_index]  # Higher means better match
                                     print(f"Matched {name} with confidence {confidence_score:.2f}")

                                     t_students[name]['attendance'] = 1
                                     if name not in attendance:
                                        attendance.append(name)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")

                # Process focus and distraction times
                if name != "Unknown" and name:
                    if emotion_label in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']:
                        if t_states[name]['focus_start'] is not None:
                            focus_duration = curTime - t_states[name]['focus_start']
                            t_students[name]['focus'] += focus_duration
                            t_states[name]['focus_start'] = None
                        if t_states[name]['distract_start'] is None:
                            t_states[name]['distract_start'] = curTime
                    else:
                        if t_states[name]['distract_start'] is not None:
                            distract_duration = curTime - t_states[name]['distract_start']
                            t_students[name]['distract'] += distract_duration
                            t_states[name]['distract_start'] = None
                        if t_states[name]['focus_start'] is None:
                            t_states[name]['focus_start'] = curTime

                # Draw Bounding Boxes and Labels
                tl = (x, y)
                br = (x + w, y + h)
                coords = (x, y - 2)

                box_color = (0, 0, 255) if emotion_label in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'] else (255, 245, 152)
                txt = f"{name} {emotion_label} [{emotion_confidence}%] | {'Distracted' if emotion_label != 'Neutral' else 'Focused'}"
                img = cv2.rectangle(img, tl, br, box_color, 2)
                cv2.putText(img, txt, coords, font, 0.8, text_color, 1, cv2.LINE_AA)

            # Display
            cv2.imshow('Camera', img)

            # Quit on 'q'
            if cv2.waitKey(20) & 0xFF == ord('q'):
                end_session_time = time.time()
                total_session_time = end_session_time - start_session_time

                for name in attendance:
                    if name in t_students:
                        focus_time = t_students[name]['focus']
                        distract_time = t_students[name]['distract']
                        if pd.isna(df.loc[df['Name'] == name, 't_focused']).any():
                            df.loc[df['Name'] == name, 't_focused'] = 0.0
                        if pd.isna(df.loc[df['Name'] == name, 't_distracted']).any():
                            df.loc[df['Name'] == name, 't_distracted'] = 0.0

                        df.loc[df['Name'] == name, 't_focused'] += focus_time
                        df.loc[df['Name'] == name, 't_distracted'] += distract_time
                        df.loc[df['Name'] == name, 't_total'] = df.loc[df['Name'] == name, 't_focused'] + df.loc[df['Name'] == name, 't_distracted']
                        df.loc[df['Name'] == name, today_date] = 'Present'

                df.loc[~df['Name'].isin(attendance), today_date] = 'Absent'
                df.to_csv('Custom/Evaluation.csv', index=False)
                break

        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for('live'))

if __name__ == '__main__':
    app.run(debug=True)
