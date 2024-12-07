import csv
import mindwave
import time
import datetime
import os
from flask import Flask, render_template, send_file,send_from_directory, request
from flask_socketio import SocketIO
import threading

app = Flask(__name__)
socketio = SocketIO(app)

app = Flask(__name__)
socketio = SocketIO(app)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/live', methods=['GET', 'POST'])
# def live():
#     return render_template('live.html')

# Function to handle inputs and setup session details
def setup_session():
    global roll_number, student_name, session_name, filename

    # Prompt for student details
    roll_number = input("Enter Roll Number: ")
    student_name = input("Enter Student Name: ")
    print("Hi, give me the name of the recording session, for example, person's name. Timestamp will be added automatically.")
    session_name = input('Session name: ')

    # Generate filename for individual session data
    ts = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')  # Replace colons with hyphens
    filename = f'{session_name}_{ts}.csv'

    print(f"Writing session data to {filename}")

# Function to record EEG data
def record_eeg_data():
    # Connect to Mindwave
    print("Connecting to Mindwave...")
    headset = mindwave.Headset('COM4')
    print(headset.status)
    print("Connected, waiting 10 seconds for data to start streaming...")
    time.sleep(10)

    print("Starting to record, automatically recording a 5-minute slice so keep on working...")

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Raw', 'Attention', 'Meditation', 'delta', 'theta', 'low-alpha',
                         'high-alpha', 'low-beta', 'high-beta', 'low-gamma', 'mid-gamma', 'Attention Level'])

        # Initialize counters for attention levels
        low_attention_count = 0
        medium_attention_count = 0
        high_attention_count = 0

        iteration_time = 0.5  # Each iteration represents 0.5 seconds
        total_iterations = int(30 / iteration_time)  # Total iterations for 2 minutes

        for _ in range(total_iterations):
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # Calculate Attention Level
            if headset.attention <= 30:
                attention_level = 0
                low_attention_count += 1
            elif 31 <= headset.attention <= 70:
                attention_level = 1
                medium_attention_count += 1
            else:
                attention_level = 2
                high_attention_count += 1

            eeg_data = {
                "timestamp": ts,
                "raw_value": headset.raw_value,
                "attention": headset.attention,
                "attention_level": attention_level,
                "meditation": headset.meditation,
                "waves": headset.waves
            }
            socketio.emit('eeg_data', eeg_data)

            # Continue saving to CSV as before
            values = list(headset.waves.values())
            values = [ts, headset.raw_value, headset.attention, headset.meditation] + values + [attention_level]
            writer.writerow(values)

            time.sleep(iteration_time)

    # Calculate total time for each attention level
    low_attention_time = low_attention_count * iteration_time
    medium_attention_time = medium_attention_count * iteration_time
    high_attention_time = high_attention_count * iteration_time

    # Send session summary to the webpage
    summary = {
        "low_attention_time": low_attention_time,
        "medium_attention_time": medium_attention_time,
        "high_attention_time": high_attention_time,
        "total_time": low_attention_time + medium_attention_time + high_attention_time,
    }
    socketio.emit('session_summary', summary)

    # Append attention times to the class summary file
    class_filename = './custom/class_attention_times.csv'
    file_exists = os.path.isfile(class_filename)

    with open(class_filename, 'a', newline='') as class_file:
        class_writer = csv.writer(class_file)
        if not file_exists:
            class_writer.writerow(['Roll Number', 'Student Name', 'High Attention Time',
                                   'Medium Attention Time', 'Low Attention Time'])
        class_writer.writerow([roll_number, student_name, high_attention_time,
                               medium_attention_time, low_attention_time])

    print(f"\nAttention times for {student_name} (Roll Number: {roll_number}) added to {class_filename}.")


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
def index():
    return render_template('index.html')
# Routes
@app.route('/live', methods=['GET', 'POST'])
def live():
    return render_template('live.html')

@app.route('/download-report')
def download_report():
    try:
        csv_path = './custom/class_attention_times.csv'
        return send_file(csv_path, as_attachment=True, download_name='Evaluation_Report.csv')
    except Exception as e:
        print(f"Error while downloading the file: {e}")
        return "Failed to download the report. pls try again later", 500

# Run Flask server in a separate thread
def run_flask():
    socketio.run(app, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    # Setup the session first
    setup_session()

    # Start the Flask server in a background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start recording EEG data
    record_eeg_data()


