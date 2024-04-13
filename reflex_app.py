import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
from threading import Thread
import movementDetection as md
import pyaudio
import numpy as np
import librosa
import wave
from pymediainfo import MediaInfo
import subprocess, json

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ReFlex")

        self.video_source = 0  # Use the default camera (you can change it if needed)

        self.cap = cv2.VideoCapture(self.video_source)

        self.container = ttk.Frame(root, padding=5)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.container)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.width = int(self.cap.get(3))
        self.height = int(self.cap.get(4))

        self.buttons_frame = ttk.Frame(self.container)
        self.buttons_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.start_button = ttk.Button(self.buttons_frame, text="Start", command=self.start_video)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.stop_button = ttk.Button(self.buttons_frame, text="Stop", command=self.stop_video)
        self.stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.upload_button = ttk.Button(self.buttons_frame, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(side=tk.TOP, pady=10)

        self.status_label = ttk.Label(self.buttons_frame, text="Not recording", font=("Helvetica", 12), foreground="red")
        self.status_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.buttons_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.is_recording = False
        self.current_segment = []
        
        #self.start_time = ttk.Label(self.buttons_frame, text="Athlete started at:", font=("Helvetica", 12))
        #self.start_time.pack(side=tk.LEFT, padx=75, pady=10)
        
        #self.gun_time = ttk.Label(self.buttons_frame, text="Gunshot heard at:", font=("Helvetica", 12))
        #self.gun_time.pack(side=tk.LEFT, padx=75, pady=20)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.update()

    def start_video(self):
        self.is_recording = True
        self.current_segment = []
        self.status_label.config(text="Recording...", foreground="green")
        self.progress_bar["value"] = 0
        
        self.audio_filename = "audio.wav"
        self.audio_recorder = AudioRecorder(self.audio_filename)
        self.audio_recorder.start_recording()
        
        

    def stop_video(self):
        if self.is_recording:
            self.is_recording = False
            if self.current_segment:
                # Use a separate thread for video processing to keep the GUI responsive
                processing_thread = Thread(target=self.process_pose, args=(self.current_segment,))
                processing_thread.start()
            self.status_label.config(text="Not recording", foreground="red")
            self.audio_recorder.stop_recording()
            

    def process_pose(self, segment):
        total_frames = len(segment)
        pose_landmarks = []

        
        out_filename = f'output_pose.mp4'
        out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.width, self.height))

        for i, frame in enumerate(segment):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame_rgb)

            # Update progress bar
            progress_value = int((i + 1) / total_frames * 100)
            self.progress_bar["value"] = progress_value
    
        out.release()
        print("Pose estimation video saved as output_pose.mp4")
        
        self.progress_bar["value"] = 0
        
        detection_time, frame_c = md.track_movement(cv2.VideoCapture("output_pose.mp4"), self.width, self.height)
        
    
        video = cv2.VideoCapture("output_pose.mp4")
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count/fps
        
        TOTAL_DURATION = duration + librosa.get_duration(filename='audio.wav')
        
        

        actual_det_time = (frame_c / frame_count) * TOTAL_DURATION
        
        self.gunshot_time = self.audio_recorder.gunshotDetection()
        
        actual_aud_time = int(self.gunshot_time)
        
        self.start_time.config(text=f"Athlete started at: " + str(actual_det_time) + " seconds")
        
        print("det time:" + str(actual_det_time))
        print("Aud time:" + str(actual_aud_time))


        # Reset progress bar after processing

    def draw_landmarks(self, frame, landmarks):
        frame_with_landmarks = frame.copy()
        h, w, _ = frame.shape

        for landmark in landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame_with_landmarks, (cx, cy), 5, (0, 255, 0), -1)

        return frame_with_landmarks

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            if self.is_recording:
                self.current_segment.append(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            self.canvas.img = img
            self.canvas.config(width=img.width(), height=img.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

        self.root.after(10, self.update)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if file_path:
            # Use a separate thread for video processing to keep the GUI responsive
            processing_thread = Thread(target=self.process_uploaded_video, args=(file_path,))
            processing_thread.start()

    def process_uploaded_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_landmarks = []

        out_filename = f'output_pose_estimation_uploaded.mp4'
        out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))

        for i in range(total_frames):
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            out.write(image)

            # Update progress bar
            progress_value = int((i + 1) / total_frames * 100)
            self.progress_bar["value"] = progress_value
            self.root.update_idletasks()

        cap.release()
        out.release()

        print(f"Pose estimation video saved as {out_filename}")

        # Reset progress bar after processing
        self.progress_bar["value"] = 0

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()



class AudioRecorder:
    def __init__(self, filename):
        self.filename = filename
        self.frames = []
        self.stream = None
        self.chunk = 256
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100


    def start_recording(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk,
                                  input_device_index=2)
        self.is_recording = True

        self.thread = Thread(target=self.record)
        self.thread.start()

    def record(self):
        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

    def stop_recording(self):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()


    def gunshotDetection(self):
        
        obj = wave.open("audio.wav", "rb")
        sample_freq = obj.getframerate()
        n_samples = obj.getnframes()
        signal_wave = obj.readframes(-1)
        obj.close()
        t_audio = n_samples / sample_freq
        signal_array = np.frombuffer(signal_wave, dtype=np.int16)
        times = np.linspace(0, t_audio, num=n_samples)
        max_index = np.argmax(signal_array)/2
        time_of_max = round(times[int(max_index)], 7)
        app.gun_time.config(text=f"Gunshot heard at: " + str(time_of_max) + " seconds")
        print("gunshot at " + str(time_of_max))
        
        return time_of_max









if __name__ == "__main__":
    
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
