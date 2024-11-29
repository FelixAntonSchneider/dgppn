import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QGridLayout, QSpacerItem, QSizePolicy, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QImage
import sounddevice as sd
from scipy.io.wavfile import write
import threading
from ExpoAI import ExpoAI
from PIL import Image


class AudioRecorderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.is_recording = False
        self.record_thread = None

    def init_ui(self):
        # Set up text box
        self.text_box = QLineEdit(self)
        self.text_box.setPlaceholderText("Describe your fear...")
        self.text_box.setFixedHeight(200)

        # Set up record button
        self.record_button = QPushButton(self)
        self.record_button.setCheckable(True)
        self.record_button.setIcon(QIcon("R.png"))
        self.record_button.setIconSize(self.record_button.size() * 3)  # Make the icon larger
        self.record_button.setStyleSheet(
            "background-color: lightgray;"
            "border-radius: 150px;"  # Making the button circular
            "min-width: 300px;"
            "min-height: 300px;"
        )
        self.record_button.clicked.connect(self.toggle_recording)

        # Set up generate button
        self.generate_button = QPushButton("Generate", self)
        self.generate_button.setStyleSheet(
            "background-color: #007AFF;"  # Apple blue color
            "color: white;"
            "border-radius: 20px;"
            "padding: 20px 40px;"
            "font-size: 32px;"
        )
        self.generate_button.clicked.connect(self.show_generated_image)

        # Set up reset button
        self.reset_button = QPushButton("Reset", self)
        self.reset_button.setStyleSheet(
            "background-color: #007AFF;"  # Apple blue color
            "color: white;"
            "border-radius: 20px;"
            "padding: 20px 40px;"
            "font-size: 32px;")
        self.reset_button.clicked.connect(self.reset_interface)

        # Set up label to display image and text
        self.text_label = QLabel(self)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setWordWrap(True)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMaximumSize(800, 800)  # Limit image size to fit in window

        # Set up layout for left side (1/4 width)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.text_box)
        left_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))  # Add space
        left_layout.addWidget(self.record_button)
        left_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))  # Add space
        left_layout.addWidget(self.generate_button)
        left_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))  # Add space

        # Set up main layout
        main_layout = QGridLayout()
        main_layout.addLayout(left_layout, 0, 0, 1, 1)
        main_layout.setColumnStretch(0, 1)  # Left side takes 1/4 of the space
        main_layout.setColumnStretch(1, 2)  # Right side takes 3/4 of the space
        main_layout.addWidget(self.text_label, 0, 1, Qt.AlignTop)  # Add text label above the image
        main_layout.addWidget(self.image_label, 0, 1, Qt.AlignCenter)  # Add image label to the right section
        main_layout.addWidget(self.reset_button, 1, 1, Qt.AlignBottom | Qt.AlignRight)

        self.setLayout(main_layout)
        self.setWindowTitle("ExpoAI")
        self.setWindowIcon(QIcon("logo.png"))
        self.setGeometry(100, 100, 1600, 1200)

    # Function to record audio
    def record_audio(self, filename="output.wav", duration=5, sample_rate=44100):
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        write(filename, sample_rate, audio)
        self.transcribe_recording()
        print(f"Recording saved as {filename}")

    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.setStyleSheet(
                "background-color: red;"
                "border-radius: 150px;"
                "min-width: 300px;"
                "min-height: 300px;"
            )
            self.record_thread = threading.Thread(target=self.start_recording)
            self.record_thread.start()
        else:
            # Stop recording
            self.is_recording = False
            self.record_button.setStyleSheet(
                "background-color: lightgray;"
                "border-radius: 150px;"
                "min-width: 300px;"
                "min-height: 300px;"
            )


    def start_recording(self):
        try:
            # Record indefinitely until the button is pressed again
            self.record_audio("output.wav", duration=10)
        except Exception as e:
            print(f"Recording interrupted: {e}")

    def transcribe_recording(self):
        try:
            # Use ExpoAI to transcribe the audio recording
            expo_ai = ExpoAI(audiofilepath="output.wav")
            transcription = expo_ai.patient_text
            self.text_box.setText(transcription)
        except Exception as e:
            print(f"Transcription failed: {e}")

    def show_generated_image(self):
        # Create an instance of ExpoAI with the text from the input box
        input_text = self.text_box.text()
        expo_ai = ExpoAI(patient_text=input_text)
        summary = expo_ai.psy_text
        image = expo_ai.image
        if image is None:
            self.text_label.setText(summary)
            return

        # Display the summary and the generated image
        self.text_label.setText(summary)
        #qt_image = Image.Image(image)
        # Convert PIL Image to QImage
        image = image.convert("RGBA")  # Ensure the image has an alpha channel
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def reset_interface(self):
        # Reset the text box, text label, image label, and recording state
        self.text_box.clear()
        self.text_label.clear()
        self.image_label.clear()
        if self.is_recording:
            self.toggle_recording()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioRecorderApp()
    window.show()
    sys.exit(app.exec_())
