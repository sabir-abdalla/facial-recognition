from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
import cv2
import tensorflow as tf
from distance_layer import L1Dist
import os
import numpy as np


APP_DIR      = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.dirname(APP_DIR)                      
_V2_PATH     = os.path.join(PROJECT_ROOT, 'siamesemodel_v2.keras')    
MODEL_PATH   = _V2_PATH if os.path.exists(_V2_PATH) else os.path.join(PROJECT_ROOT, 'siamesemodel.keras')



def find_app_data():
    # application_data may sit inside the app folder OR in the project root - use the one that holds your images
    candidates = [os.path.join(APP_DIR, 'application_data'),
                  os.path.join(PROJECT_ROOT, 'application_data')]
    for c in candidates:
        verif = os.path.join(c, 'verification_images')
        if os.path.isdir(verif) and any(f.lower().endswith('.jpg') for f in os.listdir(verif)):
            return c
    return candidates[0]


# absolute paths, so it doesn't matter which folder you start the app from
APP_DATA_DIR = find_app_data()
INPUT_DIR  = os.path.join(APP_DATA_DIR, 'input_image')
VERIF_DIR  = os.path.join(APP_DATA_DIR, 'verification_images')
INPUT_PATH = os.path.join(INPUT_DIR, 'input_image.jpg')
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(VERIF_DIR, exist_ok=True)

# Set from diagnose_model.py on the retrained model (siamesemodel_v2.keras):
#   at score > 0.5 you matched 100% of the reference photos, strangers matched 0%.
DETECTION_THRESHOLD = 0.5       # a single comparison counts as a match above this score
VERIFICATION_THRESHOLD = 0.5    # share of verification images that must match


class CamApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        Logger.info('Loading model: {}'.format(MODEL_PATH))
        Logger.info('application_data folder: {}'.format(APP_DATA_DIR))
        self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'L1Dist': L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = frame[120:120+250, 200:200+250, :]

        # Flip vertically and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and convert to 100x100px
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    # Verification function to verify person
    def verify(self, *args):
        # Capture input image from our webcam
        ret, frame = self.capture.read()
        if not ret:
            self.verification_label.text = 'Camera error'
            return
        frame = frame[120:120+250, 200:200+250, :]

        # cv2.imwrite fails silently, so check it - otherwise an OLD input image would be compared
        if not cv2.imwrite(INPUT_PATH, frame):
            self.verification_label.text = 'Could not save input image'
            Logger.error('Could not write {}'.format(INPUT_PATH))
            return

        verification_files = [f for f in os.listdir(VERIF_DIR) if f.lower().endswith('.jpg')]
        if len(verification_files) == 0:
            self.verification_label.text = 'No verification images'
            Logger.error('No .jpg files in {}'.format(VERIF_DIR))
            return

        # Compare the new input image against ALL verification images in a single prediction
        input_img = self.preprocess(INPUT_PATH)
        validation_imgs = tf.stack([self.preprocess(os.path.join(VERIF_DIR, f)) for f in verification_files])
        input_imgs = tf.repeat(tf.expand_dims(input_img, 0), len(verification_files), axis=0)
        results = self.model.predict([input_imgs, validation_imgs], verbose=0).flatten()

        # Detection Threshold: Metric above which a prediction is considered positive
        detection = int(np.sum(results > DETECTION_THRESHOLD))

        # Verification Threshold: Proportion of positive predictions / total verification images
        verification = detection / len(verification_files)
        verified = verification > VERIFICATION_THRESHOLD

        # Set verification text
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log out details (the score spread is the most useful thing to look at)
        Logger.info('images compared: {}'.format(len(verification_files)))
        Logger.info('scores: min={:.3f} mean={:.3f} max={:.3f}'.format(results.min(), results.mean(), results.max()))
        Logger.info('detection: {}'.format(detection))
        Logger.info('verification: {}'.format(verification))
        Logger.info('verified: {}'.format(verified))


if __name__ == '__main__':
    CamApp().run()
