
import os
import glob
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POS_PATH = os.path.join(BASE_DIR, 'data', 'positive')
NEG_PATH = os.path.join(BASE_DIR, 'data', 'negative')


def find_app_data():
    # application_data may sit in the project root OR inside the app folder - use the one that holds your images
    candidates = [os.path.join(BASE_DIR, 'app', 'application_data'),
                  os.path.join(BASE_DIR, 'application_data')]
    for c in candidates:
        if glob.glob(os.path.join(c, 'verification_images', '*.jpg')):
            return c
    return candidates[0]


APP_DATA_DIR = find_app_data()
VERIF_DIR = os.path.join(APP_DATA_DIR, 'verification_images')
INPUT_IMG = os.path.join(APP_DATA_DIR, 'input_image', 'input_image.jpg')
print('data folder:            ', os.path.join(BASE_DIR, 'data'))
print('application_data folder:', APP_DATA_DIR)

# put webcam photos of OTHER people here (see instructions printed at the end)
STRANGER_DIR = os.path.join(BASE_DIR, 'stranger_test')
os.makedirs(STRANGER_DIR, exist_ok=True)

# Live captures of YOU taken with the Kivy app (copy input_image.jpg here after each Verify click).
# This is the honest test for you: the 'data/positive' group mostly contains photos the model trained on.
YOU_TEST_DIR = os.path.join(BASE_DIR, 'you_test')
os.makedirs(YOU_TEST_DIR, exist_ok=True)

MODEL_FILES = [os.path.join(BASE_DIR, 'siamesemodel.keras'),
               os.path.join(BASE_DIR, 'siamesemodel_v2.keras'),   # the retrained model
               os.path.join(BASE_DIR, 'siamesemodel.h5')]

N_QUERIES = 20   # images tested per group (each is compared against every verification image)
MATCH_THRESHOLDS = (0.25, 0.5, 0.75)   # per-comparison score levels to report the 'share of reference photos matched' for


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_image(byte_img, channels=3, expand_animations=False)   # handles jpg, jpeg and png
    img = tf.image.resize(img, (100, 100))
    return img / 255.0


def jpgs(folder):
    return sorted(glob.glob(os.path.join(folder, '*.jpg')))


def images_in(folder):
    
    found = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                found.append(os.path.join(root, f))
    return sorted(found)


def score_group(model, name, query_paths, refs):
    if not query_paths:
        print('  {:<34} no images found'.format(name))
        return
    all_scores = []
    for q in query_paths:
        q_batch = tf.repeat(tf.expand_dims(preprocess(q), 0), refs.shape[0], axis=0)
        all_scores.append(model.predict([q_batch, refs], batch_size=32, verbose=0).flatten())
    s = np.concatenate(all_scores)
    print('  {:<34} n={:<3} mean={:.3f} | pairs>0.5: {:>4.0%} | pairs>0.99: {:>4.0%}'.format(
        name, len(query_paths), s.mean(), np.mean(s > 0.5), np.mean(s > 0.99)))
    # the app verifies someone when enough of the reference photos score above the detection threshold
    for t in MATCH_THRESHOLDS:
        share = np.array([np.mean(q > t) for q in all_scores])
        print('        share of reference photos matched at score>{:.2f}: min {:>4.0%} | mean {:>4.0%} | max {:>4.0%}'.format(
            t, share.min(), share.mean(), share.max()))


# reference images 
ref_files = jpgs(VERIF_DIR)
assert len(ref_files) > 0, 'No .jpg files in {}'.format(VERIF_DIR)
print('verification_images folder holds', len(ref_files), 'jpg files')
refs = tf.stack([preprocess(f) for f in ref_files])
ref_names = {os.path.basename(f) for f in ref_files}

# query groups 
random.seed(0)

# your own images that are NOT also in verification_images 
you = [f for f in jpgs(POS_PATH) if os.path.basename(f) not in ref_names]
random.shuffle(you)
you = you[:N_QUERIES]

lfw = jpgs(NEG_PATH)
random.shuffle(lfw)
lfw = lfw[:N_QUERIES]

strangers = images_in(STRANGER_DIR)[:N_QUERIES]
you_live = images_in(YOU_TEST_DIR)[:N_QUERIES]
print('you_test folder:        ', YOU_TEST_DIR, '|', len(you_live), 'images')
all_in_stranger_dir = [os.path.join(r, f) for r, _, fs in os.walk(STRANGER_DIR) for f in fs]
print('stranger_test folder:   ', STRANGER_DIR)
print('  files inside it:', len(all_in_stranger_dir), '| usable images (jpg/jpeg/png):', len(strangers))
for f in all_in_stranger_dir[:5]:
    print('   ', os.path.relpath(f, STRANGER_DIR))

last_input = [INPUT_IMG] if os.path.exists(INPUT_IMG) else []

# run every saved model that exists
for path in MODEL_FILES:
    if not os.path.exists(path):
        continue
    stamp = os.path.getmtime(path)
    print('\n=== {} (last modified {}) ==='.format(
        os.path.basename(path), time.strftime('%Y-%m-%d %H:%M', time.localtime(stamp))))
    model = tf.keras.models.load_model(path, custom_objects={'L1Dist': L1Dist}, compile=False)

    score_group(model, 'YOU (data/positive)', you, refs)
    score_group(model, 'YOU live (you_test)', you_live, refs)
    score_group(model, 'LFW negatives (data/negative)', lfw, refs)
    score_group(model, 'STRANGERS on webcam', strangers, refs)
    score_group(model, 'last app input_image.jpg', last_input, refs)

if len(strangers) == 0:
    print('''
No usable stranger images found in the stranger_test folder printed above. To add them:
  1. Run your Kivy app and click Verify with a stranger in front of the webcam.
  2. Copy application_data\\input_image\\input_image.jpg into the stranger_test folder
     under a new name (stranger1.jpg, stranger2.jpg, ...). Aim for 10+ images, ideally of different people.
  3. Run this script again.''')

print('''
SETTING THE APP THRESHOLDS (DETECTION_THRESHOLD / VERIFICATION_THRESHOLD in faceid_fixed.py)
  Pick the score level (0.25 / 0.5 / 0.75) where YOU's lowest 'share matched' is clearly ABOVE
  STRANGERS' highest. Use that level as DETECTION_THRESHOLD and a share between the two as VERIFICATION_THRESHOLD.
  If the ranges overlap at every level, the model still needs more data, not different thresholds.

HOW TO READ THIS
  YOU should be high, LFW negatives low, STRANGERS low.
  STRANGERS as high as YOU  -> the model learned the webcam look (background, lighting, colour), not the face.
  LFW low but STRANGERS high -> exactly the pattern a dataset with only LFW negatives produces.''')
