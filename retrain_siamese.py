"""
Retrain the Siamese network so it learns FACES instead of "webcam photo vs LFW photo".

What is different from the tutorial:
  1. Pairs are built from IDENTITIES. Same identity = 1, different identity = 0.
     Identities: you, webcam strangers (data\\negative_webcam\\<person>\\), and LFW people (from the file names).
  2. Negatives include webcam photos of other people, so "looks like my webcam" no longer means "match".
  3. The first image of a pair is not always you (LFW and stranger pairs too).
  4. Proper pooling (the tutorial's MaxPooling2D(64, ...) was a 64x64 window) and a Dropout layer.
  5. Augmentation (flip, brightness, contrast) on training images only.
  6. Validation uses people / frames the model has NOT trained on:
       - your last 20% of photos (by time), held-out LFW people, a held-out webcam person (if you have 3+),
         and the last 20% (by time) of loose stranger photos.
  7. The best model (lowest validation loss) is saved as siamesemodel_v2.keras.

Folder layout it expects (in the folder that holds this file):
    data\\anchor\\*.jpg            your photos
    data\\positive\\*.jpg          your photos
    data\\negative\\*.jpg          LFW photos (file names like Firstname_Lastname_0001.jpg)
    data\\negative_webcam\\<person>\\*.jpg   webcam photos of other people (one sub-folder per person)
    data\\negative_webcam\\*.jpg            (optional) loose webcam photos of strangers
"""
import os
import re
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout


# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANC_PATH = os.path.join(BASE_DIR, 'data', 'anchor')
POS_PATH = os.path.join(BASE_DIR, 'data', 'positive')
LFW_PATH = os.path.join(BASE_DIR, 'data', 'negative')
CAM_NEG_PATH = os.path.join(BASE_DIR, 'data', 'negative_webcam')
MODEL_OUT = os.path.join(BASE_DIR, 'siamesemodel_v2.keras')

EPOCHS = 15
BATCH_SIZE = 16
PATIENCE = 4                         # stop if validation loss hasn't improved for this many epochs
SEED = 42

# training pairs per epoch (new random pairs are drawn every epoch)
N_ME_POS = 800        # you vs you
N_CAM_POS = 200       # same webcam stranger vs same webcam stranger (needs one sub-folder per person)
N_LFW_POS = 400       # same LFW person vs same LFW person
N_ME_VS_CAM = 500     # you vs webcam stranger
N_ME_VS_LFW = 300     # you vs LFW
N_CAM_VS_LFW = 100    # webcam stranger vs LFW
N_CAM_VS_CAM = 200    # different webcam strangers
N_LFW_VS_LFW = 300    # different LFW people

# validation pairs (built once)
V_ME_POS, V_CAM_POS, V_LFW_POS = 150, 40, 100
V_CAM_VS_ME, V_LFW_VS_ME, V_LFW_VS_LFW = 150, 100, 40

MIN_WEBCAM_STRANGER_IMAGES = 100     # below this the model would repeat the original problem
ALLOW_FEW_WEBCAM_NEGATIVES = False   # set True only if you really want to train anyway

IMG_EXT = ('.jpg', '.jpeg', '.png')


# ----------------------------------------------------------------------
# FINDING IMAGES / IDENTITIES
# ----------------------------------------------------------------------
def images_directly_in(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def load_me():
    imgs = images_directly_in(ANC_PATH) + images_directly_in(POS_PATH)
    imgs.sort(key=os.path.getmtime)          # oldest first, so the LAST ones become validation
    return imgs


def load_cam_strangers():
    people, loose = {}, []
    if os.path.isdir(CAM_NEG_PATH):
        loose = images_directly_in(CAM_NEG_PATH)
        loose.sort(key=os.path.getmtime)
        for d in sorted(os.listdir(CAM_NEG_PATH)):
            full = os.path.join(CAM_NEG_PATH, d)
            if os.path.isdir(full):
                imgs = images_directly_in(full)
                if imgs:
                    people[d] = imgs
    return people, loose


def load_lfw():
    identities = {}
    for f in images_directly_in(LFW_PATH):
        stem = os.path.splitext(os.path.basename(f))[0]
        name = re.sub(r'_\d+$', '', stem)   
        identities.setdefault(name, []).append(f)
    return identities


def flatten(identities):
    return [p for paths in identities.values() for p in paths]



# training/validation - has to be random
def split_data(me, cam_people, cam_loose, lfw):
    rng = random.Random(SEED)

    cut = int(len(me) * 0.8)
    me_train, me_val = me[:cut], me[cut:]

    names = sorted(lfw)
    rng.shuffle(names)
    n_val = max(1, int(len(names) * 0.15)) if len(names) >= 2 else 0
    lfw_val = {n: lfw[n] for n in names[:n_val]}
    lfw_train = {n: lfw[n] for n in names[n_val:]}

    pnames = sorted(cam_people)
    held = [rng.choice(pnames)] if len(pnames) >= 3 else []
    cam_val_people = {n: cam_people[n] for n in held}
    cam_train_people = {n: v for n, v in cam_people.items() if n not in held}

    loose_cut = int(len(cam_loose) * 0.8)
    loose_train, loose_val = cam_loose[:loose_cut], cam_loose[loose_cut:]

    return {
        'me_train': me_train, 'me_val': me_val,
        'lfw_train': lfw_train, 'lfw_val': lfw_val,
        'cam_train_people': cam_train_people, 'cam_val_people': cam_val_people,
        'loose_train': loose_train, 'loose_val': loose_val,
    }



# pair building
def pairs_same(identities, n):
    """Same-identity pairs (label 1) from a dict identity -> image paths."""
    names = [k for k, v in identities.items() if len(v) >= 2]
    out = []
    if not names:
        return out
    for _ in range(n):
        k = random.choice(names)
        a, b = random.sample(identities[k], 2)
        out.append((a, b, 1.0))
    return out


def pairs_between(pool_a, pool_b, n, label):
    """Random pairs with one image from each pool."""
    out = []
    if not pool_a or not pool_b:
        return out
    for _ in range(n):
        a, b = random.choice(pool_a), random.choice(pool_b)
        if a != b:
            out.append((a, b, float(label)))
    return out


def pairs_across_identities(identities, n):
    """Different-identity pairs (label 0) from a dict identity -> image paths."""
    names = list(identities)
    out = []
    if len(names) < 2:
        return out
    for _ in range(n):
        k1, k2 = random.sample(names, 2)
        out.append((random.choice(identities[k1]), random.choice(identities[k2]), 0.0))
    return out


def balance(pos, neg):
    n = min(len(pos), len(neg))
    random.shuffle(pos)
    random.shuffle(neg)
    pairs = pos[:n] + neg[:n]
    random.shuffle(pairs)
    return pairs


def build_train_pairs(sp):
    cam_all = flatten(sp['cam_train_people']) + sp['loose_train']
    lfw_all = flatten(sp['lfw_train'])
    pos = (pairs_same({'me': sp['me_train']}, N_ME_POS)
           + pairs_same(sp['cam_train_people'], N_CAM_POS)
           + pairs_same(sp['lfw_train'], N_LFW_POS))
    neg = (pairs_between(sp['me_train'], cam_all, N_ME_VS_CAM, 0)
           + pairs_between(sp['me_train'], lfw_all, N_ME_VS_LFW, 0)
           + pairs_between(cam_all, lfw_all, N_CAM_VS_LFW, 0)
           + pairs_across_identities(sp['cam_train_people'], N_CAM_VS_CAM)
           + pairs_across_identities(sp['lfw_train'], N_LFW_VS_LFW))
    return balance(pos, neg)


def build_val_pairs(sp):
    """Mimics the app: a NEW image is compared against your reference (training) photos."""
    cam_val_all = flatten(sp['cam_val_people']) + sp['loose_val']
    lfw_val_all = flatten(sp['lfw_val'])
    pos = (pairs_between(sp['me_val'], sp['me_train'], V_ME_POS, 1)
           + pairs_same(sp['cam_val_people'], V_CAM_POS)
           + pairs_same(sp['lfw_val'], V_LFW_POS))
    neg = (pairs_between(cam_val_all, sp['me_train'], V_CAM_VS_ME, 0)
           + pairs_between(lfw_val_all, sp['me_train'], V_LFW_VS_ME, 0)
           + pairs_across_identities(sp['lfw_val'], V_LFW_VS_LFW))
    return balance(pos, neg), len(cam_val_all)



# building data pipeline
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_image(byte_img, channels=3, expand_animations=False)   # jpg, jpeg or png
    img = tf.image.resize(img, (100, 100))
    return img / 255.0


def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 1.0)


def make_dataset(pairs, training):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    y = np.array([p[2] for p in pairs], dtype='float32')
    ds = tf.data.Dataset.from_tensor_slices((a, b, y))
    if training:
        ds = ds.shuffle(len(pairs), reshuffle_each_iteration=True)

    def load(pa, pb, label):
        ia, ib = preprocess(pa), preprocess(pb)
        if training:
            ia, ib = augment(ia), augment(ib)
        return ia, ib, tf.expand_dims(label, -1)

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



# model

class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(pool_size=(2, 2), padding='same')(c1)

    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(pool_size=(2, 2), padding='same')(c2)

    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(pool_size=(2, 2), padding='same')(c3)

    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    f1 = Dropout(0.3)(f1)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=inp, outputs=d1, name='embedding')


def make_siamese_model(embedding):
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_image', shape=(100, 100, 3))

    distances = L1Dist(name='distance')([embedding(input_image), embedding(validation_image)])
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')



# evaluation

def evaluate(model, ds):
    preds, trues = [], []
    for a, b, y in ds:
        preds.append(model([a, b], training=False).numpy())
        trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    loss = float(tf.keras.losses.BinaryCrossentropy()(trues, preds).numpy())
    acc = float(np.mean((preds > 0.5) == (trues > 0.5)))
    return loss, acc, preds.flatten(), trues.flatten()



# main

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    me = load_me()
    cam_people, cam_loose = load_cam_strangers()
    lfw = load_lfw()
    n_cam = sum(len(v) for v in cam_people.values()) + len(cam_loose)

    print('your photos:               ', len(me))
    print('webcam strangers:          ', n_cam, '({} people in sub-folders + {} loose photos)'.format(len(cam_people), len(cam_loose)))
    print('LFW photos / identities:   ', sum(len(v) for v in lfw.values()), '/', len(lfw))

    if len(me) < 50:
        print('Need at least 50 photos of you in data\\anchor and data\\positive.')
        return
    if n_cam < MIN_WEBCAM_STRANGER_IMAGES and not ALLOW_FEW_WEBCAM_NEGATIVES:
        print('\nSTOP: only {} webcam photos of other people (need {}+).'.format(n_cam, MIN_WEBCAM_STRANGER_IMAGES))
        print('Without them the model repeats the original mistake (webcam look = match).')
        print('Run collect_negatives.py once per person, then run this again.')
        return
    if len(cam_people) < 3 and not cam_loose:
        print('Tip: with 3+ people in separate sub-folders, one whole person is held out for validation.')

    sp = split_data(me, cam_people, cam_loose, lfw)
    val_pairs, n_cam_val = build_val_pairs(sp)
    if not val_pairs:
        print('Could not build validation pairs - check your folders.')
        return
    if n_cam_val == 0:
        print('WARNING: no held-out webcam strangers, so validation cannot test unseen strangers.')
        print('         Use diagnose_model.py with a stranger_test folder as the real check.')
    val_ds = make_dataset(val_pairs, training=False)
    print('validation pairs:', len(val_pairs), '| held-out webcam stranger photos:', n_cam_val)

    embedding = make_embedding()
    siamese_model = make_siamese_model(embedding)
    siamese_model.summary()

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(a, b, y):
        with tf.GradientTape() as tape:
            yhat = siamese_model([a, b], training=True)
            loss = loss_fn(y, yhat)
        grads = tape.gradient(loss, siamese_model.trainable_variables)
        opt.apply_gradients(zip(grads, siamese_model.trainable_variables))
        return loss

    best_val, wait = float('inf'), 0
    for epoch in range(1, EPOCHS + 1):
        train_pairs = build_train_pairs(sp)          # fresh random pairs every epoch
        train_ds = make_dataset(train_pairs, training=True)
        n_batches = -(-len(train_pairs) // BATCH_SIZE)

        print('\nEpoch {}/{}  ({} training pairs)'.format(epoch, EPOCHS, len(train_pairs)))
        progbar = tf.keras.utils.Progbar(n_batches)
        losses = []
        for i, (a, b, y) in enumerate(train_ds):
            losses.append(float(train_step(a, b, y)))
            progbar.update(i + 1)

        val_loss, val_acc, _, _ = evaluate(siamese_model, val_ds)
        print('train loss {:.4f} | val loss {:.4f} | val accuracy {:.1%}'.format(np.mean(losses), val_loss, val_acc))

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            siamese_model.save(MODEL_OUT)
            print('  saved best model ->', MODEL_OUT)
        else:
            wait += 1
            if wait >= PATIENCE:
                print('  validation loss has not improved for {} epochs - stopping.'.format(PATIENCE))
                break

    # ---- report on the best saved model ----
    best = tf.keras.models.load_model(MODEL_OUT, custom_objects={'L1Dist': L1Dist}, compile=False)
    _, acc, preds, trues = evaluate(best, val_ds)
    print('\nBEST MODEL on held-out validation pairs: accuracy {:.1%}'.format(acc))
    print('  same person:      mean score {:.3f}'.format(preds[trues == 1].mean()))
    print('  different people: mean score {:.3f}'.format(preds[trues == 0].mean()))
    thresholds = np.arange(0.05, 1.0, 0.05)
    accs = [np.mean((preds > t) == (trues == 1)) for t in thresholds]
    k = int(np.argmax(accs))
    print('  best single threshold on validation: {:.2f} (accuracy {:.1%})'.format(thresholds[k], accs[k]))
    print('\nNext: run diagnose_model.py (it now tests siamesemodel_v2.keras) with your stranger_test photos.')
    print('The strangers line should be LOW and clearly below the YOU line.')


if __name__ == '__main__':
    main()
