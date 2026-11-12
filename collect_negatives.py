"""
Collect webcam photos of OTHER people (the negatives the model was missing).

Put this file in D:\\computer-projects\\facial-recognition\\ and run it once per person.
Photos are saved to  data\\negative_webcam\\<person name>\\  using the same 250x250 crop as everything else.

Keys (click the webcam window first):
    n = save one photo
    s = start / stop AUTO-SAVE (one photo every 0.3 s - slowly turn the head, change expression)
    q = quit
"""
import os
import time
import uuid
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEG_WEBCAM = os.path.join(BASE_DIR, 'data', 'negative_webcam')

person = input('Name for this person (no spaces, e.g. person1): ').strip().replace(' ', '_') or 'person1'
person_dir = os.path.join(NEG_WEBCAM, person)
os.makedirs(person_dir, exist_ok=True)
count = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print('Saving to', person_dir)
print('n = one photo | s = auto-save on/off | q = quit')

auto = False
last_save = 0.0
cap = cv2.VideoCapture(0)


def save(frame):
    global count
    path = os.path.join(person_dir, '{}.jpg'.format(uuid.uuid1()))
    if cv2.imwrite(path, frame):
        count += 1
    else:
        print('Could not write', path)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Could not read from the webcam')
        break

    # same 250px by 250px crop used everywhere else
    frame = frame[120:120+250, 200:200+250, :]

    shown = frame.copy()
    cv2.putText(shown, '{}: {} saved{}'.format(person, count, ' (AUTO)' if auto else ''),
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow('Collect negatives', shown)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('n'):
        save(frame)
    elif key == ord('s'):
        auto = not auto
    elif key == ord('q'):
        break

    if auto and time.time() - last_save > 0.3:
        save(frame)
        last_save = time.time()

cap.release()
cv2.destroyAllWindows()
print('Done.', count, 'photos in', person_dir)
