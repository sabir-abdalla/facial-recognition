Siamese Network Facial Recognition

A real-time facial recognition application using a Siamese Neural Network, built with TensorFlow/Keras and Kivy.

How it works
This project uses a Siamese Network with an L1 Distance layer to calculate the similarity between two images. The model is trained to recognize faces based on one-shot learning.


Project Structure
- `faceid.py`: The main Kivy application for real-time webcam verification.
- `retrain_siamese.py`: Script to train the Siamese network.
- `collect_negatives.py`: Helper script to capture webcam images of other people for training.
- `diagnose_model.py`: Evaluates the model against true positives, false positives, and strangers.
- `distance_layer.py`: Custom Keras layer for L1 distance calculation.


Setup Instructions
1. Clone the repo: `git clone https://github.com/sabir-abdalla/facial-recognition.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your data:
   - Place your own images in `data/anchor` and `data/positive`
   - Place negative images in `data/negative` (and `data/negative_webcam`)
4. Train the model: `python retrain_siamese.py`
5. Run the app: `python faceid.py`
