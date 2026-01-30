import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load the Saved Artifacts
print("Loading model and artifacts...")
model = tf.keras.models.load_model('emotion_gru_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

# 2. Define Cleaning Function (Must match training!)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# 3. Prediction Function
def predict_emotion(text):
    # Clean and Tokenize
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
    
    # Predict
    pred = model.predict(padded, verbose=0)
    class_idx = np.argmax(pred)
    sentiment = le.inverse_transform([class_idx])[0]
    confidence = pred[0][class_idx]
    
    return sentiment, confidence

# 4. Test Loop
if __name__ == "__main__":
    print("\n--- Emotion Detection CLI ---")
    while True:
        user_input = input("\nEnter text (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        sentiment, conf = predict_emotion(user_input)
        print(f"Prediction: {sentiment.upper()} ({conf*100:.2f}%)")
