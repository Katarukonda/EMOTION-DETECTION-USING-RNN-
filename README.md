🎭 Sentiment Analysis & Emotion Detection using GRU
This project implements a robust Deep Learning pipeline for Sentiment Analysis, classifying text into Positive, Negative, or Neutral categories. Built with TensorFlow/Keras, it processes the Google GoEmotions dataset using a stacked GRU (Gated Recurrent Unit) architecture optimized for sequential text data.

🧠 Project Overview & Methodology
The original dataset contains 28 specific emotion labels (e.g., Joy, Remorse, Curiosity), which often leads to sparse data and poor model performance. This project solves that challenge through Target Engineering, mapping these 28 labels into three distinct sentiment classes. Ambiguous emotions like confusion or surprise are intelligently mapped to the Neutral class to maximize clarity.

To address class imbalance, the pipeline utilizes RandomOverSampler from the imblearn library. This ensures the model trains on an equal distribution of positive, negative, and neutral examples, preventing bias toward the majority class.

🏗️ Model Architecture
The neural network is designed for both accuracy and stability:

Embedding with Masking: The Embedding layer uses mask_zero=True, forcing the model to ignore padding zeros. This significantly improves accuracy on variable-length text.

Stacked GRU Layers: Two GRU layers (128 and 64 units) capture long-term dependencies in the text.

Stability Configuration: The layers utilize recurrent_dropout=0.2, a critical configuration that allows the model to handle masking correctly without triggering GPU compatibility errors.

Regularization: SpatialDropout1D and standard Dropout layers are applied to prevent overfitting.

🛠️ Tech Stack
Core: Python, TensorFlow, Keras

Data Processing: Pandas, NumPy, Scikit-Learn

NLP Tools: Tokenizer, RegEx, Padding

Balancing: Imbalanced-Learn (RandomOverSampler)
