import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import sys

# Set encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Convert to lowercase and handle encoding
    text = str(text).lower().encode('ascii', 'ignore').decode('ascii')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Keep only letters and important punctuation
    text = re.sub(r'[^a-z\s!?.]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def augment_text(text):
    tokens = text.split()
    if len(tokens) < 3:
        return [text]
        
    augmented = []
    # Original text
    augmented.append(text)
    
    # Shuffle some words
    if len(tokens) > 3:
        shuffled = tokens.copy()
        if len(shuffled) >= 2:
            # Safely shuffle adjacent pairs
            for i in range(0, len(shuffled)-1, 2):
                shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
            augmented.append(' '.join(shuffled))
    
    # Remove random word
    if len(tokens) > 3:
        removed = tokens.copy()
        remove_idx = random.randint(0, len(removed)-1)
        del removed[remove_idx]
        augmented.append(' '.join(removed))
    
    return augmented

print("Loading and preprocessing data...")
with open('C:/Users/PRAVEENA/Downloads/Annotated json files/annotated.json', encoding='utf-8') as f:
    data = json.load(f)

texts = []
labels = []

for item in data['annotations']:
    if item and len(item) > 1 and item[1]:
        text = item[0]
        entities = item[1].get('entities', [])
        if entities:
            processed_text = preprocess_text(text)
            if len(processed_text.split()) >= 3:
                texts.append(processed_text)
                labels.append(entities[0][2])

print(f"Initial samples: {len(texts)}")

# Remove rare classes and balance dataset
min_samples = 5
class_distribution = Counter(labels)
valid_labels = {label for label, count in class_distribution.items() if count >= min_samples}
filtered_texts = []
filtered_labels = []

for text, label in zip(texts, labels):
    if label in valid_labels:
        filtered_texts.append(text)
        filtered_labels.append(label)

texts = filtered_texts
labels = filtered_labels

print(f"After filtering rare classes: {len(texts)}")

# Augment data
augmented_texts = []
augmented_labels = []

for text, label in zip(texts, labels):
    aug_texts = augment_text(text)
    augmented_texts.extend(aug_texts)
    augmented_labels.extend([label] * len(aug_texts))

texts = augmented_texts
labels = augmented_labels

print(f"After augmentation: {len(texts)}")

# Calculate class weights
class_weights = {}
label_distribution = Counter(labels)
max_count = max(label_distribution.values())
for label, count in label_distribution.items():
    class_weights[label] = max_count / count

# Convert class weights to numeric format
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)
numeric_weights = {i: class_weights[label] for i, label in enumerate(label_encoder.classes_)}

# Tokenization
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)

# Padding
max_length = 50
X = pad_sequences(X, maxlen=max_length, padding='post', truncating='post')

# Convert labels
y = label_encoder.transform(labels)
y = to_categorical(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("\nBuilding model...")
model = Sequential([
    Embedding(vocab_size, 100, mask_zero=True),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

# Compile with custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.0001
)

# Print model summary
model.summary()

# Train
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    class_weight=numeric_weights,
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")

# Save model
print("\nSaving model...")
model.save('dis_lstm.keras')

# Save tokenizer and label encoder
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

