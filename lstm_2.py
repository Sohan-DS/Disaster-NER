import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical  # For one-hot encoding of labels

# Load your dataset
with open('C:/Users/PRAVEENA/Downloads/Annotated json files/annotated.json') as f:
    data = json.load(f)

texts = []
labels = []

for item in data['annotations']:
    if item and len(item) > 1 and item[1]:  # Ensure item and its entity data exist
        text = item[0]
        entities = item[1].get('entities', [])
        if entities:  # Only add texts that have corresponding labels
            texts.append(text)
            labels.append(entities[0][2])  # Take the first entity's label

print(f"Number of text samples after filtering: {len(texts)}")
print(f"Number of labels after filtering: {len(labels)}")

# Tokenizing the texts
tokenizer = Tokenizer(num_words=5000)  # Keep only the top 5000 words
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)

# Padding the sequences to ensure uniform input size
max_length = 500  # Ensure a uniform length
X = pad_sequences(X, maxlen=max_length)

# Encoding the labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Convert the labels to one-hot encoding for categorical cross-entropy
y = to_categorical(y)

# Now splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(75, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary of the model
print(model.summary())

# Training the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluating the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Save the trained model
model.save('disaster_lstm_model.h5')

# Load the saved model
from tensorflow.keras.models import load_model
loaded_model = load_model('disaster_lstm_model.h5')

# Example new data
new_texts = [
    "Severe flood warnings in place for coastal regions", 
    "Wildfire spreads rapidly across the mountains"
]

# Tokenizing the new texts
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_X = pad_sequences(new_sequences, maxlen=max_length)

# Predict using the loaded model
predictions = loaded_model.predict(new_X)

# Get the predicted class indices
predicted_labels = predictions.argmax(axis=-1)

# Optionally decode the numeric labels back to category names
decoded_labels = label_encoder.inverse_transform(predicted_labels)

for i, text in enumerate(new_texts):
    print(f"Text: {text}")
    print(f"Predicted Label: {decoded_labels[i]}")
