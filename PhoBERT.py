import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
import re
from underthesea import word_tokenize, text_normalize
import tensorflow as tf

# Load data
with open('tagged_events.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load Vietnamese stopwords
with open(r'/content/vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()

# Preprocess text function
def preprocess_text(text):
    text = str(text).lower()
    text = text_normalize(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = word_tokenize(text, format="text")
    return text

# Extract texts and labels
texts = [' '.join(item['tokens']) for item in data]
labels = [item['event_mentions'][0]['event_type'] if item.get('event_mentions') else 'Unknown' for item in data]

# Create DataFrame
df = pd.DataFrame({'text': texts})
df['text'] = df['text'].apply(preprocess_text)

# Split data
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels)
dev_texts, test_texts, dev_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
max_length = 150
X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_length, padding='post')
X_dev = pad_sequences(tokenizer.texts_to_sequences(dev_texts), maxlen=max_length, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_length, padding='post')

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels + dev_labels + test_labels)
y_train = label_encoder.transform(train_labels)
y_dev = label_encoder.transform(dev_labels)
y_test = label_encoder.transform(test_labels)

# Define BiLSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=256, input_length=max_length, trainable=True),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    Bidirectional(LSTM(32)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(set(labels)), activation='softmax')
])

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=0.00005)

# Train the model
history = model.fit(
    X_train, y_train, epochs=50, batch_size=64, validation_data=(X_dev, y_dev),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

# Plot loss history
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.show()