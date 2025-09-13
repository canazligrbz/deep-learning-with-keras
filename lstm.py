from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, BatchNormalization, SpatialDropout1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import re
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Metin temizleme fonksiyonu 
def clean_text(text):
    # HTML tag'larını kaldır
    text = re.sub(r'<.*?>', '', text)
    # URL'leri kaldır
    text = re.sub(r'http\S+', '', text)
    # E-posta adreslerini kaldır
    text = re.sub(r'\S+@\S+', '', text)
    # Özel karakterleri ve sayıları kaldır
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Birden fazla boşluğu tek boşluğa indirge
    text = re.sub(r'\s+', ' ', text)
    # Küçük harfe çevir
    text = text.lower().strip()
    return text

# veri yükleme
newsgroup = fetch_20newsgroups(subset="all", remove=('headers', 'footers', 'quotes'))
X = [clean_text(text) for text in newsgroup.data]
y = newsgroup.target

min_length = 30  
filtered_texts = []
filtered_labels = []
for text, label in zip(X, y):
    words = text.split()
    if len(words) >= min_length:
        # Çok uzun metinleri kısalt
        if len(words) > 500:
            text = ' '.join(words[:500])
        filtered_texts.append(text)
        filtered_labels.append(label)

X, y = filtered_texts, filtered_labels

# Kelime sıklığına göre filtreleme
tokenizer = Tokenizer(num_words=8000) 
tokenizer.fit_on_texts(X)

# Çok nadir kelimeleri kaldır
word_counts = tokenizer.word_counts
min_count = 3  # En az 3 kez geçmeli
words_to_keep = [word for word, count in word_counts.items() if count >= min_count]
tokenizer = Tokenizer(num_words=len(words_to_keep), oov_token='<OOV>')
tokenizer.fit_on_texts(X)

X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, maxlen=150)

# Etiketleri sayisal hale donustur
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Veriyi train ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Validation set için 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

def build_simple_model(vocab_size=8000, maxlen=150, num_classes=20):
    
    model = Sequential()
    
    # Embedding layer - daha küçük boyut
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen))  
    
    # Daha yüksek dropout
    model.add(SpatialDropout1D(0.4))  
    
    # Daha küçük LSTM
    model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.4, return_sequences=False))
    
    # Batch Normalization
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Daha küçük dense layer
    model.add(Dense(24, activation='relu',
                   kernel_regularizer=regularizers.l2(0.01)))  
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Output layer
    model.add(Dense(num_classes, activation="softmax",
                    kernel_regularizer=regularizers.l2(0.01)))  
    
    # Daha düşük learning rate
    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    
    return model

model = build_simple_model(vocab_size=len(tokenizer.word_index) + 1)
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Test seti ile değerlendir
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Grafikleri çiz
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Overfitting analizi
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
gap = final_train_acc - final_val_acc
print("\nOverfitting analizi:")
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Accuracy Gap: {gap:.4f}")
print(f"Overfitting {'var' if gap > 0.1 else 'yok' if gap < 0.05 else 'hafif'}")