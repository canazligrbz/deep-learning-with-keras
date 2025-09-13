import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

import keras_tuner as kt

from kerastuner.tuners import RandomSearch

from sklearn.metrics import classification_report, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

num_words=10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

maxlen=200
x_train= pad_sequences(x_train, maxlen=maxlen)
x_test= pad_sequences(x_test, maxlen=maxlen)

def build_model(hp): # hp: hyperparameter
    
    model = Sequential() # base model
    
    # embedding katmani: kelimeleri vektorlere cevirir
    model.add(Embedding(input_dim = 10000,
                        output_dim = hp.Int("embedding_output", min_value=32, max_value = 128, step = 32), # vektor boyutlari (32, 64, 96, 128 olabilir)
                        input_length = maxlen))
    # simpleRNN: rnn katmani
    model.add(SimpleRNN(units = hp.Int("rnn_units", min_value = 32, max_value = 128, step = 32))) # rnn hucre sayisi 32, 64, 96, 128 olabilir
    
    # dropout katmani: overfittingi engellemek icin rasgele bazi cell'leri kapatir
    model.add(Dropout(rate = hp.Float("dropout_rate", min_value = 0.2, max_value = 0.5, step = 0.1))) # 0.2, 0.3, 0.4, 0.5
    
    # cikti katmani: 1 cell ve sigmoid
    model.add(Dense(1, activation="sigmoid")) # sigmoid activaiton : ikili siniflandirma icin kullanilir. (cikti: 0 yada 1 arasinda olur)
    
    # modelin compile edilmesi
    model.compile(optimizer = hp.Choice("optimizer", ["adam", "rmsprop"]), # adam veya rmsprop kullanilabilir
                  loss = "binary_crossentropy", # ikili siniflandirma icin kullanilan loss fonksiyonu
                  metrics = ["accuracy", "AUC"] # AUC: area under curve
                  )
    return model

# hyperparameter search: random search ile hiperparametre aranacak
tuner = RandomSearch(
    build_model, # optimize edilecek model fonksiyonu
    objective = "val_loss", # val_loss en dusuk olan en iyisidir, val_accuracy: yuksek olan en iyisidir.
    max_trials=2, # 2 farkli model deneyecek
    executions_per_trial = 1, # her model icin 1 egitim denemesi
    directory = "rnn_tuner_directory", # modellerin kayit edilecegi dizin
    project_name= "imdb_rnn" # projenin adi
    )

# erken durdurma: dogrula hatasi duzelmezse (azalmazsa) egitimi durdur
early_stopping = EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True)

# modelin egitimi
tuner.search(x_train, y_train,
             epochs = 2, 
             validation_split = 0.2, # egitim veri setinin %20 si validation olacak
             callbacks = [early_stopping] 
             )

# en iyi modelin alinmasi
best_model = tuner.get_best_models(num_models=1)[0] # en iyi performans gosteren model

# en iyi modeli kullanarak test et
loss, accuracy, auc_score = best_model.evaluate(x_test, y_test) 
print(f"Test loss: {loss}, test accuracy: {accuracy:.3f}, test auc: {auc_score:.3f}")

# tahmin yapma ve modelin performansini degerlendirme
y_pred_prob = best_model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype("int32") # tahmin edilen degerler 0.5 den buyukse 1 e yuvarlanir yani olumlu olur

print(classification_report(y_test, y_pred))

# roc egrisi hesaplama
fpr, tpr, _ = roc_curve(y_test, y_pred_prob) # roc egrisi icin fpr (false positive rate) ve tpr (true positive rate)

roc_auc = auc(fpr, tpr) # roc egrsiinin altinda kalan alan hesaplanir.

# roc egrisi gorsellestirme
plt.figure()
plt.plot(fpr, tpr, color = "darkorange", lw = 2, label = "ROC Curve (area = %0.2f)" % roc_auc) # roc curve
plt.plot([0,1],[0,1], color = "blue", lw = 2, linestyle = "--") # rasgele tahmin cizgisi
plt.xlim([0,1])
plt.ylim([0, 1.05]) 
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()






