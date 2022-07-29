<center> <h2> EMBA-Natural Language Processing (NLP) </h2> </center>

[GitHub Link](https://github.com/Youteng/Big-Data-Analytics-AI)

### Tensorflow dataset
Tensroflow data set (TFDS) is a python model that can help use to get datasets.
```
!pip install tensorflow-datasets
```

Before we started, make sure all the package we need is imported.
```
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
```

Try to load fashion_mnist dataset with TFDS then print the detailed information.
```
mnist_data, info = tfds.load("fashion_mnist", with_info='true')
print(info)
```

### Tokenization
First step of NLP, tokenization.
Tokenization is to create an vocabulary then fit wrods to indexes.
```
sentences = [
             'I love dog',
             'I love cat',
             'I love a cat',
             'I love a cat?'
]

tokenizer = Tokenizer(num_words=100) #分割句子為字元
tokenizer.fit_on_texts(sentences) #轉換字元為數字
wordIndex = tokenizer.word_index #獲取轉換結果
print(wordIndex) #印出轉換結果
```

Sometimes the received word might out of vocabulary (OOV), the OOV token can be set when constructing the tokenizer.
```
tokenizer = Tokenizer(num_words=100, oov_token='OOV') #設定‘OOV’
tokenizer.fit_on_texts(sentences)
sqeuences = tokenizer.texts_to_sequences(testData)
wordIndex = tokenizer.word_index
print(wordIndex)
print(sqeuences)
```

A huge vocabulary can be created to avoid OOV during traning.
```
#取得imdb_reviews資料集中train的部分
tranData = tfds.load('imdb_reviews', split='train')

#建立空list
imdbSentences = list()

#依序取出句子，轉換為string之後存進list
for item in tranData:
  imdbSentences.append(str(item['text']))

print(imdbSentences[0])

#Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdbSentences)

wordIndex = tokenizer.word_index
print(wordIndex)
```

### Sarcasm Detection Model
#### Model Training
The major subject of this course is to build a sarcasm detection model with tensorflow.
Data data pre-processing flow also included as well.

Parameters declaration
```
vocab_size = 1000 #文字總數
embedding_dim = 16 #Coefficients 數量
max_length = 120 #句子最大長度
trunc_type = 'post' #切割類型
padding_type = 'post' #填字類型
oov_tok = "<OOV>" #OOV
```

The provided .josn file should be uploaded to cloab.
![](https://i.imgur.com/PGkvIQU.png)



Loading dataset from .json
```
import json

#讀檔
with open("/content/sarcasmDataset.json", 'r') as f:
  data = json.load(f)

#建立lists
sentences = list()
labels = list()
urls = list()

#依序存取json block, 並將內容放至對應list
for item in data:
  sentence = item['headline'].lower()
  sentences.append(sentence)
  labels.append(item['is_sarcastic'])
  urls.append(item['article_link'])
```

```
#句子總數
print(len(sentences))
```

The data set should be split into training and testing.
```
#設定訓練集資料量
training_size = 24000

#取得前24000筆資料
training_sentences = sentences
#取得24000筆後所有資料
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

print(testing_sentences)
```

Tokenization
```
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
```
Padding the sentences for making sure all of that has the same length with function pad_sequences().

For instance:
![](https://i.imgur.com/b4Vtuyo.png)

```
from tensorflow.keras.preprocessing.sequence import pad_sequences

training_sequences = tokenizer.texts_to_sequences(training_sentences)
#將句子設定為固定長度, 句尾補0
training_padded = pad_sequences(training_sequences, maxlen=152)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(training_sequences)
```
Converting the data to the numpy array for training convenience.
```
#將資料轉換為numpy array
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
```
Creating a basic model.
```
#建立model
model = keras.Sequential(
    [
     keras.layers.Embedding(10000, 16),
     keras.layers.GlobalAveragePooling1D(),
     keras.layers.Dense(24, activation='relu'),
     keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

Training the created model.
```
#開始訓練model
history = model.fit(training_padded, training_labels,
          epochs=30, verbose=1,
          validation_data=(testing_padded, testing_labels))
```

Show the results with linecharts.
```
#將結果以line chart顯示
import matplotlib.pyplot as plt

#Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
![](https://i.imgur.com/o7HAd1E.png)


```
#儲存model
model.save('NLP_model_1.h5')
```

#### Model Improving
Five improving methods:
1. Increase the training epochs.
2. Add more hidden layers.
3. Use another NN for training.
4. Increase the amounts of data.
5. Improve the optimizer.

Increase the training epochs to 100 might improve the results.
```
#建立model
model = keras.Sequential(
    [
     keras.layers.Embedding(10000, 16),
     keras.layers.GlobalAveragePooling1D(),
     keras.layers.Dense(24, activation='relu'),
     keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

```
history = model.fit(training_padded, training_labels,
          epochs=100, verbose=1,
          validation_data=(testing_padded, testing_labels))
```

Overfitting.
![](https://i.imgur.com/XGTj949.png)

Add hedden layers.
```
#增加hidden layers
model = keras.Sequential(
    [
     keras.layers.Embedding(10000, 16),
     keras.layers.GlobalAveragePooling1D(),
     keras.layers.Dense(128, activation='relu'),
     keras.layers.Dense(64, activation='relu'),
     keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

```
history = model.fit(training_padded, training_labels,
          epochs=30, verbose=1,
          validation_data=(testing_padded, testing_labels))
```
![](https://i.imgur.com/XMmvSPu.png)

Try to use LSTM for improving.
```
model = keras.Sequential(
    [
     keras.layers.Embedding(10000, 16),
     keras.layers.Bidirectional(keras.layers.LSTM(16)), #改用LSTM
     keras.layers.Dense(24, activation='relu'),
     keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

```
history = model.fit(training_padded, training_labels,
          epochs=30, verbose=1,
          validation_data=(testing_padded, testing_labels))
```

Well...
![](https://i.imgur.com/l9pq4DD.png)

Increase the amounts of data.
```
#下載資料
!wget --no-check-certificate \
  https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip \
  - 0 /content/glove.zip
```

```
import zipfile

#解壓縮
f = '/content/glove.twitter.27B.zip'
zip = zipfile.ZipFile(f, 'r')
zip.extractall('/content')
zip.close()
```
```
glove_embdding = dict()
#讀取檔案, 並將文字與coefficients取出
with open('/content/glove.twitter.27B.25d.txt', 'r') as f:
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embdding[word] = coefs
```
```
#印出結果
print(glove_embdding['frog'])
```
```
#設定文字數量
vocab_size = 13200
#設定embedding輸出數量
embedding_dim = 25

#預先建立空的array
embedding_matrix = np.zeros((vocab_size, embedding_dim))

#取出coefficients並存進array
for word, index in tokenizer.word_index.items():
  if index > vocab_size - 1:
    break
  else:
    embedding_vector = glove_embdding.get(word)
    if embedding_vector is not None:
      embedding_matrix[index] = embedding_vector
```
```
#建立model, 使用兩層LSTM
model = keras.Sequential(
    [
     keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False),
     keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, return_sequences=True)),
     keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim)),
     keras.layers.Dense(24, activation='relu'),
     keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```
```
history = model.fit(training_padded, training_labels,
          epochs=30, verbose=1,
          validation_data=(testing_padded, testing_labels))
```
Still overfitting...
![](https://i.imgur.com/fXmTZ6o.png)

Improve the optimizer.
![](https://i.imgur.com/TMpVOrV.png)


```
model = keras.Sequential(
    [
     keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False),
     keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, return_sequences=True)),
     keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim)),
     keras.layers.Dense(16, activation='relu'),
     keras.layers.Dense(1, activation='sigmoid')
    ]
)

#調整optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
```
```
history = model.fit(training_padded, training_labels,
          epochs=30, verbose=1,
          validation_data=(testing_padded, testing_labels))
```
learning_rate
![](https://i.imgur.com/k2XtPvN.png)

50 epochs results
![](https://i.imgur.com/mSjIRZZ.png)

#### Model Testing

```
#讀取model
import keras

model = keras.models.load_model('/content/NLP_model_1.h5')
```

```
#建立測試字串
test_sequences = [
                  "Today is a sunny day",
                  "It was, For, Uh, Medcial Reasons, Says Doctor To Boris Johnson, Explaining Why They Had To Give Hime Hairvut",
                  "Pokémon Go player stabbed, keeps playing",
                  "thirtysomething scientists unveil doomsday clock of hair loss"]
```
```
#Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(test_sequences)

word_index = tokenizer.word_index
```
```
#Pad squences
from tensorflow.keras.preprocessing.sequence import pad_sequences

testing_sequences = tokenizer.texts_to_sequences(test_sequences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
```
```
#判斷結果
model.predict(testing_padded)
```

    

