#1. Instala as dependências e inicializa
import os
import tensorflow as tf
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

# Caminho completo para a pasta 'data'
data_dir = r"C:\Users\Z004N5ZE\Documents\2023.2\tcc1\exemplos\exe_tf_keras\data"
data_dir = 'data' 
image_exts = ['jpeg','jpg', 'bmp', 'png']

# Lista todos os arquivos na pasta 'data'
files = os.listdir(os.path.join(data_dir, 'happy'))
print("Files in 'happy':", files)

#1.5 Visualiza as imagens 
image_array = cv2.imread(os.path.join(data_dir,'happy','154006829.jpg'))
print("Image array:", image_array)
print("Shape of the image:", image_array.shape)
plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
plt.show()

#2. Remove imagens ruins
for image_class in os.listdir(data_dir):
    print("Processing images in class:", image_class)
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        print("Processing image:", image_path)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
           # print("Image type:", tip)
            if tip not in image_exts: 
                print('Image not in ext list, removing:', image_path)
                os.remove(image_path)
            else:
                print('Image is in ext list')
        except Exception as e: 
           print('Issue with image, consider removing:', image_path)

#3.Carrega os dados
data = tf.keras.preprocessing.image_dataset_from_directory(data_dir)
print(data)

#4.Refaz a escala das imagens
data = data.map(lambda x,y: (x/255, y))

# Encontra o valor maximo e minimo do lote de imagens
for images, labels in data.take(1):
    min_val = np.min(images)
    max_val = np.max(images)
    print("Valor mínimo do lote de imagens:", min_val)
    print("Valor máximo do lote de imagens:", max_val)

# Cria um lote de imagens
for images, labels in data.take(1):
    batch = (images, labels)

# Representa as imagens como um array numpy
print(batch[0].shape)
print(batch[1])
# Plota as primeiras 4 imagens do lote
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(str(batch[1][idx].numpy()))
plt.show()
#printa quantos batchs a pasta data tem 
print(len(data))

#Divide a pasta data em imagens de treinamento, validacao e teste
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

print(train_size)

#Divide quantos batchs para o treinamento, validacao e teste
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
print(len(train))

#5 Construção da CNN

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# Treina o modelo

logdir = r'C:\Users\Z004N5ZE\Documents\2023.2\tcc1\exemplos\exe_tf_keras\logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Plota a performance

fig, axs = plt.subplots(2)

# Plota a função de perda 
axs[0].plot(hist.history['loss'], color='teal', label='loss')
axs[0].plot(hist.history['val_loss'], color='orange', label='val_loss')
axs[0].set_title('Loss')
axs[0].legend(loc="upper left")

# Plota a acuracia
axs[1].plot(hist.history['accuracy'], color='teal', label='accuracy')
axs[1].plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
axs[1].set_title('Accuracy')
axs[1].legend(loc="upper left")

plt.show()

#6. Avalia o modelo
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

#7. Faz um teste com uma imagem nova
img = cv2.imread(r'c:\Users\Z004N5ZE\Documents\2023.2\tcc1\exemplos\exe_tf_keras\sadtest.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

resize.shape 
np.expand_dims(resize, 0).shape
yhat = model.predict(np.expand_dims(resize/255, 0))
print('yhat')
if yhat > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')

#8. Salva o modelo

model.save(os.path.join(r'c:\Users\Z004N5ZE\Documents\2023.2\tcc1\exemplos\exe_tf_keras\models','happysadmodel.h5'))
new_model = load_model(os.path.join(r'c:\Users\Z004N5ZE\Documents\2023.2\tcc1\exemplos\exe_tf_keras\models','happysadmodel.h5'))
yhatnew = new_model.predict(np.expand_dims(resize/255, 0))
if yhatnew > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')