# ml2_project

This repository contains a trained model for a three-class image classification in this specific task. You could extend the model with the other 98 classes of food but due to the limited ressources I would suggest to use only three classes and if you are familiar with the notebook and have the ressources feel free to extend the model-classes.Of course you could reduce the images per class but I decided to use three classes with all the availiable images. The model was trained using TensorFlow and Keras. The purpose of this README file is to provide an overview of the model and how it was trained. 

## Installation
First, you've to import the following packages:
```
import tensorflow as tf
import matplotlib.image as img
%matplotlib inline
import numpy as np
from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU
from tensorflow import keras
from tensorflow.keras import models
import cv2
```

Then you can download and extract the data with this code:
```
# Helper function to download data and extract
def get_data_extract():
  if "food-101" in os.listdir():
    print("Dataset already exists")
  else:
    print("Downloading the data...")
    !wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
    print("Dataset downloaded!")
    print("Extracting data..")
    !tar xzvf food-101.tar.gz
    print("Extraction done!")

# Download data and extract it to folder
# Uncomment this below line if you are on Colab

get_data_extract()
```
Note: That the code above contains not all the steps for preparing the data.

## Model Architecture:

The model architecture is based on the InceptionV3 convolutional neural network (CNN) architecture, pre-trained on the ImageNet dataset. The last fully connected layer of the InceptionV3 model was replaced with custom layers for the specific classification task.

The modified architecture consists of the following layers:

Input Layer: The input shape of the images is (299, 299, 3).

Global Average Pooling Layer: This layer is used to reduce the spatial dimensions of the feature maps.

Dense Layer 1: A fully connected layer with 128 units and ReLU activation function.

Dropout Layer 1: A dropout layer with a rate of 0.2 to prevent overfitting.

Dense Layer 2: A fully connected layer with 128 units.

Leaky ReLU Layer: A leaky ReLU activation layer with an alpha value of 0.2.

Dropout Layer 2: A dropout layer with a rate of 0.3.

Output Layer: A dense layer with 3 units (corresponding to the three classes) and a softmax activation function for multi-class classification.

## Training Configuration:

The model was trained using the following configuration:

Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.0001 and momentum of 0.9.
Loss Function: Categorical Crossentropy.
Regularization: L2 regularization with a weight decay of 0.005 was applied to the output layer.
Training Data: The training data was sourced from the 'train_mini' directory.
Validation Data: The validation data was sourced from the 'test_mini' directory.
Data Augmentation: Data augmentation techniques such as rescaling, shear range, zoom range, and horizontal flip were applied to the training data using the ImageDataGenerator class.
Batch Size: The batch size was set to 16.
Number of Training Samples: There were 2250 training samples.
Number of Validation Samples: There were 750 validation samples.
Training Epochs: The model was trained for 5 epochs.
Saving the Model:

The trained model was saved in the HDF5 file format with the name 'model_trained_3class.hdf5'. The model weights were saved during training if they showed improvement on the validation set using the ModelCheckpoint callback.

Please note that I have used different layers and number of epochs and have had the best results with these layers and either 2 or 5 epochs of training. The highest number of epochs with which I trained the model was 5, as one epoch is already very time intensive. But feel free to increase the number of epochs to probablly achieve an even better result.

## Training History:

The training history, including the loss and accuracy values for each epoch, was logged to the file 'history_3class.log' using the CSVLogger callback.

Please note that this repository does not include the actual training and validation data. The provided code and information serve as a guide to understand the model architecture and training process. You will find further the details in the notebook provided in this repository.

Code for training the model:
```
K.clear_session()
n_classes = 3
img_width, img_height = 299, 299
train_data_dir = 'train_mini'
validation_data_dir = 'test_mini'
nb_train_samples = 2250 #75750
nb_validation_samples = 750 #25250
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128)(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.3)(x)

predictions = Dense(3,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_3class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_3class.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=5,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('model_trained_3class.hdf5')
```

## Prediction
This code defines the predictions function for the model:
```
def predict_class(model, images, show = True):
  predictions = []  # Leere Liste zum Speichern der Vorhersagen
  for img in images:
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]

    predictions.append(pred_value)  # Vorhersage zur Liste hinzufügen
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        plt.show()
        
    return predictions
```

 And with this code we print the prediction output:
```

predictions = predict_class(model_best, images, True)


for prediction in predictions:
    print(prediction)
```

## After the training we use GPT 

After the training and validation of the model I used the output to get some more information about the food like the nutritional values and some recipes for the predicted food. So the goal was to have one model which detects the food and then to use gpt for some futher information.

### Prerequisites
Before running the code, make sure you have the following:

Python 3 installed
OpenAI Python library (openai) installed
API key for the OpenAI GPT-3 model (Hit me up if you don't have your own Open AI API Key: stammnoah@students.zhaw.com)

### First approach to use gpt with sarcasm
As a small extension, I tried to add sarcasm to the answer first with gpt 3.5 turbo to make the answer a little more humorous. For this I have already given the model a small conversation and tried to steer it in the right direction.
```
import requests
import json





url = 'https://api.openai.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}



def generate_recipes():
    data = {
        'messages': [{'role': 'user', 'content': 'How many pounds are in a kilogram?'},
        {'role': 'assistant', 'content': 'This again? There are 2.2 pounds in a kilogram. Please make a note of this.'},
        {'role': 'user', 'content': 'What does HTML stand for?'},
        {'role': 'assistant', 'content': 'Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.'},
        {'role': 'user', 'content': 'When did the first airplane fly?'},
        {'role': 'assistant', 'content': 'On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they would come and take me away.'},
        {'role': 'user', 'content': 'What is the meaning of life?'},
        {'role': 'assistant', 'content': 'Im not sure. I will ask my friend Google.'},
        {'role': 'user', 'content': f'Can you give me two recipes examples for {food}?'}],
        'model': 'gpt-3.5-turbo',
        'temperature': 0.5,
        'max_tokens': 2000,
        'n': 1,
        'stop': None,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    print(response_json)
    return response_json['choices'][0]['message']['content'].strip()

recipe = generate_recipes()
print(recipe)


```

### Second approach to use gpt with sarcasm
The answer from gpt 3.5 turbo, however, unfortunately remained factual. So I tried a slightly older model from open ai davinci. This time I didn't just give him a conversation, I already made the conversation very specific to the topic at hand, in what way I imagined the answer would be.

```
import os
import openai

openai.api_key = api_key

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=f'Marv is a chatbot that reluctantly answers questions with sarcastic responses:\n\nYou: How many pounds are in a kilogram?\nMarv: This again? There are 2.2 pounds in a kilogram. Please make a note of this.\nYou: What is a recipe?\nMarv: Was Google too busy? It is for people who do not know how to cook. Basically, it is a guide for preparing food\nYou: Can you give me an example of an recipe?\nMarv: Yes, logically I can. But you can also buy a cookbook on Google. Otherwise try this:100g flour, 2 eggs, 10g baking powder, 50g butter\nYou:Are you able to give me two recipes for oats?\nMarv: Yes, but I would advise you to go to a restaurant instead of cooking yourself.\nYou:Are you able to give me two recipes for {food}?\nMarv:',
  temperature=0.5,
  max_tokens=500,
  top_p=0.3,
  frequency_penalty=0.5,
  presence_penalty=0.0
)

generated_text = response.choices[0].text
print(generated_text)
```


### Use gpt for the nutritional values
But here, too, the answer remained very matter-of-fact. I have found that when a very simple question is asked without much context, the answer is sarcastic, but not for such a specific question as: Can you give me a recipe for the following foodstuff?

In order to get an appropriate sarcastic answer for every question, I would either have to finetune gpt or directly train a model from scratch in sarcasm.

To finish the project, I asked GPT one more time for the nutritional values.
```
import requests
import json





url = 'https://api.openai.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}



def generate_nutrition(food):
    data = {
        'messages': [{'role': 'user', 'content': f'Can you give me the average nutritional values of {food} for 100 grams??'}],
        'model': 'gpt-3.5-turbo',
        'temperature': 0.5,
        'max_tokens': 256,
        'n': 1,
        'stop': None,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    return response_json['choices'][0]['message']['content'].strip()


values = generate_nutrition(food)
print(values)
```

### Results
With the following code you can print the results in summary form.
```
print(f'Your food has the following nutritional values:\n{values} \nAnd here are some recipes ideas:\n{recipe}' )
```
