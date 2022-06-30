import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,TensorBoard
import time


OUTPUT_FOLDER='\\data_concreate\\img\\'

IMAGE_SHAPE=64

BATCH_SIZE = 32 
EPOCHS = 50


#Folder addres to keep tensorboard files
TENSORBOARD_LOGS="cement_binary-{}".format(int(time.time())) # log file name for tensorboard

EARLY_STOP = EarlyStopping(monitor='val_loss', patience=8, verbose=1,mode='auto')
LEARNING_RATE = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')
TENSORBOARD=TensorBoard(log_dir='./logs/{}'.format(TENSORBOARD_LOGS))

#Metrics to check while training
METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
   ]


#Generating datasets for training and test
datagen = ImageDataGenerator(rescale = 1./255.)

def generate_dataset(dataadres:str):
    """
    Function to create datasets 
     Args:
      dataadres (string) - local address of dataset (train, val or test) 
      
    """
    ds=datagen.flow_from_directory('./'+OUTPUT_FOLDER + dataadres,
                                                 target_size = (IMAGE_SHAPE,IMAGE_SHAPE),
                                                 batch_size = BATCH_SIZE,
                                                 shuffle=True,
                                                 class_mode = 'binary')
    return ds


def create_model():
    """
    Function to create models
      
    """

    inputs = tf.keras.Input(shape=(IMAGE_SHAPE,IMAGE_SHAPE,3))
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(16,activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


training_set = generate_dataset('train') 
val_set = generate_dataset('val')
test_set = generate_dataset('test')

model=create_model()

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(), # default from_logits=False
              metrics=METRICS)


def save_model():
    """
    Function to save model
      
    """
    model_json = model.to_json()
    with open("./model/modelbm.json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights("./model/modelbm.h5")



if __name__ == '__main__':
    history = model.fit(training_set, validation_data=val_set, 
                     epochs = EPOCHS,callbacks = [LEARNING_RATE, EARLY_STOP, TENSORBOARD])

    score = model.evaluate(test_set)
    save_model()

