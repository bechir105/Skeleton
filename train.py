import csv
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from data import BodyPart 
import tensorflow as tf
import tensorflowjs as tfjs
from sklearn.preprocessing import LabelEncoder
import numpy as np

tfjs_model_dir = 'model'


# loading final csv file
def load_csv(csv_path):
    le = LabelEncoder()


    df = pd.read_csv(csv_path, header=0)
    df.drop(['file_name'],axis=1, inplace=True)
    classes = df.pop('class_name')

    labels = le.fit_transform(classes)
    y = labels
    print(f'labels: {np.unique(classes)}')
    print(f'classifiers: {np.unique(y)}')
    
    X = df.astype('float64')
    y = keras.utils.to_categorical(y)
    
    return X, y, classes.unique()

def create_model():
        inputs = tf.keras.Input(shape=(132))
        layer = keras.layers.Dense(128, activation=tf.nn.relu6)(inputs)
        layer = keras.layers.Dropout(0.5)(layer)
        layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
        layer = keras.layers.Dropout(0.5)(layer)
        outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

        model = keras.Model(inputs, outputs)


        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

X, y, class_names = load_csv('train_data.csv')
if __name__ == '__main__':
    X, y, class_names = load_csv('train_data.csv')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    X_test, y_test, _ = load_csv('test_data.csv')


    processed_X_train = X_train
    processed_X_val =  X_val
    processed_X_test = X_test

    model = create_model()

    # Add a checkpoint callback to store the checkpoint that has the highest
    # validation accuracy.
    checkpoint_path = "weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                mode='max')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                patience=20)

    # Start training
    print('--------------TRAINING----------------')
    history = model.fit(processed_X_train, y_train,
                        epochs=200,
                        batch_size=16,
                        validation_data=(processed_X_val, y_val),
                        callbacks=[checkpoint, earlystopping])


    print('-----------------EVAUATION----------------')
    loss, accuracy = model.evaluate(processed_X_test, y_test)
    print('LOSS: ', loss)
    print("ACCURACY: ", accuracy)


    tfjs.converters.save_keras_model(model, tfjs_model_dir)
    print('tfjs model saved at ',tfjs_model_dir)