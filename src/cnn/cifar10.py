from pymongo import MongoClient
import gridfs
import keras
from keras.datasets import cifar10
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import scipy
from scipy import misc
import os
import io
from math import ceil
from tempfile import TemporaryFile

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D


class Training:

    def __init__(self, mongouri='mongodb://trainingdb', database='trainings', subset_pct=1.0):
        print("training...")
        client = MongoClient(mongouri)
        self.subset_pct = subset_pct
        self.bottleneck_features_filename = "cifar10_bottleneck_features"
        self.db = client[database]
        self.fs = gridfs.GridFS(self.db)

        self.__load_images()
        self.__init_base_model()
        self.__extract_bottleneck_features()
        self.__init_shallow_neural_network()
        self.__train_shallow_neural_network()
        self.__test_model()

        print("training.")

    def __init_base_model(self):
        print("load model...")
        self.base_model = InceptionV3(weights='imagenet',
                                      include_top=False, input_shape=(139, 139, 3))
        print('model loaded.')

    def __init_shallow_neural_network(self):
        model = Sequential()
        model.add(Conv2D(filters=100, kernel_size=2,
                         input_shape=self.training_features.shape[1:]))
        model.add(Dropout(0.4))
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        self.model = model

    def __train_shallow_neural_network(self):

        checkpointer = ModelCheckpoint(filepath='model.best.hdf5',
                                       verbose=1, save_best_only=True)
        self.model.fit(self.training_features, self.y_train, batch_size=50, epochs=50,
                       validation_split=0.2, callbacks=[checkpointer],
                       verbose=2, shuffle=True)

    def __get_bottleneck_features(self, x, dataset_type):
        filename = "{}_{}_{}.npz".format(
            self.bottleneck_features_filename, len(x), dataset_type)
        try:
            data = io.BytesIO(self.fs.get_last_version(filename).read())
            features = np.load(data)['features']
        except Exception as e:
            print('bottleneck features file not detected (train)')
            print('calculating now for {} images...'.format(len(x)))
            big_x = np.array([scipy.misc.imresize(x[i], (139, 139, 3))
                              for i in range(0, len(x))]).astype('float32')
            inception_input = preprocess_input(big_x)
            print('data preprocessed')
            features = self.base_model.predict(inception_input)
            features = np.squeeze(features)

            outfile = io.BytesIO()
            np.savez(outfile, features=features)
            self.fs.put(outfile.getvalue(), filename=filename)

        return features

    def __extract_bottleneck_features(self):
        self.training_features = self.__get_bottleneck_features(
            self.x_train, "train")
        self.test_features = self.__get_bottleneck_features(
            self.x_test, "test")

    def __load_images(self):
        (self.x_train, self.y_train), (self.x_test,
                                       self.y_test) = cifar10.load_data()
        self.x_train = self.x_train[:ceil(self.subset_pct * len(self.x_train))]
        self.y_train = self.y_train[:ceil(self.subset_pct * len(self.y_train))]
        self.x_test = self.x_test[:ceil(self.subset_pct * len(self.x_test))]
        self.y_test = self.y_test[:ceil(self.subset_pct * len(self.y_test))]

        self.y_train = keras.utils.to_categorical(
            np.squeeze(self.y_train), num_classes=10)
        self.y_test = keras.utils.to_categorical(
            np.squeeze(self.y_test), num_classes=10)

    def __test_model(self):
        self.model.load_weights('model.best.hdf5')

        # evaluate test accuracy
        score = self.model.evaluate(self.test_features, self.y_test, verbose=0)
        accuracy = 100*score[1]

        # print test accuracy
        print('Test accuracy: %.2f%%' % accuracy)


if __name__ == '__main__':
    Training(mongouri="mongodb://localhost:27018", subset_pct=1)
