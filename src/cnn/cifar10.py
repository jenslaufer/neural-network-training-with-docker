from pymongo import MongoClient
import gridfs
import numpy as np
import scipy
from scipy import misc
import os
import io
from math import ceil
from tempfile import TemporaryFile

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D
from keras.datasets import cifar10
from keras.applications.inception_v3 import InceptionV3, preprocess_input


import json
import datetime


class Training:

    def __init__(self, mongouri, database, collection, session_id,
                 loss, optimizer, batch_size, epochs, subset_pct):
        print("training...")
        client = MongoClient(mongouri)

        self.session_id = session_id
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.loss = loss
        self.optimizer = optimizer

        self.subset_pct = subset_pct
        self.bottleneck_features_filename = "cifar10_bottleneck_features"
        self.db = client[database]
        self.collection = db[collection]
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
        print(self.training_features.shape[1:])
        model = Sequential()
        model.add(Conv2D(filters=100, kernel_size=2,
                         input_shape=self.training_features.shape[1:]))
        model.add(Dropout(0.4))
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))
        self.model = model
        self.model.summary()

        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

        filename = 'model_arch.hdf5'
        self.model.save(filename)

        with open(filename, 'rb') as f:
            self.fs.put(f.read(), filename=filename,
                        type='model_arch', session_id=self.session_id, contentType="application/x-hdf")
        os.remove(filename)

    def __train_shallow_neural_network(self):
        filename = 'model_weights.hdf5'

        checkpointer = ModelCheckpoint(filepath=filename,
                                       verbose=1, save_best_only=True)
        history = self.model.fit(self.training_features, self.y_train, batch_size=self.batch_size,
                                 epochs=self.epochs, validation_split=0.2, callbacks=[checkpointer],
                                 verbose=2, shuffle=True)

        out = io.StringIO(json.dumps(history.history))
        self.fs.put(out.getvalue(), filename="training_history.json",
                    type='training_history', session_id=self.session_id,
                    encoding="utf-8", contentType="text/json")

        with open(filename, 'rb') as f:
            self.fs.put(f.read(), filename=filename,
                        type='model_weights', session_id=self.session_id, contentType="application/x-hdf")
        os.remove(filename)

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
            self.fs.put(outfile.getvalue(), filename=filename,
                        type='bottleneck_features',  contentType="application/x-hdf")

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

        self.collection.update_one({'_id': self.session_id},  {
            '$set': {'test_sample_size': len(self.x_test),
                     'train_sample_size': len(self.x_train)}})

    def __load_weights(self):
        filename = 'model_weights.hdf5'
        session = db['fs.files'].find_one(
            {'session_id': self.session_id, 'filename': filename})
        with open(filename, 'wb') as f:
            f.write(self.fs.get(session['_id']).read())
        self.model.load_weights(filename)
        os.remove(filename)

    def __test_model(self):
        self.__load_weights()

        # evaluate test accuracy
        score = self.model.evaluate(self.test_features, self.y_test, verbose=0)
        accuracy = score[1]

        self.collection.update_one({'_id': self.session_id}, {
            '$set': {'accuracy': accuracy, "date": datetime.datetime.utcnow()}})

        # print test accuracy
        print('Test accuracy: %.2f%%' % (100 * accuracy))


if __name__ == '__main__':
    mongouri = "mongodb://trainingdb"
    database = "trainings"
    collection = "sessions"

    client = MongoClient(mongouri)
    db = client[database]

    sessions = list(db[collection].find({"accuracy": {'$exists': 0}}))

    [Training(mongouri, database, collection, session['_id'],
              session['loss'], session['optimizer'], session['batch_size'],
              session['epochs'], session['subset_pct']) for session in sessions]
