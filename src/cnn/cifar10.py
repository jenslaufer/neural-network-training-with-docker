from pymongo import MongoClient
import gridfs


class Training:

    def __init__(self, mongouri='mongodb://trainingdb', database='trainings'):
        print("start training...")
        self.mongouri = mongouri
        client = MongoClient(mongouri)
        self.db = client[database]
        self.fs = gridfs.GridFS(self.db)

        self.__load_images()

        print("training finalized.")

    def __load_images(self):
        print("to insert")
        test = self.db['test']
        test.insert_one({'name': 'jens'})


if __name__ == '__main__':
    Training()
