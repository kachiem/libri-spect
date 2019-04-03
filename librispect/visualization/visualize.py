from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output


class plotLoss(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.x_axis = []
        self.count = 0
        self.train_loss = []
        self.vali_loss = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x_axis.append(self.count)
        self.train_loss.append(logs.get('loss'))
        self.vali_loss.append(logs.get('val_loss'))
        self.count += 1

        clear_output(wait=True)
        plt.plot(self.x_axis, self.train_loss, label="training_loss")
        plt.plot(self.x_axis, self.vali_loss, label="validation_loss")
        plt.legend()
        plt.show()
