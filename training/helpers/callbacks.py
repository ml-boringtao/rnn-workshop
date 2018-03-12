from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np 
import json
import os
from .utils import Utils

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
    
    def on_train_begin(self, logs={}): 
        self.history = {} 
        
        if self.jsonPath is not None: 
            if os.path.exists(self.jsonPath): 
                self.history = json.loads(open(self.jsonPath).read())
                
                if self.startAt > 0: 
                    for key in self.history.keys(): 
                        self.history[key] = self.history[key][:self.startAt]
                        
    def on_epoch_end(self, epoch, logs={}): 
        for (key, value) in logs.items(): 
            logs = self.history.get(key, [])
            logs.append(value)
            self.history[key] = logs
        if self.jsonPath is not None: 
            file = open(self.jsonPath, "w")
            file.write(json.dumps(self.history)) 
            file.close()
        if len(self.history["loss"]) > 1: 
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(Utils.smooth_curve(self.history['loss']), label="train_loss")
            plt.plot(Utils.smooth_curve(self.history['val_loss']), label="val_loss")
            plt.plot(Utils.smooth_curve(self.history['acc']), label="train_acc")
            plt.plot(Utils.smooth_curve(self.history['val_acc']), label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.history["loss"])))
            plt.legend()
            
            plt.savefig(self.figPath)
            plt.close()