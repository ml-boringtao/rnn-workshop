import h5py
import os

class HDF5DatasetWriter: 
    def __init__(self, xdims, ydims, outputPath, bufSize=1000):
        try:
            if os.path.exists(outputPath):
                os.remove(outputPath)
        except: 
            pass
                       
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset("images", xdims, dtype="float")
        self.labels = self.db.create_dataset("labels", ydims, dtype=h5py.special_dtype(vlen=str))
        
        self.bufSize = bufSize
        self.buffer = {"images": [], "labels": []}
        self.idx = 0
        
    def add(self, rows, labels): 
        self.buffer["images"].extend(rows)
        self.buffer["labels"].extend(labels)
        if len(self.buffer["images"]) >= self.bufSize:
            self.flush()
            
    def flush(self): 
        i = self.idx + len(self.buffer["images"])
        self.data[self.idx:i] = self.buffer["images"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i 
        self.buffer = {"images": [], "labels": []}
    
    def storeClassLabels(self, classLabels): 
        dt = h5py.special_dtype(vlen=str) 
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt) 
        labelSet[:] = classLabels
    
    def close(self): 
        if len(self.buffer["images"]) > 0: 
            self.flush() 
        self.db.close()