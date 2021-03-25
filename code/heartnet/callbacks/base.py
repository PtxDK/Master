from tensorflow.keras.callbacks import CSVLogger

class CSVEvaluateLogger(CSVLogger):
    def __init__(self, filename, separator=',', append=False):
        super().__init__(filename, separator=separator, append=append)
        
    def on_test_begin(self, logs):
        super().on_train_begin(logs=logs)
    
    def on_test_end(self, logs):
        super().on_epoch_end(0, logs=logs)
        super().on_train_end(logs=logs)