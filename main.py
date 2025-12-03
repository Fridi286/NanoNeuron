import data_loader as dl

train_images = dl.load_images() / 255   # normalization
train_labels = dl.load_labels()


