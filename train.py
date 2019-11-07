#import plaidml.keras
#plaidml.keras.install_backend()
from data_utils import get_imgs_masks, resize_imgs_masks
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from unet import custom_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from metrics import iou, iou_np
from utils import plot_segm_history
import os
import numpy as np
from callbacks import TestResults
import plaidml.keras
from generators import PatchGenerator

def train(num_classes, num_layers, path, epochs, batch_size, patch_size, show_history=True):

    x, y = get_imgs_masks(path)
    print("Found {num} images and {num1} masks".format(num=len(x), num1=len(y)))

    x = np.asarray(x, dtype=np.float32) / 255 
    y = np.asarray(y, dtype=np.uint8)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=0)
    print('Train data size: {} images and {} masks'.format(x_train.shape[0], y_train.shape[0]))
    print('Validation data size: {} images and {} masks'.format(x_val.shape[0], y_val.shape[0]))

    aug_factor = 1
    print(x_train.shape)
    steps_per_epoch = np.ceil((x_train.shape[0] * x_train.shape[1] * x_train.shape[2] * aug_factor) / (batch_size * patch_size * patch_size)).astype('int')
    print('Steps per epoch: {}'.format(steps_per_epoch))

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # train_gen = ImageDataGenerator(
    #     featurewise_center=False,
    #     featurewise_std_normalization=False,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True,
    #     vertical_flip=True
    # )

    input_shape = (patch_size, patch_size, 3)

    model = custom_unet(
        input_shape,
        num_classes=num_classes,
        filters=16,
        use_batch_norm=True,
        dropout=0.0,
        dropout_change_per_layer=0.0,
        num_layers=num_layers,
        output_activation='softmax'
    )

    model_filename = 'model_v1.h5'
    callback_checkpoint = ModelCheckpoint(
        model_filename, 
        verbose=1, 
        monitor='val_loss', 
        save_best_only=True,
    )

    model.compile(
        optimizer=Adam(), 
        loss = 'categorical_crossentropy',
        metrics=[iou]
    )

    callback_test = TestResults(
        images=x_val, 
        masks=y_val, 
        model=model, 
        n_classes=num_classes,
        batch_size=batch_size,
        patch_size=patch_size,
        offset=2 * num_layers,
        output_path='output'
    )

    train_generator = PatchGenerator(images=x_train, masks=y_train, patch_size=patch_size, batch_size=batch_size)
    valid_generator = PatchGenerator(images=x_val, masks=y_val, patch_size=patch_size, batch_size=batch_size)
    steps_per_epoch = 1
    history = model.fit_generator(
        iter(train_generator),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=iter(valid_generator),
        validation_steps=steps_per_epoch,
        callbacks=[callback_checkpoint, callback_test]
    )

    if show_history:
        plot_segm_history(history)

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "input", "dataset", "*_NEW.png")
    train(num_classes=4, num_layers=3, epochs=2, path=path, batch_size=8, patch_size=512)