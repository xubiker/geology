import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

import config
from metrics import iou_tf
from unet import res_unet, weightedLoss
from utils.base import MaskLoadParams, prepare_experiment
from utils.callbacks import TestCallback, TestCallbackPolarized
from utils.generators import AutoBalancedPatchGenerator, SimpleBatchGenerator, AutoBalancedPatchGeneratorPolarized
from utils.patches import combine_from_patches, split_into_patches

def set_gpu(gpu_index):
	import tensorflow as tf
	physical_devices  = tf.config.experimental.list_physical_devices('GPU')
	print(f'Available GPUs: {len(physical_devices )}')
	if physical_devices:
		print(f'Choosing GPU #{gpu_index}')
		try:
			tf.config.experimental.set_visible_devices([physical_devices[gpu_index]], 'GPU')
			logical_devices = tf.config.list_logical_devices('GPU')
			assert len(logical_devices) == 1
			print(f'Success. Now visible GPUs: {len(logical_devices)}')
		except RuntimeError as e:
			print('Something went wrong!')
			print(e)


class GeoModel:

    def __init__(self, n_pol, patch_size, batch_size, offset, n_classes, LR, patch_overlay, class_weights=None):
        self.n_pol = n_pol
        self.patch_s = patch_size
        self.batch_s = batch_size
        self.offset = offset
        self.n_classes = n_classes
        self.model = None
        self.LR = LR
        self.patch_overlay = patch_overlay
        self.class_weights = class_weights
        if self.class_weights is None:
            self.class_weights = [1 / self.n_classes] * self.n_classes

    def initialize(self, n_filters, n_layers):
        self.model = res_unet(
            # (None, None, 3),
            (None, None, 3 * (self.n_pol + 1)),
            n_classes=self.n_classes,
            BN=True,
            filters=n_filters,
            n_layers=n_layers,
        )

        # self.model.compile(
        #     optimizer=Adam(learning_rate=self.LR), 
        #     loss = weightedLoss(categorical_crossentropy, self.class_weights),
        #     metrics=[iou_tf]
        # )

    def load(self, model_path):
        # self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.load_weights(model_path)

    def predict_image(self, img: np.ndarray):
        patches = split_into_patches(img, self.patch_s, self.offset, self.patch_overlay)
        init_patch_len = len(patches)

        while (len(patches) % self.batch_s != 0):
            patches.append(patches[-1])
        pred_patches = []

        for i in range(0, len(patches), self.batch_s):
            batch = np.stack(patches[i : i+self.batch_s])
            prediction = self.model.predict_on_batch(batch)
            for x in prediction:
                pred_patches.append(x)
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_from_patches(pred_patches, self.patch_s, self.offset, self.patch_overlay, img.shape[:2])
        return result
    
    def predict_image_polarized(self, img: np.ndarray):
        patches = split_into_patches(img, self.patch_s, self.offset, self.patch_overlay) 
        init_patch_len = len(patches)

        while (len(patches) % self.batch_s != 0):
            patches.append(patches[-1])
        pred_patches = []

        for i in range(0, len(patches), self.batch_s):
            batch = np.stack(patches[i : i+self.batch_s])
            prediction = self.model.predict_on_batch(batch)
            for x in prediction:
                pred_patches.append(x)
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_from_patches(pred_patches, self.patch_s, self.offset, self.patch_overlay, img.shape[:2])
        return result

    def train(
        self, train_gen, val_gen, n_steps, epochs, val_steps,
        test_img_folder: Path, test_mask_folder: Path, test_img_folder_polarized: Path, test_mask_folder_polarized: Path, test_output: Path,
        codes_to_lbls, lbls_to_colors, mask_load_p: MaskLoadParams
    ):

        callback_test = TestCallbackPolarized(
            test_img_folder, test_mask_folder, test_img_folder_polarized, test_mask_folder_polarized, 
            lambda img: self.predict_image_polarized(img),
            test_output, codes_to_lbls, lbls_to_colors, self.offset, mask_load_p, self.n_pol
        )
        
        (test_output / 'models').mkdir()
        checkpoint_path = str(test_output / 'models' / 'best.hdf5')

        callback_checkpoint = ModelCheckpoint(
            monitor='val_loss', filepath=checkpoint_path, save_best_only=True, save_weights_only=True
        )

        _ = self.model.fit(
            train_gen,
            validation_data=val_gen, 
            steps_per_epoch=n_steps,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=[
                callback_test,
                callback_checkpoint,
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4),
            ],
        )

n_classes = len(config.class_names)

patch_s = 384
batch_s = 16
n_layers = 4
n_filters = 16
LR = 0.001
patch_overlay = 0.5

assert len(sys.argv) > 1 and sys.argv[1].isnumeric()
gpu_index = int(sys.argv[1])
assert gpu_index >= 0
set_gpu(gpu_index)


exp_path = prepare_experiment(Path('output_pol'))

data_path = '/home/d.sorokin/dev/geology/input/'

dataset_name = 'S1_v1_and_S3_v2'
dataset_name_base = dataset_name + '_base'
dataset_name_pol = dataset_name + '_polarized'

n_polazied = 6
add_dataset_pol_name = 'S3_v2_reg_results_' + str(n_polazied)


# missed_classes = (3, 5, 7, 9, 10, 12) # for S1_v1
missed_classes = (5, 7) # for S3_v1

n_classes = len(config.class_names)
present_classes = tuple(i for i in range(len(config.class_names)) if i not in missed_classes)
n_classes_sq = len(present_classes)

pg = AutoBalancedPatchGeneratorPolarized(
    Path(data_path + 'dataset/' + dataset_name + '/imgs/train/'),
    Path(data_path + 'dataset/' + dataset_name + '/masks/train/'),
    Path(data_path + 'cache/maps/'),
    Path(data_path + add_dataset_pol_name + '/imgs'),
    Path(data_path + add_dataset_pol_name + '/valid_zones'),
    patch_s, n_classes=n_classes_sq, distancing=0.5, mixin_random_every=5)

# missed_classes = pg.get_missed_classes()
print('==== missed classes:', missed_classes)

# pg.benchmark()

# exit(0)

squeeze_code_mappings = {code: i for i, code in enumerate(present_classes)}
codes_to_lbls = {i: config.class_names[code] for i, code in enumerate(present_classes)}

mask_load_p = MaskLoadParams(None, squeeze=True, squeeze_mappings=squeeze_code_mappings)

# loss_weights = recalc_loss_weights_2(pg.get_class_weights(remove_missed_classes=True))

bg = SimpleBatchGenerator(pg, batch_s, mask_load_p, augment=True)


model = GeoModel(n_polazied, patch_s, batch_s, offset=8, n_classes=n_classes_sq, LR=LR, patch_overlay=patch_overlay, class_weights=None)
model.initialize(n_filters, n_layers)
# # model.load(Path('./output/exp_89/models/best.hdf5'))

model.model.compile(
    optimizer=Adam(learning_rate=LR),
    # loss = weightedLoss(categorical_crossentropy, loss_weights),
    loss=categorical_crossentropy,
    metrics=[iou_tf]
)

model.train(
    # bg.g_balanced(), bg.g_random(), n_steps=400, epochs=50, val_steps=80,
    bg.g_balanced(), bg.g_random(), n_steps=800, epochs=50, val_steps=80,
    # bg.g_balanced(), bg.g_random(), n_steps=10, epochs=5, val_steps=5,
    test_img_folder=Path(data_path + '/dataset/' + dataset_name_base + '/imgs/test/'),
    test_mask_folder=Path(data_path + '/dataset/' + dataset_name_base + '/masks/test/'),
    test_img_folder_polarized=Path(data_path + '/dataset/' + dataset_name_pol + '/imgs/test/'),
    test_mask_folder_polarized=Path(data_path + '/dataset/' + dataset_name_pol + '/masks/test/'),
    test_output=exp_path, codes_to_lbls=codes_to_lbls, lbls_to_colors=config.lbls_to_colors,
    mask_load_p=mask_load_p
)
