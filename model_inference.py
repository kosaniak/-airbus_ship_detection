# Building Unet model
import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Concatenate
from keras.layers import GlobalAveragePooling2D, UpSampling2D, Conv2D, MaxPooling2D
import keras.backend as K
from keras.losses import binary_crossentropy
from model_training import *

inp = Input(shape=(768, 768, 3))

# first block
conv_1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
conv_1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_1_1)
pool_1 = MaxPooling2D(2)(conv_1_2)


# second block
conv_2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_1)
conv_2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_2_1)
pool_2 = MaxPooling2D(2)(conv_2_2)


# third block
conv_3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_2)
conv_3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_3_1)
pool_3 = MaxPooling2D(2)(conv_3_2)


# fourth block
conv_4_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_3)
conv_4_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_4_1)
pool_4 = MaxPooling2D(2)(conv_4_2)


# fifth block
conv_5_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
conv_5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_5_1)
pool_5 = MaxPooling2D(2)(conv_5_2)


# first decoding block
up_1 = UpSampling2D(2, interpolation='bilinear')(pool_5)
conc_1 = Concatenate()([conv_5_2, up_1])
conv_up_1_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(conc_1)
conv_up_1_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_up_1_1)

# second decoding block
up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_2)
conc_2 = Concatenate()([conv_4_2, up_2])
conv_up_2_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc_2)
conv_up_2_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_up_2_1)


# third decodinc block
up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_2)
conc_3 = Concatenate()([conv_3_2, up_3])
conv_up_3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc_3)
conv_up_3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_up_3_1)

# fourth decoding block
up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_2)
conc_4 = Concatenate()([conv_2_2, up_4])
conv_up_4_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc_4)
conv_up_4_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_up_4_1)

# fifth decoding block
up_5 = UpSampling2D(2, interpolation='bilinear')(conv_up_4_2)
conc_5 = Concatenate()([conv_1_2, up_5])
conv_up_5_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc_5)
conv_up_5_2 = Conv2D(1, (3, 3), padding='same')(conv_up_5_1)
result = Activation('sigmoid')(conv_up_5_2)


unet_model = Model(inputs=inp, outputs=result)

unet_model.summary()


# callbacks

best_w = keras.callbacks.ModelCheckpoint('best_unet.w',
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        patience=1)

last_w = keras.callbacks.ModelCheckpoint('last_unet.w',
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=True,
                                        mode='auto',
                                        patience=20)

callbacks = [best_w, last_w]

## IoU of boats
def IoU(y_true, y_pred, eps=1e-6):
    y_true=K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)

## IoU of non-boats
def zero_IoU(y_true, y_pred):
    return IoU(1-y_true, 1-y_pred)

def agg_loss(in_gt, in_pred):
    return -1e-2 * zero_IoU(in_gt, in_pred) - IoU(in_gt, in_pred)
  

# optimizer
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
  
unet_model.compile(optimizer=adam, loss=agg_loss, metrics=[IoU, zero_IoU, 'binary_accuracy'])

loss_history = [unet_model.fit_generator(keras_generator(train_df),
                                        steps_per_epoch=10, 
                                        epochs=15, 
                                        validation_data=keras_generator(valid_df),
                                        validation_steps=50,
                                        callbacks=callbacks)]


def show_loss(loss_history):
    epochs = np.concatenate([mh.epoch for mh in loss_history])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')
    
    _ = ax2.plot(epochs, np.concatenate([mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_binary_accuracy'] for mh in loss_history]), 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('Binary Accuracy (%)')

show_loss(loss_history)