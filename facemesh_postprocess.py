import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
np.random.seed(0)

# Create a model
i = tf.keras.layers.Input(
    shape=[
        1,
        1,
        1404
    ],
    batch_size=1,
    dtype=tf.float32,
)

crop_x1 = tf.keras.layers.Input(
    shape=[
        1,
    ],
    batch_size=1,
    dtype=tf.int32,
)

crop_y1 = tf.keras.layers.Input(
    shape=[
        1,
    ],
    batch_size=1,
    dtype=tf.int32,
)

crop_width = tf.keras.layers.Input(
    shape=[
        1,
    ],
    batch_size=1,
    dtype=tf.int32,
)

crop_height = tf.keras.layers.Input(
    shape=[
        1,
    ],
    batch_size=1,
    dtype=tf.int32,
)

scale_w = tf.divide(tf.cast(crop_width, dtype=tf.float32), 192.0)
scale_h = tf.divide(tf.cast(crop_height, dtype=tf.float32), 192.0)

landmarks_result = tf.reshape(i, (1, 468, 3))
x = landmarks_result[:, :, 0] * scale_w + 0.5 + tf.cast(crop_x1, dtype=tf.float32)
y = landmarks_result[:, :, 1] * scale_h + 0.5 + tf.cast(crop_y1, dtype=tf.float32)
z = landmarks_result[:, :, 2]

x = tf.reshape(x, [1,468,1])
y = tf.reshape(y, [1,468,1])
z = tf.reshape(z, [1,468,1])

con = tf.concat([x,y,z], axis=2)
o = tf.cast(con, dtype=tf.int32)

model = tf.keras.models.Model(inputs=[i,crop_x1,crop_y1,crop_width,crop_height], outputs=[o])
model.summary()
output_path = 'saved_model_postprocess'
tf.saved_model.save(model, output_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
open(f"{output_path}/test.tflite", "wb").write(tflite_model)
