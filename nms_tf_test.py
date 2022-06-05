import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
from pprint import pprint
np.random.seed(0)

# dummy_input = np.ones([1,10], dtype=np.float32)

BOXES=18900

# Create a model
boxes = tf.keras.layers.Input(
    shape=[
        BOXES,
        1,
        4,
    ],
    batch_size=1,
    dtype=tf.float32,
)

scores = tf.keras.layers.Input(
    shape=[
        BOXES,
        1,
    ],
    batch_size=1,
    dtype=tf.float32,
)

boxes_non_batch = tf.squeeze(boxes)
x1 = boxes_non_batch[:,0][:,np.newaxis]
y1 = boxes_non_batch[:,1][:,np.newaxis]
x2 = boxes_non_batch[:,2][:,np.newaxis]
y2 = boxes_non_batch[:,3][:,np.newaxis]
boxes_y1x1y2x2 = tf.concat([y1,x1,y2,x2], axis=1)

scores_non_batch = tf.squeeze(scores)

selected_indices = tf.image.non_max_suppression(
    boxes=boxes_y1x1y2x2,
    scores=scores_non_batch,
    max_output_size=100,
    iou_threshold=0.5,
    score_threshold=float('-inf'),
)

# selected_indices = tf.raw_ops.NonMaxSuppressionV3(
#     boxes=boxes[0],
#     scores=scores[0],
#     max_output_size=100,
#     iou_threshold=0.5,
#     score_threshold=float('-inf'),
# )

# def NonMaxSuppressionV3_(boxes, scores, max_output_size: int, iou_threshold, score_threshold):
#     selected_indices = \
#         tf.raw_ops.NonMaxSuppressionV3(
#             boxes=boxes,
#             scores=scores,
#             max_output_size=max_output_size,
#             iou_threshold=iou_threshold,
#             score_threshold=score_threshold
#         )
#     return selected_indices

# selected_indices = \
#     tf.keras.layers.Lambda(
#         NonMaxSuppressionV3_,
#         arguments={
#             'scores': scores[0],
#             'max_output_size': 100,
#             'iou_threshold': 0.5,
#             'score_threshold': float('-inf'),
#         }
#     )(boxes[0])

selected_boxes =  tf.gather(
    boxes_non_batch,
    selected_indices
)

selected_scores = tf.gather(
    scores_non_batch,
    selected_indices
)
selected_scores = tf.expand_dims(selected_scores, axis=1)

outputs = tf.concat([selected_boxes, selected_scores], axis=1)

model = tf.keras.models.Model(inputs=[boxes,scores], outputs=[outputs])
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
