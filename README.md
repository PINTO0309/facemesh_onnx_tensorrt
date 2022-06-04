# [WIP] facemesh_onnx_tensorrt
Verify that the post-processing merged into FaceMesh works correctly. The object detection model can be anything other than BlazeFace.

# 1. Pre-trained model
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/032_FaceMesh
https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark/face_landmark.tflite

# 2. Benchmark
## 2-1. 1 batch + ONNX + TensorRT
```bash
$ sit4onnx \
--input_onnx_file_path face_mesh_192x192_post.onnx

INFO: file: face_mesh_192x192_post.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input shape: [1, 3, 192, 192] dtype: float32
INFO: input_name.2: crop_x1 shape: [1, 1] dtype: int32
INFO: input_name.3: crop_y1 shape: [1, 1] dtype: int32
INFO: input_name.4: crop_width shape: [1, 1] dtype: int32
INFO: input_name.5: crop_height shape: [1, 1] dtype: int32
INFO: test_loop_count: 10
INFO: total elapsed time:  5.5561065673828125 ms
INFO: avg elapsed time per pred:  0.5556106567382812 ms
INFO: output_name.1: score shape: [1, 1] dtype: float32
INFO: output_name.2: final_landmarks shape: [1, 468, 3] dtype: int32
```
## 2-2. 100 batch + ONNX + TensorRT
```bash
$ sit4onnx \
--input_onnx_file_path face_mesh_Nx3x192x192_post.onnx \
--batch_size 100

INFO: file: face_mesh_Nx3x192x192_post.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input shape: [100, 3, 192, 192] dtype: float32
INFO: input_name.2: crop_x1 shape: [100, 1] dtype: int32
INFO: input_name.3: crop_y1 shape: [100, 1] dtype: int32
INFO: input_name.4: crop_width shape: [100, 1] dtype: int32
INFO: input_name.5: crop_height shape: [100, 1] dtype: int32
INFO: test_loop_count: 10
INFO: total elapsed time:  103.13057899475098 ms
INFO: avg elapsed time per pred:  10.313057899475098 ms
INFO: output_name.1: score shape: [100, 1] dtype: float32
INFO: output_name.2: final_landmarks shape: [100, 468, 3] dtype: int32
```
