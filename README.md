# [WIP] facemesh_onnx_tensorrt
Verify that the post-processing merged into `FaceMesh` works correctly. The object detection model can be anything other than BlazeFace.

# 1. Pre-trained model
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/032_FaceMesh

https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark/face_landmark.tflite

# 2. ModelCard

https://drive.google.com/file/d/1QvwWNfFoweGVjsXF3DXzcrCnz-mx-Lha/preview

https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection

- Standard facial landmark 68 points

  ![image](https://user-images.githubusercontent.com/33194443/172013276-3b640648-8bfd-4d2a-b435-4dc610ebc0bb.png)

- Full facemesh 468 points

  ![image](https://user-images.githubusercontent.com/33194443/172013054-4a826611-cb5b-4dfb-ab14-addf0acaa06e.png)

- INPUT:
  - input: float32 `[N, 3, 192, 192]`
  - crop_x1: int32 `[N, 1]`, Coordinates reflecting about 25% margin on the `X1` coordinates of the object detection result.
  - crop_y1: int32 `[N, 1]`, Coordinates reflecting about 25% margin on the `Y1` coordinates of the object detection result.
  - crop_width: int32 `[N, 1]`, Width of face image reflecting about 25% margin to the `left` and `right` of object detection results.
  - crop_height: int32 `[N, 1]`, Height of face image reflecting about 25% margin to the `top` and `bottom` object detection results.
    ![icon_design drawio (1)](https://user-images.githubusercontent.com/33194443/172016038-8c0928a4-d8d2-4966-b131-b1ca778097d4.png)

- OUTPUT:
  - final_landmarks: int32 `[N, 468, 3]`, `X, Y, Z`
  - score: float32 `[N, 1]`

# 3. Benchmark
## 3-1. 1 batch + ONNX + TensorRT, 10 times loop
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
## 3-2. 100 batch + ONNX + TensorRT, 10 times loop
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
