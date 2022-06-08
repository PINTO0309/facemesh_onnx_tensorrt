# facemesh_onnx_tensorrt
Verify that the post-processing merged into `FaceMesh` works correctly. The object detection model can be anything other than BlazeFace. YOLOv4 and FaceMesh committed to this repository have modified post-processing.

- YOLOv4 + Modified FaceMesh

  https://user-images.githubusercontent.com/33194443/172059573-4ceafd5c-5881-4133-8367-746adb9464c5.mp4

  https://user-images.githubusercontent.com/33194443/172077025-cfc86269-cd1f-4762-82fd-34c912365cc7.mp4

  https://user-images.githubusercontent.com/33194443/172284562-4af14c8f-ef95-4abc-b6bc-0f89a57ecfde.mp4

  The **`C`** and **`B`** keys on the keyboard can be used to switch between display modes.
  ```bash
  python demo_video.py
  ```

# 1. Pre-trained model
1. https://github.com/PINTO0309/facemesh_onnx_tensorrt/releases
2. https://github.com/PINTO0309/PINTO_model_zoo/tree/main/032_FaceMesh
3. https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark/face_landmark.tflite

# 2. ONNX Modification Tools
1. https://github.com/PINTO0309/simple-onnx-processing-tools
2. https://github.com/PINTO0309/tflite2tensorflow
3. https://github.com/PINTO0309/openvino2tensorflow

# 3. ModelCard

1. https://drive.google.com/file/d/1QvwWNfFoweGVjsXF3DXzcrCnz-mx-Lha/preview
2. https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection

- Standard facial landmark 68 points

  ![image](https://user-images.githubusercontent.com/33194443/172013276-3b640648-8bfd-4d2a-b435-4dc610ebc0bb.png)

- Full facemesh 468 points

  ![image](https://user-images.githubusercontent.com/33194443/172013054-4a826611-cb5b-4dfb-ab14-addf0acaa06e.png)

- INPUT: (`N` is the number of bounding boxes for object detection. Also called batch size.)
  - `input`: `float32 [N, 3, 192, 192]`

    Object detection models for face detection have very narrow or wide detection areas, depending on the model type. Therefore, in order for FaceMesh to accurately detect feature points, the area to be cropped must not be too narrow, but must be expanded to fit the entire face as much as possible before cropping. In this case, it is necessary to crop a region that extends the actual x1, y1, x2, y2 area detected by object detection to some extent in the vertical and horizontal directions. This is the green area in the figure below. The cropped face area with a margin is resized to 192x192 without considering the aspect ratio, and then input into the FaceMesh model. If the area to be cropped is a large enough area, it may not be necessary to provide a margin. The 25% margin listed in the official model card is based on `BlazeFace`. `BlazeFace` has a very small face detection area.
  - `crop_x1`: `int32 [N, 1]`

    Coordinates reflecting about 25% margin on the `X1` coordinates of the object detection result.
  - `crop_y1`: `int32 [N, 1]`

    Coordinates reflecting about 25% margin on the `Y1` coordinates of the object detection result.
  - `crop_width`: `int32 [N, 1]`

    Width of face image reflecting about 25% margin to the `left` and `right` of object detection results.
  - `crop_height`: `int32 [N, 1]`

    Height of face image reflecting about 25% margin to the `top` and `bottom` object detection results.
  
  ![icon_design drawio (2)](https://user-images.githubusercontent.com/33194443/172016342-f67b3e28-db0e-4d2d-af12-2ef38b08395b.png)

- OUTPUT: (`N` is the number of bounding boxes for object detection. Also called batch size.)
  - `final_landmarks`: `int32 [N, 468, 3]`

    468 key points. X, Y, and Z coordinates.
  - `score`: `float32 [N, 1]`

    Probability value indicating whether a facial feature point has been successfully detected.

# 4. Benchmark
## 4-1. 1 batch + ONNX + TensorRT, 10 times loop
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
## 4-2. 100 batch + ONNX + TensorRT, 10 times loop
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
## 5. Structure of PINTO Special FaceMesh model
![face_mesh_Nx3x192x192_post onnx (1)](https://user-images.githubusercontent.com/33194443/172060695-fce7db47-f103-4993-bc65-a7594c023424.png)
