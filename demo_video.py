import numpy as np
import cv2
import argparse
import onnxruntime


def resize_and_pad(src, size, pad_color=0):
    img = src.copy()
    h, w = img.shape[:2]
    sh, sw = size
    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    aspect = w/h
    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = \
            np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = \
            np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color]*3
    scaled_img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interp
    )
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return scaled_img


def main(args):
    # Load Face Detection Model
    face_detection_model = 'yolov4_headdetection_480x640_post.onnx'
    session_option_detection = onnxruntime.SessionOptions()
    session_option_detection.log_severity_level = 3
    face_detection_sess = onnxruntime.InferenceSession(
        face_detection_model,
        sess_options=session_option_detection,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    face_detection_input_name = face_detection_sess.get_inputs()[0].name
    face_detection_input_shapes = face_detection_sess.get_inputs()[0].shape
    face_detection_output_names = [output.name for output in face_detection_sess.get_outputs()]

    # Load FaceMesh Model
    face_mesh_model = 'face_mesh_Nx3x192x192_post.onnx'
    session_option_facemesh = onnxruntime.SessionOptions()
    session_option_facemesh.log_severity_level = 3
    face_mesh_sess = onnxruntime.InferenceSession(
        face_mesh_model,
        sess_options=session_option_facemesh,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    face_mesh_input_name = [
        input.name for input in face_mesh_sess.get_inputs()
    ]
    face_mesh_output_names = [
        output.name for output in face_mesh_sess.get_outputs()
    ]

    cap_width = int(args.height_width.split('x')[1])
    cap_height = int(args.height_width.split('x')[0])
    if args.device.isdecimal():
        cap = cv2.VideoCapture(int(args.device))
    else:
        cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    WINDOWS_NAME = 'Demo'
    cv2.namedWindow(WINDOWS_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWS_NAME, cap_width, cap_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ============================================================= Face Detection
        # Resize
        resized_frame = resize_and_pad(
            frame,
            (
                face_detection_input_shapes[2],
                face_detection_input_shapes[3],
            )
        )
        width = resized_frame.shape[1]
        height = resized_frame.shape[0]
        # BGR to RGB
        rgb = resized_frame[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # normalize to [0, 1] interval
        chw = np.asarray(chw / 255., dtype=np.float32)
        # hwc --> nhwc
        nchw = chw[np.newaxis, ...]

        outputs_x1y1x2y2score = face_detection_sess.run(
            output_names = face_detection_output_names,
            input_feed = {face_detection_input_name: nchw}
        )
        heads = []
        for x1y1x2y2socre in outputs_x1y1x2y2score[0]:
            score = x1y1x2y2socre[4]
            if score > 0.60:
                heads.append(
                    [
                        int(max(x1y1x2y2socre[0] * width - 20, 0)),
                        int(max(x1y1x2y2socre[1] * height - 20, 0)),
                        int(min(x1y1x2y2socre[2] * width + 20, width)),
                        int(min(x1y1x2y2socre[3] * height + 20, height)),
                        x1y1x2y2socre[4],
                    ]
                )
        canvas = resized_frame.copy()

        # ============================================================= FaceMesh
        if len(heads) > 0:
            facemesh_input_images = []
            crop_x1 = []
            crop_y1 = []
            crop_width = []
            crop_height = []

            for head in heads:
                x_min = head[0]
                y_min = head[1]
                x_max = head[2]
                y_max = head[3]
                facemesh_input_images.append(
                    cv2.resize(resized_frame[y_min:y_max,x_min:x_max,:], (192,192))[:,:,::-1] / 255
                )
                crop_x1.append(int(x_min))
                crop_y1.append(int(y_min))
                crop_width.append(int(x_max-x_min))
                crop_height.append(int(y_max-y_min))

                # Face Bounding Box drawing
                cv2.rectangle(
                    canvas,
                    (x_min, y_min),
                    (x_max, y_max),
                    color=(255, 0, 0),
                    thickness=2
                )

            np_facemesh_input_images = np.asarray(
                facemesh_input_images, dtype=np.float32
            ).transpose(0,3,1,2)
            np_crop_x1 = np.asarray(crop_x1, dtype=np.int32).reshape(-1,1)
            np_crop_y1 = np.asarray(crop_y1, dtype=np.int32).reshape(-1,1)
            np_crop_width = np.asarray(crop_width, dtype=np.int32).reshape(-1,1)
            np_crop_height = np.asarray(crop_height, dtype=np.int32).reshape(-1,1)

            # FaceMesh inference
            score, final_landmarks = face_mesh_sess.run(
                output_names = face_mesh_output_names,
                input_feed = {
                    face_mesh_input_name[0]: np_facemesh_input_images,
                    face_mesh_input_name[1]: np_crop_x1,
                    face_mesh_input_name[2]: np_crop_y1,
                    face_mesh_input_name[3]: np_crop_width,
                    face_mesh_input_name[4]: np_crop_height,
                }
            )

            # Face Landmark drawing
            for face in final_landmarks:
                for keypoint in face:
                    x = keypoint[0]
                    y = keypoint[1]
                    z = keypoint[2]
                    cv2.circle(
                        img=canvas,
                        center=(x, y),
                        radius=2,
                        color=(255, 209, 0),
                        thickness=1,
                    )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        cv2.imshow(WINDOWS_NAME, canvas)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default='0',
        help='Path of the mp4 file or device number of the USB camera. Default: 0',
    )
    parser.add_argument(
        "--height_width",
        type=str,
        default='480x640',
        help='{H}x{W}. Default: 480x640',
    )
    args = parser.parse_args()
    main(args)