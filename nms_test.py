import torch
import torch.nn as nn
import torchvision as tv

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, boxes, scores, iou_threshold=0.5):
        selected_indices = tv.ops.nms(
            boxes=boxes,
            scores=scores,
            iou_threshold=iou_threshold,
        )
        return selected_indices

if __name__ == "__main__":
    model = Model()

    import onnx
    from onnxsim import simplify
    MODEL = f'nms'
    BOXES = 18900

    onnx_file = f"{MODEL}_{BOXES}.onnx"
    boxes = torch.randn(1,18900,4)
    scores = torch.randn(1,18900,1)

    torch.onnx.export(
        model,
        args=(boxes, scores),
        f=onnx_file,
        opset_version=11,
        input_names = ['boxes','scores'],
        output_names=['selected_indices'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    import sys
    sys.exit(0)