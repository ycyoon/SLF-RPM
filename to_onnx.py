from models import classifier
import torch.onnx

model = classifier.LinearClsResNet3D(model_depth=18, n_class=1)
dummy_data = torch.empty(1, 75, 3, 64, 64, dtype = torch.float32)
torch.onnx.export(model, dummy_data, "output.onnx")

