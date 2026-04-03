import torch
from ultralytics import YOLO
from PIL import Image

model = YOLO('yolov8m.pt')
backbone_layer = model.model.model[9]

feats = None
def hook(m, i, o):
    global feats
    feats = o

backbone_layer.register_forward_hook(hook)

dummy_img = Image.new('RGB', (640, 640))
model(dummy_img, verbose=False)
print("Features shape:", feats.shape)
