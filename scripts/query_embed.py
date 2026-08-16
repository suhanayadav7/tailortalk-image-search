import sys
import warnings

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

MODEL_NAME = "facebook/dinov2-small"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to("cpu")
model.eval()

image = Image.open(sys.argv[1]).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :]
    vec = torch.nn.functional.normalize(vec, dim=-1)

sys.stdout.write(",".join(map(str, vec[0].cpu().numpy().astype("float32"))))
