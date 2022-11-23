from jittor.utils.pytorch_converter import convert
pytorch_code="""
import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from utils import rend_util
"""

jittor_code = convert(pytorch_code)
print(jittor_code)