import cv2
import numpy as np
import imageio
import skimage

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def load_rgb(path):
    # print("path:          ", path)
    img = imageio.imread(path)
    # print("099999999999999999")
    
    img = skimage.img_as_float32(img)
    # print(img.shape)
    # pixel values between [-1,1]
    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    # print(img.shape)
    
    return img

def load_mask(path):
    # /print("path:          ", path)
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    object_mask = alpha > 127.5

    return object_mask