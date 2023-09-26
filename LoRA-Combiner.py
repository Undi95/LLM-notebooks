import os, torch
from tqdm import tqdm
import torch.nn.functional as F

def resize_2D_tensor(input_tensor, output_size):
    # Detect which dimension needs resizing
    resize_dim = 0 if input_tensor.shape[0] != output_size[0] else 1

    # Create a linear layer for the resizing
    in_size = input_tensor.shape[resize_dim]
    out_size = output_size[resize_dim]
    linear = torch.nn.Linear(in_size, out_size, bias=False)
    
    # Initialize the weights for linear interpolation
    with torch.no_grad():
        weight = torch.linspace(0, 1, steps=out_size).unsqueeze(1)
        weight = weight / (weight.sum(dim=0) + 1e-8)
        linear.weight.copy_(weight)
    
    # Resize the tensor along the detected dimension
    if resize_dim == 0:
        resized_tensor = linear(input_tensor.T).T
    else:
        resized_tensor = linear(input_tensor)

    return resized_tensor

def slerp(v0, v1, t):
    # Compute the cosine of the angle between the two vectors.
    dot = (v0 * v1).sum()
    
    # If the dot product is negative, the interpolation will take the long way around the sphere.
    # To prevent this, we can flip one of the input vectors.
    if dot < 0.0:
        v1 = -v1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the inputs are too close, linearly interpolate and normalize the result.
        result = v0 + t * (v1 - v0)
        return result / result.norm()

    # Compute the angle between the two vectors and use it to compute the interpolating weights.
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * v0 + s1 * v1

def BlendWithSLERP(aPath, bPath, outName, t=0.5):
    a = torch.load(aPath, map_location="cpu")
    b = torch.load(bPath, map_location="cpu")

    for key in tqdm(a.keys()):
        
        # Ensure tensors are the same size
        if a[key].shape != b[key].shape:
            if a[key].shape[0] != b[key].shape[0]:
                b[key] = resize_2D_tensor(b[key], (a[key].shape[0], b[key].shape[1]))
            else:
                b[key] = resize_2D_tensor(b[key], (b[key].shape[0], a[key].shape[1]))
        
        a[key] = slerp(a[key], b[key], t)

    torch.save(a, outName)
    
    
loraPath1 = ".lora1/adapter_model.bin"
loraPath2 = ".lora2/adapter_model.bin"
savePath = ".loraResult/adapter_model.bin"

BlendWithSLERP(loraPath1, loraPath2, savePath, 0.6)










