import torch
import cv2

def preprocess_frame(frame, device=None):
    """
    Resize → Normalize → Convert to tensor
    """
    img = cv2.resize(frame, (416, 416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inp = torch.tensor(img / 255.0).permute(2, 0, 1).unsqueeze(0).float()

    if device:
        inp = inp.to(device)

    return inp
