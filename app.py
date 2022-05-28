#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import sys
import tarfile
from typing import Callable

import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from scipy.spatial.transform import Rotation

sys.path.insert(0, 'face_detection')
sys.path.insert(0, 'deep-head-pose/code')

from hopenet import Hopenet
from ibug.face_detection import RetinaFacePredictor

TITLE = 'natanielruiz/deep-head-pose'
DESCRIPTION = 'This is an unofficial demo for https://github.com/natanielruiz/deep-head-pose.'
ARTICLE = '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.hopenet" alt="visitor badge"/></center>'

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_sample_images() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        image_dir.mkdir()
        dataset_repo = 'hysts/input-images'
        filenames = ['001.tar']
        for name in filenames:
            path = huggingface_hub.hf_hub_download(dataset_repo,
                                                   name,
                                                   repo_type='dataset',
                                                   use_auth_token=TOKEN)
            with tarfile.open(path) as f:
                f.extractall(image_dir.as_posix())
    return sorted(image_dir.rglob('*.jpg'))


def load_model(model_name: str, device: torch.device) -> nn.Module:
    path = huggingface_hub.hf_hub_download('hysts/Hopenet',
                                           f'models/{model_name}.pkl',
                                           use_auth_token=TOKEN)
    state_dict = torch.load(path, map_location='cpu')
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def create_transform() -> Callable:
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def crop_face(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    x0 -= 2 * w // 4
    x1 += 2 * w // 4
    y0 -= 3 * h // 4
    y1 += h // 4
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image.shape[1])
    y1 = min(y1, image.shape[0])
    image = image[y0:y1, x0:x1]
    return image


@torch.inference_mode()
def predict(image: np.ndarray, transform: Callable, model: nn.Module,
            device: torch.device) -> np.ndarray:
    indices = torch.arange(66).float().to(device)

    image = PIL.Image.fromarray(image)
    data = transform(image)
    data = data.to(device)

    # the output of the model is a tuple of 3 tensors (yaw, pitch, roll)
    # the shape of each tensor is (1, 66)
    out = model(data[None, ...])
    out = torch.stack(out, dim=1)  # shape: (1, 3, 66)
    out = F.softmax(out, dim=2)
    out = (out * indices).sum(dim=2) * 3 - 99
    out = out.cpu().numpy()[0]
    return out


def draw_axis(image: np.ndarray, pose: np.ndarray, origin: np.ndarray,
              length: int) -> None:
    # (yaw, pitch, roll) -> (roll, yaw, pitch)
    pose = pose[[2, 0, 1]]
    pose *= np.array([1, -1, 1])
    rot = Rotation.from_euler('zyx', pose, degrees=True)

    vectors = rot.as_matrix().T[:, :2]  # shape: (3, 2)
    pts = np.round(vectors * length + origin).astype(int)

    cv2.line(image, tuple(origin), tuple(pts[0]), (0, 0, 255), 3)
    cv2.line(image, tuple(origin), tuple(pts[1]), (0, 255, 0), 3)
    cv2.line(image, tuple(origin), tuple(pts[2]), (255, 0, 0), 2)


def run(image: np.ndarray, model_name: str, face_detector: RetinaFacePredictor,
        models: dict[str, nn.Module], transform: Callable,
        device: torch.device) -> np.ndarray:
    model = models[model_name]

    # RGB -> BGR
    det_faces = face_detector(image[:, :, ::-1], rgb=False)

    res = image[:, :, ::-1].copy()
    for det_face in det_faces:
        box = np.round(det_face[:4]).astype(int)

        # RGB
        face_image = crop_face(image, box.tolist())

        # (yaw, pitch, roll)
        angles = predict(face_image, transform, model, device)

        center = (box[:2] + box[2:]) // 2
        length = (box[3] - box[1]) // 2
        draw_axis(res, angles, center, length)

    return res[:, :, ::-1]


def main():
    args = parse_args()
    device = torch.device(args.device)

    face_detector = RetinaFacePredictor(
        threshold=0.8,
        device=device,
        model=RetinaFacePredictor.get_model('mobilenet0.25'))

    model_names = [
        'hopenet_alpha1',
        'hopenet_alpha2',
        'hopenet_robust_alpha1',
    ]
    models = {name: load_model(name, device) for name in model_names}

    transform = create_transform()

    func = functools.partial(run,
                             face_detector=face_detector,
                             models=models,
                             transform=transform,
                             device=device)
    func = functools.update_wrapper(func, run)

    image_paths = load_sample_images()
    examples = [[path.as_posix(), model_names[0]] for path in image_paths]

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Radio(model_names,
                            type='value',
                            default=model_names[0],
                            label='Model'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
