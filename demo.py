import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
from models import build_model
import torch
from main import get_args_parser
import argparse
from pathlib import Path
# import util.misc as utils
# import numpy as np
# import random
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * (torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device))
    return b


def detect(im, model, transform):
    # print(f"device ========>>>>>>>> {device}")
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    img = img.to(device)
    # print(f"image input device ========>>>>>>>> {img.get_device()}")
    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[
        -1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    print(f"model device =========>>>>>>>> {next(model.parameters()).device}")
    outputs = model(img)
    # print(f"{'-'*10}outputs{'-'*10}\n{outputs}\n")
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    print(f"{'-' * 10}probas{'-' * 10}\n{probas}\n")
    keep = probas.max(-1).values >= 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def plot_results(pil_img, prob, boxes, output_dir, img_name):
    plt.figure(figsize=(40, 30))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(f"{output_dir}/{img_name}")


def demo(args):
    # utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    #
    # # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    model, _, _ = build_model(args)
    print(f"{'-' * 10}DONE BUILD MODEL{'-' * 10}\n")
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if os.path.isdir(args.image_input):
        for img_filename in os.listdir(args.image_input):
            im = Image.open(os.path.join(args.image_input, img_filename))
            scores, boxes = detect(im, model, transform)
            plot_results(im, scores, boxes, args.output_dir, f"{img_filename}")
    else:
        im = Image.open(args.image_input)
        scores, boxes = detect(im, model, transform)
        plot_results(im, scores, boxes, args.output_dir, "result.png")
    print(f"{'-' * 10}DONE{'-' * 10}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument("--image_input", type=str)
    args = parser.parse_args()
    # args.device = device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES = ['cattle']
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    demo(args)
