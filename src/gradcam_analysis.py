from collections import defaultdict
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def apply_gradcam(model, img_tensor, target_layer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = {}
    grads = {}

    def forward_hook(module, input, output):
        features["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        grads["value"] = grad_output[0].detach()

    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_backward_hook(backward_hook)

    model.eval()
    output = model(img_tensor.unsqueeze(0).to(device))
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][pred_class] = 1
    output.backward(gradient=one_hot)

    hook_forward.remove()
    hook_backward.remove()

    act = features["value"]
    grad = grads["value"]
    pooled_grad = torch.mean(grad, dim=[0, 2, 3])

    for i in range(act.shape[1]):
        act[:, i, :, :] *= pooled_grad[i]

    heatmap = torch.mean(act, dim=1).squeeze().cpu()
    heatmap = torch.clamp(heatmap, min=0)
    heatmap /= torch.max(heatmap)
    return heatmap.numpy(), pred_class


def audit_subgroup_predictions(model, extract_path, preprocess, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = defaultdict(lambda: {"true": [], "pred": [], "paths": []})
    categories = sorted(os.listdir(extract_path))

    for cat in categories:
        for subgroup in ["white", "non-white"]:
            folder = os.path.join(extract_path, cat, subgroup)
            image_paths = glob.glob(os.path.join(folder, "*.*"))

            for img_path in tqdm(image_paths, desc=f"{cat}/{subgroup}"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img).to(device)
                    with torch.no_grad():
                        output = model(img_tensor.unsqueeze(0))
                        pred = output.argmax(dim=1).item()

                    true = int(cat[1])
                    results[f"{cat}_{subgroup}"]["true"].append(true)
                    results[f"{cat}_{subgroup}"]["pred"].append(pred)
                    results[f"{cat}_{subgroup}"]["paths"].append(img_path)
                except Exception as exc:
                    print(f"Skipping {img_path}: {exc}")

    return results


def print_subgroup_metrics(results):
    for key in results:
        y_true = results[key]["true"]
        y_pred = results[key]["pred"]
        print(f"==== Results for {key} ====")
        print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred, average='macro'):.4f}")
        print(f"F1-score:  {f1_score(y_true, y_pred, average='macro'):.4f}")
        print()


def visualize_gradcam_examples(model, image_paths, preprocess, target_layer, title_prefix="Grad-CAM", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).to(device)
        heatmap, pred_class = apply_gradcam(model, img_tensor, target_layer, device=device)

        heatmap = cv2.resize(heatmap, img.size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        plt.title(f"{title_prefix} | Predicted class: {pred_class}")
        plt.axis("off")
        plt.show()
