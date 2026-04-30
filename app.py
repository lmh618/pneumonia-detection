import os
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm

from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "se_densenet121_pneumonia_balanced.pth")
EXAMPLE_DIR = os.path.join(BASE_DIR, "example")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def find_workflow_image():
    candidates = [
        "workflow.png",
        "workflow.jpg",
        "workflow.jpeg",
        "pipeline.png",
        "framework.png",
        "pneumonia_detection_workflow_overview.png",
        "pneumonia_detection_workflow_infographic.png",
        "wolkflow.png",
    ]
    for name in candidates:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    return None


WORKFLOW_IMAGE = find_workflow_image()

print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)
print("WORKFLOW_IMAGE =", WORKFLOW_IMAGE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = SimpleCNN(num_classes=2).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

feature_maps = None
gradients = None

def forward_hook(module, inp, out):
    global feature_maps
    feature_maps = out

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer = model.features
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

print("Grad-CAM target layer:", target_layer)

def generate_gradcam(pil_img):
    global feature_maps, gradients

    img_rgb = pil_img.convert("RGB")
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    model.zero_grad(set_to_none=True)
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()

    score = output[0, pred_class]
    score.backward()

    if feature_maps is None or gradients is None:
        raise RuntimeError("Grad-CAM hook failed: feature_maps or gradients is None.")

    fmap = feature_maps[0]
    grad = gradients[0]

    weights = torch.mean(grad, dim=(1, 2))

    cam = torch.zeros(fmap.shape[1:], dtype=torch.float32, device=device)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    return pred_class, confidence, cam

def overlay_heatmap_on_image(pil_img, cam):
    original = pil_img.convert("RGB").resize((224, 224))
    original_np = np.array(original).astype(np.float32) / 255.0

    heatmap = Image.fromarray(np.uint8(cam * 255)).resize((224, 224))
    heatmap_np = np.array(heatmap).astype(np.float32) / 255.0

    colored_heatmap = cm.jet(heatmap_np)[:, :, :3]
    overlay = 0.45 * colored_heatmap + 0.55 * original_np
    overlay = np.clip(overlay, 0, 1)

    return Image.fromarray((overlay * 255).astype(np.uint8))

def predict_image(image):
    if image is None:
        return (
            "Please upload a chest X-ray image.",
            None,
            "No interpretation available.",
        )

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    try:
        pred_class, confidence, cam = generate_gradcam(image)
        heatmap_img = overlay_heatmap_on_image(image, cam)
        pred_label = CLASS_NAMES[pred_class]

        if pred_label == "PNEUMONIA":
            interpretation = (
                "The model predicts this image as PNEUMONIA. "
                "The Grad-CAM heatmap highlights the regions contributing most strongly "
                "to the model's decision. Pneumonia cases often show patchy or dense opacities "
                "in the lung field, and the highlighted regions can help readers understand "
                "which abnormal areas influenced the prediction most."
            )
        else:
            interpretation = (
                "The model predicts this image as NORMAL. "
                "The Grad-CAM heatmap highlights the regions contributing most strongly "
                "to the model's decision. Compared with pneumonia cases, a normal chest X-ray "
                "usually has clearer lung fields and no obvious focal dense opacity."
            )

        result_text = f"Prediction: {pred_label}\nConfidence: {confidence:.4f}"
        return result_text, heatmap_img, interpretation

    except Exception as e:
        return (
            f"Prediction failed.\nError: {str(e)}",
            None,
            "Grad-CAM generation failed. Please check the model and image input.",
        )

def get_example_images():
    examples = []
    if os.path.exists(EXAMPLE_DIR):
        for fname in sorted(os.listdir(EXAMPLE_DIR)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                examples.append(os.path.join(EXAMPLE_DIR, fname))
    return examples

example_images = get_example_images()

css = '''
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
    background: #f7f5f1 !important;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin: 10px 0 14px 0;
}
.info-box {
    background: #eef4ff;
    border-left: 5px solid #4a74ff;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
}
.sub-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}
.note-box {
    background: #fff7ed;
    border-left: 5px solid #ea580c;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
}
'''

with gr.Blocks(title="Pneumonia Detection Platform", theme=gr.themes.Soft(), css=css) as demo:

    with gr.Tab("Algorithm & Background"):
        gr.HTML('<div class="section-title">Algorithm Principles & Our SE-DenseNet121 Framework</div>')

        gr.HTML("""
        <div class="info-box">
            <div class="sub-title">Overview of Our Work</div>
            <p>
            We developed an SE-DenseNet121-based pneumonia detection framework for chest X-ray classification.
            The pipeline includes dataset preparation, image preprocessing, class imbalance handling,
            transfer learning, model training, official test-set evaluation, and Grad-CAM-based visualization.
            </p>
            <ul>
                <li>Kaggle Chest X-ray Pneumonia Dataset</li>
                <li>SE-enhanced DenseNet121 backbone</li>
                <li>Weighted cross-entropy for class imbalance mitigation</li>
                <li>Grad-CAM visualization for interpretability support</li>
            </ul>
        </div>
        """)

        gr.Markdown("### Overall Pipeline of the Proposed Framework")
        if WORKFLOW_IMAGE is not None:
            gr.Image(value=WORKFLOW_IMAGE, label="Framework Overview", interactive=False, height=700)
        else:
            gr.Markdown("No workflow image found. Please place `workflow.png` in the same folder as app.py.")

        gr.HTML("""
        <div class="info-box" style="margin-top:16px;">
            <div class="sub-title">SE-DenseNet121 Architecture</div>
            <p>
            The proposed method uses DenseNet121 as the backbone, where dense connections encourage feature reuse
            and improve gradient propagation. An SE attention module recalibrates channel-wise feature responses
            to emphasize informative pneumonia-related patterns. Global Average Pooling and a fully connected layer
            then produce the final binary classification output.
            </p>
        </div>
        """)

    with gr.Tab("Single Image Annotation"):
        gr.HTML('<div class="section-title">Single Image Pneumonia Analysis</div>')
        gr.Markdown("Upload one chest X-ray image and run the model to obtain prediction results and visualization.")

        gr.HTML("""
        <div class="info-box">
            <div class="sub-title">What the reader should compare</div>
            <p>
            A healthy chest X-ray usually shows relatively clear lung fields and no obvious focal dense opacity,
            while a pneumonia chest X-ray often presents patchy or confluent opacities, local dense regions,
            or abnormal lung texture. The heatmap is used to show which image regions contributed most strongly
            to the model's decision.
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Chest X-ray")
                run_btn = gr.Button("Run Analysis", variant="primary")

            with gr.Column(scale=1):
                result_box = gr.Textbox(label="Prediction Result", lines=4)
                cam_box = gr.Image(type="pil", label="Heatmap / Visualization")
                interp_box = gr.Textbox(label="Interpretation Note", lines=6)

        run_btn.click(
            fn=predict_image,
            inputs=input_image,
            outputs=[result_box, cam_box, interp_box]
        )

        gr.Markdown("### Two Suggested Comparison Cases")
        gr.Markdown("Prepare one NORMAL image and one PNEUMONIA image in the `example/` folder so readers can directly compare the model outputs.")

        gr.HTML("""
        <div class="note-box">
            <div class="sub-title">Suggested comparison logic</div>
            <ul>
                <li><b>Normal case:</b> clear lung fields, relatively uniform texture, no obvious focal dense opacity.</li>
                <li><b>Pneumonia case:</b> patchy opacity, local high-density region, or abnormal lung texture.</li>
                <li><b>Heatmap meaning:</b> the highlighted area indicates which region the model relied on most.</li>
                <li><b>Reader takeaway:</b> the model is responding to visually abnormal regions rather than making a random decision.</li>
            </ul>
        </div>
        """)

        gr.Markdown("### Quick Test Images")
        gr.Markdown("Put a few sample X-ray images into the `example/` folder. Ideally include one healthy case and one pneumonia case. Click one image below to automatically load it.")

        if len(example_images) > 0:
            gr.Examples(
                examples=example_images,
                inputs=input_image,
                outputs=[],
                label="Click a test image to load it"
            )
        else:
            gr.Markdown("No example images found in the `example/` folder.")

        gr.HTML("""
        <div class="info-box">
            <div class="sub-title">Recommended explanation text for readers</div>
            <p>
            If the model predicts <b>PNEUMONIA</b>, the heatmap should mainly focus on abnormal dense or patchy lung regions.
            If the model predicts <b>NORMAL</b>, the highlighted regions are expected to be weaker and more evenly distributed,
            without concentrating on a suspicious focal opacity. This comparison helps readers understand how the model
            distinguishes healthy and pneumonia chest X-ray images from image-level visual evidence.
            </p>
        </div>
        """)

    with gr.Tab("Explore More"):
        gr.HTML('<div class="section-title">Explore More</div>')

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="info-box">
                    <div class="sub-title">Model Comparison</div>
                    <ul>
                        <li>Compare VGG16, ResNet50, MobileNetV2, and SE-DenseNet121</li>
                        <li>Review final Accuracy, Precision, Recall, F1-score, and AUC</li>
                        <li>Display published-method comparison from the paper</li>
                    </ul>
                </div>
                """)

            with gr.Column():
                gr.HTML("""
                <div class="info-box">
                    <div class="sub-title">Explainability</div>
                    <ul>
                        <li>Display Grad-CAM examples</li>
                        <li>Review radiographic plausibility</li>
                        <li>Support visual interpretation of model decisions</li>
                    </ul>
                </div>
                """)

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="info-box">
                    <div class="sub-title">Deployment</div>
                    <ul>
                        <li>Web demo screenshots</li>
                        <li>Workflow figure display</li>
                        <li>Paper figure and result export support</li>
                    </ul>
                </div>
                """)

demo.launch()
