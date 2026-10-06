import os
import cv2
import glob
import yaml
import numpy as np
import torch

from pathlib import Path
from models.networks import get_generator


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = r"C:\restoration_app\test"

OUTPUT_ROOT = r"C:\restoration_app\test_out"

# Change these to your actual weight paths
INCEPTION_WEIGHTS = "weights/fpn_inception.h5"
MOBILENET_WEIGHTS = "weights/fpn_mobilenet.h5"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# LOAD CONFIG
# ============================================================

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(weights_path, backbone):

    print("\n" + "=" * 70)
    print("Loading model")
    print("Backbone :", backbone)
    print("Weights  :", weights_path)
    print("=" * 70)

    # --------------------------------------------------------
    # Set generator backbone
    # --------------------------------------------------------

    model_config = config["model"].copy()

    model_config["g_name"] = backbone

    # --------------------------------------------------------
    # Build generator
    # --------------------------------------------------------

    model = get_generator(
        model_config
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    checkpoint = torch.load(
        weights_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:

        model.load_state_dict(
            checkpoint["model"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(image):

    # BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # 0-255 → 0-1
    image = image.astype(
        np.float32
    ) / 255.0

    # HWC → CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # [0,1] → [-1,1]
    image = (
        image * 2.0
    ) - 1.0

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# POSTPROCESS
# ============================================================

def postprocess(output):

    output = output[0]

    output = (
        output
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    # CHW → HWC
    output = np.transpose(
        output,
        (1, 2, 0)
    )

    # [-1,1] → [0,255]
    output = (
        (output + 1.0)
        / 2.0
        * 255.0
    )

    output = np.clip(
        output,
        0,
        255
    )

    output = output.astype(
        np.uint8
    )

    # RGB → BGR
    output = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )

    return output


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    model,
    image_path,
    output_dir
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "Could not read:",
            image_path
        )

        return

    original_height, original_width = (
        image.shape[:2]
    )

    tensor = preprocess(
        image
    )

    tensor = tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Pad to a multiple of 32 (the FPN needs it, as in DeblurGANv2's predict.py)
    # --------------------------------------------------------

    pad_h = (32 - original_height % 32) % 32
    pad_w = (32 - original_width % 32) % 32

    tensor = torch.nn.functional.pad(
        tensor,
        (0, pad_w, 0, pad_h),
        mode="reflect"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor
        )

    # Crop the padding off again
    output = output[:, :, :original_height, :original_width]

    # --------------------------------------------------------
    # Convert output
    # --------------------------------------------------------

    restored = postprocess(
        output
    )

    # --------------------------------------------------------
    # Resize if model changed dimensions
    # --------------------------------------------------------

    if (
        restored.shape[0] != original_height
        or
        restored.shape[1] != original_width
    ):

        restored = cv2.resize(
            restored,
            (
                original_width,
                original_height
            ),
            interpolation=cv2.INTER_CUBIC
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = Path(
        image_path
    ).stem

    output_path = os.path.join(
        output_dir,
        filename + "_deblurred.png"
    )

    cv2.imwrite(
        output_path,
        restored
    )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# PROCESS FOLDER
# ============================================================

def run_model(
    model,
    model_name
):

    output_dir = os.path.join(
        OUTPUT_ROOT,
        model_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff"
    ]

    image_files = []

    for extension in extensions:

        image_files.extend(
            glob.glob(
                os.path.join(
                    INPUT_DIR,
                    extension
                )
            )
        )

    image_files = sorted(
        image_files
    )

    print("\n")
    print("=" * 70)
    print(
        f"Testing {model_name}"
    )
    print(
        "Images:",
        len(image_files)
    )
    print("=" * 70)

    for image_path in image_files:

        print(
            "\nProcessing:",
            os.path.basename(
                image_path
            )
        )

        try:

            process_image(
                model,
                image_path,
                output_dir
            )

        except Exception as e:

            print(
                "ERROR:",
                e
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # MODEL 1
    # --------------------------------------------------------

    inception_model = load_model(
        INCEPTION_WEIGHTS,
        "fpn_inception"
    )

    run_model(
        inception_model,
        "inception"
    )

    # Free GPU memory
    del inception_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # MODEL 2
    # --------------------------------------------------------

    mobilenet_model = load_model(
        MOBILENET_WEIGHTS,
        "fpn_mobilenet"
    )

    run_model(
        mobilenet_model,
        "mobilenet"
    )

    print("\n")
    print("=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)

    print(
        "Inception output:"
    )

    print(
        "outputs/inception/"
    )

    print(
        "\nMobileNet output:"
    )

    print(
        "outputs/mobilenet/"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()