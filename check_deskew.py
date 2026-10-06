import cv2
import numpy as np
import os
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = r"C:\restoration_app\TEST1"
OUTPUT_DIR = r"C:\restoration_app\TEST1_OUTPUT"

# Angles to generate
SKEW_ANGLES = [
    -10,
    -7,
    -5,
    -3,
    -2,
    -1,
    1,
    2,
    3,
    5,
    7,
    10
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# ROTATE IMAGE WITHOUT CROPPING
# ============================================================

def rotate_without_cropping(image, angle):

    h, w = image.shape[:2]

    center = (
        w / 2,
        h / 2
    )

    # Rotation matrix
    M = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    # Calculate new dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_width = int(
        (h * sin) +
        (w * cos)
    )

    new_height = int(
        (h * cos) +
        (w * sin)
    )

    # Move image to center of new canvas
    M[0, 2] += (
        new_width / 2
        - center[0]
    )

    M[1, 2] += (
        new_height / 2
        - center[1]
    )

    rotated = cv2.warpAffine(
        image,
        M,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return rotated


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def create_synthetic_images(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"Could not read: {image_path}"
        )

        return

    print(
        f"\nProcessing: {image_path.name}"
    )

    for angle in SKEW_ANGLES:

        # ----------------------------------------------------
        # Create skewed image
        # ----------------------------------------------------

        skewed = rotate_without_cropping(
            image,
            angle
        )

        # ----------------------------------------------------
        # File name
        # ----------------------------------------------------

        # Convert -5 -> minus5
        if angle < 0:
            angle_string = f"minus{abs(angle)}"
        else:
            angle_string = f"plus{angle}"

        output_name = (
            f"{image_path.stem}"
            f"_skew_{angle_string}.png"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        cv2.imwrite(
            output_path,
            skewed
        )

        print(
            f"  Created: "
            f"{output_name}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    input_path = Path(
        INPUT_DIR
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
            input_path.glob(extension)
        )

    image_files = sorted(
        image_files
    )

    if len(image_files) == 0:

        print(
            f"No images found in "
            f"'{INPUT_DIR}'"
        )

        return

    print("=" * 70)
    print("SYNTHETIC SKEW DATASET GENERATOR")
    print("=" * 70)

    print(
        "Input images:",
        len(image_files)
    )

    print(
        "Angles:",
        SKEW_ANGLES
    )

    total_generated = 0

    for image_path in image_files:

        create_synthetic_images(
            image_path
        )

        total_generated += len(
            SKEW_ANGLES
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "Original images:",
        len(image_files)
    )

    print(
        "Images generated:",
        total_generated
    )

    print(
        "Output folder:",
        OUTPUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()