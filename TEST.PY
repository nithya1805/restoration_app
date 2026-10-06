import os
import cv2
import numpy as np
import csv
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FOLDER = r"C:\restoration_app\TEST1"
OUTPUT_FOLDER = r"C:\restoration_app\TEST1_OUTPUT"

# Images with detected skew below this value won't be rotated
SKEW_THRESHOLD = 1.0

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# SKEW DETECTION
# ============================================================

def determine_skew(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Connect text into larger horizontal regions
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (30, 5)
    )

    dilated = cv2.dilate(
        thresh,
        kernel,
        iterations=1
    )

    # Detect edges
    edges = cv2.Canny(
        dilated,
        50,
        150
    )

    # Detect straight lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return 0.0

    angles = []

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        angle = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1
            )
        )

        # Only consider mostly-horizontal lines
        if -45 < angle < 45:
            angles.append(angle)

    if len(angles) == 0:
        return 0.0

    # Median is more robust against outlier lines
    return float(np.median(angles))


# ============================================================
# ROTATE IMAGE
# ============================================================

def rotate_image(image, angle):

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    # Calculate new image dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(
        (h * sin) +
        (w * cos)
    )

    new_h = int(
        (h * cos) +
        (w * sin)
    )

    # Adjust rotation matrix
    M[0, 2] += (
        new_w / 2
    ) - center[0]

    M[1, 2] += (
        new_h / 2
    ) - center[1]

    # Rotate
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return rotated


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(input_path, output_path):

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(
            f"Could not read: {input_path}"
        )

    angle = determine_skew(input_path)

    print(
        f"{os.path.basename(input_path)} "
        f"-> detected skew: {angle:.2f} deg"
    )

    # --------------------------------------------------------
    # No correction required
    # --------------------------------------------------------

    if abs(angle) < SKEW_THRESHOLD:

        corrected = image.copy()

        print(
            f"   Skipped "
            f"(below {SKEW_THRESHOLD} deg)"
        )

    # --------------------------------------------------------
    # Apply correction
    # --------------------------------------------------------

    else:

        corrected = rotate_image(
            image,
            angle
        )

        print(
            f"   Corrected by {-angle:.2f} deg"
        )

    cv2.imwrite(
        output_path,
        corrected
    )

    return angle


# ============================================================
# FIND IMAGES
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

image_files = [
    p
    for p in Path(INPUT_FOLDER).rglob("*")
    if p.suffix.lower() in extensions
]

print("=" * 60)
print("SKEW CORRECTION TEST")
print("=" * 60)

print(
    f"Input folder : {INPUT_FOLDER}"
)

print(
    f"Images found : {len(image_files)}"
)

print(
    f"Threshold    : {SKEW_THRESHOLD}°"
)

print("=" * 60)


# ============================================================
# CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "skew_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "image",
        "detected_skew_angle",
        "correction_applied",
        "correction_angle",
        "output"
    ])

    # ========================================================
    # PROCESS FOLDER
    # ========================================================

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        try:

            output_name = (
                image_path.stem +
                "_deskewed" +
                image_path.suffix
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_name
            )

            angle = process_image(
                str(image_path),
                output_path
            )

            if abs(angle) < SKEW_THRESHOLD:
                applied = "NO"
                correction_angle = 0
            else:
                applied = "YES"
                correction_angle = -angle

            writer.writerow([
                image_path.name,
                f"{angle:.2f}",
                applied,
                f"{correction_angle:.2f}",
                output_name
            ])

            print(
                f"[{index}/{len(image_files)}] "
                "DONE\n"
            )

        except Exception as e:

            print(
                f"[{index}/{len(image_files)}] "
                f"ERROR: {e}\n"
            )

            writer.writerow([
                image_path.name,
                "",
                "ERROR",
                "",
                str(e)
            ])


# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)
print("SKEW CORRECTION COMPLETE")
print("=" * 60)

print(
    f"Corrected images: {OUTPUT_FOLDER}"
)

print(
    f"CSV results: {csv_path}"
)