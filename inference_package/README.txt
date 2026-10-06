IMAGE RESTORER - FOLDER INFERENCE
=================================================
Restores a folder of document images and writes the results to another folder, keeping the
sub-folder structure. No web app, no server: one command, one folder in, one folder out.

Each image is checked and only what is found is corrected:

    dewarp detection  -> dewarping        (UVDoc)
    shadow detection  -> shadow removal   (DocRes)
    blur detection    -> deblurring       (DocRes)
    resolution check  -> 2x upscaling     (Real-ESRGAN, if a side is below 1200 px)
    enhancement       -> lighting, background and colour (DocRes, every image)
    damage detection  -> inpainting       (big-LaMa, tears / missing paper)


WHAT IS IN THIS FOLDER
  inference.py       run it on a folder (this is the script you use)
  restoration.py     the pipeline: models, checks and corrections
  tear_mask_debug.py finds tears / missing paper
  lama_inpaint.py    fills them with big-LaMa
  models\            all the model weights (~570 MB)
  docres\            the DocRes network definition
  uvdoc\             the UVDoc network definition and its weights
  requirements.txt   Python packages


REQUIREMENTS
  * Windows 10/11 64-bit (Linux and macOS work too; use forward slashes)
  * Python 3.12 or newer, 64-bit   https://www.python.org/downloads/
  * An NVIDIA GPU is strongly recommended: about 20-60 s per image with one,
    several minutes per image without. 8 GB of GPU memory is enough.


INSTALL (once, about 3 GB of packages)
  Open a terminal in this folder and run:

      python -m venv .venv
      .venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

  No NVIDIA GPU? Use the CPU build instead:

      .venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

  (then remove the "+cu126" from the torch and torchvision lines in requirements.txt first)


RUN
      .venv\Scripts\python inference.py  C:\path\to\input_folder  C:\path\to\output_folder

  The first lines tell you which models loaded and whether the GPU is being used:

      Device:       GPU Quadro P4000 (8.0 GB, CUDA 12.6)     <- or "Device: CPU"

  Then one line per image:

      [3/10] Grave_A\page3.jpg: Shadow removed, then Blur corrected, then Enhanced  (61.4 s)


WHAT YOU GET
  * The output folder mirrors the input, sub-folders and file names unchanged:

        input                          output
        Grave_A\page1.jpg       ->     Grave_A\page1.jpg
        Grave_B\2024\page9.tif  ->     Grave_B\2024\page9.tif

  * report.csv in the output folder: one row per image with the resolution, the blur and shadow
    probabilities, the dewarp score, the damaged percentage, what was corrected and how long it took.
  * Supported: .jpg .jpeg .png .bmp .tif .tiff .webp .j2k .jp2
    Anything else in the folder is ignored, and one unreadable image does not stop the run.


OPTIONS
  --overwrite        redo images that already have an output (without it, finished images are
                     skipped, so an interrupted run can simply be started again)
  --checks ...       run only some steps, e.g.
                         --checks appearance              only the enhancement (fastest)
                         --checks shadow blur appearance  no dewarping, upscaling or inpainting
                     Choices: dewrap shadow blur upscale appearance inpaint


SETTINGS
  The CONFIG block at the top of restoration.py holds everything worth changing:

      device              "auto" (GPU if there is one), or force "cuda" / "cpu"
      upscale_below_px    1200   upscale when a side is smaller than this
      shadow_threshold    0.5    shadow probability needed to remove a shadow
      blur_threshold      0.5    blur probability needed to deblur
      dewarp_threshold    0.02   bending needed before a page is flattened
      deshadow_max_side   1600   lowering these two to 1024 is about 40% faster
      appearance_max_side 1600     overall, with a very small difference in the result
      docres_tile_gpu     768    tile size on the GPU; raise it if you have more than 8 GB


HOW LONG IT TAKES
  Measured on a Quadro P4000 (8 GB) with 1300x1700 scans:
      about 20 s per image if it only needs enhancement
      about 60 s per image if it needs shadow removal and deblurring as well
      so roughly 1.5-2 hours for 500 images
  A recent 24-48 GB GPU is around 10x faster.
