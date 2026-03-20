import os
import uuid
import io
import math
import requests
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# ------------------ Path Configuration ------------------
BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
OUTPUT_FOLDER = STATIC_DIR / "outputs"
SEGMENT_FOLDER = STATIC_DIR / "segments"

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, SEGMENT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# ------------------ Debugging ------------------
print(f"--- PATH DEBUGGING ---")
print(f"BASE_DIR:      {BASE_DIR}")
print(f"TEMPLATES_DIR: {TEMPLATES_DIR}")
print(f"Files found in templates: {list(TEMPLATES_DIR.glob('*'))}")
print(f"----------------------")

# ------------------ App & Mounts ------------------
app = FastAPI(title="SAR Image Segmentation")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ------------------ Models ------------------
COLOR_MODEL_PATH = BASE_DIR / "best_model.onnx"
SEG_MODEL_PATH = BASE_DIR / "segmentation_model.onnx"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

try:
    color_session = ort.InferenceSession(str(COLOR_MODEL_PATH), providers=["CPUExecutionProvider"])
    seg_session = ort.InferenceSession(str(SEG_MODEL_PATH), providers=["CPUExecutionProvider"])
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

# ------------------ Utils ------------------
def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def colorize_image(image_path, output_path):
    img = Image.open(image_path).convert("L").resize((256, 256))
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=(0, -1))

    input_name = color_session.get_inputs()[0].name
    output_name = color_session.get_outputs()[0].name
    pred = color_session.run([output_name], {input_name: img})[0]

    out = ((pred[0] + 1) * 127.5).astype(np.uint8)
    Image.fromarray(out).save(output_path)


def segment_image(image_path, segmented_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    input_name = seg_session.get_inputs()[0].name
    output = seg_session.run(None, {input_name: img})[0]
    pred = np.argmax(output.squeeze(), axis=-1)

    color_map = {
        0: (255, 0, 0), 1: (0, 255, 0), 2: (160, 82, 45),
        3: (0, 100, 0), 4: (0, 255, 255), 5: (255, 255, 0),
        6: (128, 128, 128)
    }

    label_names = {
        0: "Urban Land",
        1: "Agriculture Land",
        2: "Rangeland",
        3: "Forest Land",
        4: "Water",
        5: "Barren Land",
        6: "Unknown"
    }

    seg_img = np.zeros((256, 256, 3), dtype=np.uint8)

    for k, v in color_map.items():
        seg_img[pred == k] = v

    Image.fromarray(seg_img).save(segmented_path)

    total = pred.size
    return [f"{label_names[k]}: {(pred == k).sum() / total * 100:.2f}%" for k in label_names]


# ------------------ FREE Satellite Image (Replaces Google Maps) ------------------
def lat_lon_to_tile(lat: float, lon: float, zoom: int):
    """Convert lat/lon to XYZ tile coordinates."""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int(
        (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi)
        / 2 * n
    )
    return x, y


def fetch_tile(zoom: int, x: int, y: int) -> Image.Image:
    """
    Fetch a single 256x256 satellite tile from ESRI World Imagery.
    Completely free, no API key required.
    """
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    )
    headers = {"User-Agent": "SAR-Segmentation-App/1.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def get_satellite_image(lat: float, lon: float, zoom: int = 16, grid: int = 3) -> Image.Image:
    """
    Stitch a grid of tiles (default 3x3) around the given coordinate
    to produce a larger image (~768x768), then crop to 640x640.

    Args:
        lat, lon : Target coordinates
        zoom     : Zoom level (16 = neighbourhood scale, 17 = street scale)
        grid     : How many tiles in each direction (3 = 3x3 grid)
    """
    center_x, center_y = lat_lon_to_tile(lat, lon, zoom)
    half = grid // 2

    tile_size = 256
    canvas = Image.new("RGB", (tile_size * grid, tile_size * grid))

    for row in range(grid):
        for col in range(grid):
            tx = center_x - half + col
            ty = center_y - half + row
            try:
                tile = fetch_tile(zoom, tx, ty)
            except Exception as e:
                print(f"Warning: Could not fetch tile ({tx},{ty}): {e}")
                tile = Image.new("RGB", (tile_size, tile_size), (128, 128, 128))
            canvas.paste(tile, (col * tile_size, row * tile_size))

    # Crop the stitched image to 640x640 from the center
    canvas_w, canvas_h = canvas.size
    left   = (canvas_w - 640) // 2
    top    = (canvas_h - 640) // 2
    right  = left + 640
    bottom = top  + 640
    return canvas.crop((left, top, right, bottom))


# ------------------ Routes ------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_image(request: Request, image: UploadFile = File(...)):
    if not allowed_file(image.filename):
        raise HTTPException(400, "Invalid file type")

    file_id = f"{uuid.uuid4().hex}.png"

    upload_path       = UPLOAD_FOLDER / file_id
    colorized_filename = f"colorized_{file_id}"
    segmented_filename = f"segmented_{file_id}"
    colorized_path    = OUTPUT_FOLDER / colorized_filename
    segmented_path    = SEGMENT_FOLDER / segmented_filename

    with open(upload_path, "wb") as f:
        f.write(await image.read())

    colorize_image(upload_path, colorized_path)
    percentages = segment_image(colorized_path, segmented_path)

    return templates.TemplateResponse(
        "segmentation_result.html",
        {
            "request": request,
            "colorized_filename": f"outputs/{colorized_filename}",
            "segmented_filename": f"segments/{segmented_filename}",
            "percentages": percentages,
        },
    )


@app.post("/location_segment")
def location_segment(lat: float = Form(...), lng: float = Form(...)):
    """
    Fetches a free satellite image from ESRI World Imagery for the given
    coordinates, then runs segmentation on it.
    No API key required.
    """
    try:
        img = get_satellite_image(lat, lng, zoom=16, grid=3)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch satellite image: {str(e)}")

    file_id            = uuid.uuid4().hex
    colorized_filename = f"colorized_{file_id}.png"
    segmented_filename = f"segmented_{file_id}.png"
    colorized_path     = OUTPUT_FOLDER / colorized_filename
    segmented_path     = SEGMENT_FOLDER / segmented_filename

    img.save(colorized_path)
    percentages = segment_image(colorized_path, segmented_path)

    return JSONResponse({
        "success":    True,
        "colorized":  f"outputs/{colorized_filename}",
        "segmented":  f"segments/{segmented_filename}",
        "percentages": percentages,
    })