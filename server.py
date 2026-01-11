import os
import io
import glob
import time
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import face_recognition
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

# ----------------------------
# Config (override with env vars)
# ----------------------------
KNOWN_DIR = os.getenv("KNOWN_DIR", "./known_faces")
TOLERANCE = float(os.getenv("TOLERANCE", "0.50"))  # lower => stricter (0.45-0.60 typical)
MODEL = os.getenv("MODEL", "hog")                  # "hog" (CPU) or "cnn" (GPU; slower on CPU)
MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))      # downscale large images for speed

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

app = FastAPI(title="Face Recognition Server", version="1.0")

_known_names: List[str] = []
_known_encodings: List[np.ndarray] = []
_last_loaded_ts: float = 0.0


def _iter_known_images(known_dir: str) -> List[str]:
    paths = []
    for ext in ALLOWED_EXT:
        paths.extend(glob.glob(os.path.join(known_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(known_dir, f"*{ext.upper()}")))
    paths.sort()
    return paths


def _safe_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name.strip()


def _load_image_bytes_to_rgb_np(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    img = img.convert("RGB")

    # Downscale if huge (faster face detection/encoding)
    w, h = img.size
    max_side = max(w, h)
    if max_side > MAX_SIDE:
        scale = MAX_SIDE / float(max_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h))

    return np.array(img)


def load_known_faces() -> Tuple[List[str], List[np.ndarray]]:
    if not os.path.isdir(KNOWN_DIR):
        raise RuntimeError(f"KNOWN_DIR does not exist: {os.path.abspath(KNOWN_DIR)}")

    names: List[str] = []
    encs: List[np.ndarray] = []

    paths = _iter_known_images(KNOWN_DIR)
    if not paths:
        print(f"[WARN] No known face images found in: {os.path.abspath(KNOWN_DIR)}")

    for p in paths:
        name = _safe_name_from_path(p)
        try:
            img = face_recognition.load_image_file(p)
            # For known images, we assume 1 face; take the first encoding found.
            enc_list = face_recognition.face_encodings(img)
            if len(enc_list) == 0:
                print(f"[WARN] No face found in known image: {p} (skipping)")
                continue

            if len(enc_list) > 1:
                print(f"[WARN] Multiple faces in known image: {p}. Using the first one.")

            names.append(name)
            encs.append(enc_list[0])
            print(f"[OK] Loaded known face: {name}  ({os.path.basename(p)})")
        except Exception as e:
            print(f"[WARN] Failed to process {p}: {e}")

    return names, encs


def ensure_loaded():
    global _known_names, _known_encodings, _last_loaded_ts
    if not _known_encodings:
        _known_names, _known_encodings = load_known_faces()
        _last_loaded_ts = time.time()


@app.on_event("startup")
def _startup():
    ensure_loaded()


@app.get("/health")
def health():
    ensure_loaded()
    return {
        "ok": True,
        "known_dir": os.path.abspath(KNOWN_DIR),
        "known_count": len(_known_names),
        "tolerance": TOLERANCE,
        "model": MODEL,
        "last_loaded_ts": _last_loaded_ts,
    }


@app.post("/reload")
def reload_known_faces():
    """Optional: call this after adding/removing known face images."""
    global _known_names, _known_encodings, _last_loaded_ts
    _known_names, _known_encodings = load_known_faces()
    _last_loaded_ts = time.time()
    return {"ok": True, "known_count": len(_known_names), "last_loaded_ts": _last_loaded_ts}


@app.post("/recognize", response_class=PlainTextResponse)
async def recognize(image: UploadFile = File(...)):
    """
    POST multipart/form-data with field name 'image'
    Returns: plain text -> recognized name or 'unknown'
    """
    ensure_loaded()
    if len(_known_encodings) == 0:
        raise HTTPException(status_code=500, detail="No known face encodings loaded.")

    # Basic content-type check (not perfect, but helpful)
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected image/*, got {image.content_type}")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    rgb = _load_image_bytes_to_rgb_np(image_bytes)

    # Detect faces in the uploaded image
    locations = face_recognition.face_locations(rgb, model=MODEL)
    if not locations:
        return "unknown"

    # Encode faces found (choose best match among them)
    encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
    if not encodings:
        return "unknown"

    best_name = "unknown"
    best_distance = 1e9

    known_encs_np = np.array(_known_encodings)

    for enc in encodings:
        # Compute distances to all known encodings
        dists = face_recognition.face_distance(known_encs_np, enc)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])

        if dist < best_distance:
            best_distance = dist
            best_name = _known_names[idx] if dist <= TOLERANCE else "unknown"

    return best_name


# Optional JSON endpoint (if you later want confidence/distance)
@app.post("/recognize_json")
async def recognize_json(image: UploadFile = File(...)):
    ensure_loaded()
    name = await recognize(image)  # uses the same logic
    return JSONResponse({"name": name})
