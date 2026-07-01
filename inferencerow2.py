import os
import cv2
import time
import threading
import numpy as np
from collections import deque
from ultralytics import YOLO

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


# =========================================================
# KONFIGURASI MODEL
# =========================================================

ROW_PT_PATH = "bestrow2lama.pt"     # model deteksi row 2 rack
BOX_PT_PATH = "bestkfs1juli.pt"      # model deteksi box

ROW_ONNX_PATH = "bestrow2lama.onnx"
BOX_ONNX_PATH = "bestkfs1juli.onnx"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0

ROW_CONF = 0.40
BOX_CONF = 0.50

# Box tetap dicek cukup sering, row dicek lebih jarang lalu di-cache.
BOX_INFER_EVERY_N = 4#berapa frame untuk infer box
ROW_INFER_EVERY_N = 4 #eberapa frame untuk infer row
ROW_CACHE_MAX_AGE = 40 #berapa frame row cache masih valid, kalau lebih dari ini harus infer ulang

# Stabilkan output occupancy agar tidak mudah flicker antar frame.
OUTPUT_HISTORY_SIZE = 5
OUTPUT_MIN_VOTES = 3

# Assignment box ke cell memakai overlap, bukan hanya titik tengah bbox.
MIN_BOX_CELL_OVERLAP_RATIO = 0.10

# Validasi HSV untuk memastikan box yang terdeteksi memang punya bukti warna
# di area cell rack, bukan hanya objek di belakang rack.
USE_HSV_VALIDATION = True
HSV_MIN_OVERLAP_MASK_RATIO = 0.5 #berapa minimum warna dalam rack
HSV_MIN_CELL_MASK_RATIO = 0.4 #berapa minimum box dalam rack
HSV_MIN_PATCH_AREA = 25

EMPTY_VALUE = 0

# FP16 hanya masuk akal kalau GPU CUDA tersedia.
# Kalau CPU dipaksa FP16, sering malah lambat/error.
USE_FP16 = CUDA_AVAILABLE

DEVICE = 0 if CUDA_AVAILABLE else "cpu"

# Ukuran input YOLO
IMGSZ = 640

# Kalau ONNX error, ubah False dulu
AUTO_EXPORT_ONNX = True

# Tampilkan window
SHOW_WINDOW = True


# =========================================================
# LABEL MODEL BOX
# =========================================================

labels = {
    0: 'FB1', 1: 'FB10', 2: 'FB12', 3: 'FB13',
    4: 'FB2', 5: 'FB3', 6: 'FB4', 7: 'FB5',
    8: 'FB6', 9: 'FB7', 10: 'FB8', 11: 'FB9',

    12: 'FR1', 13: 'FR10', 14: 'FR11', 15: 'FR12',
    16: 'FR13', 17: 'FR14', 18: 'FR15',
    19: 'FR2', 20: 'FR3', 21: 'FR4',
    22: 'FR5', 23: 'FR6', 24: 'FR7',
    25: 'FR8', 26: 'FR9',

    27: 'Fb11', 28: 'Fb14', 29: 'Fb15',

    30: 'RB1', 31: 'RB10', 32: 'RB11',
    33: 'RB13', 34: 'RB14',
    35: 'RB2', 36: 'RB3', 37: 'RB4',
    38: 'RB5', 39: 'RB6', 40: 'RB7',
    41: 'RB8', 42: 'RB9',

    43: 'RR1', 44: 'RR10', 45: 'RR11',
    46: 'RR12', 47: 'RR13', 48: 'RR14',
    49: 'RR15',
    50: 'RR2', 51: 'RR3', 52: 'RR4',
    53: 'RR5', 54: 'RR6', 55: 'RR7',
    56: 'RR8', 57: 'RR9',

    58: 'Rb12', 59: 'Rb15'
}


# =========================================================
# MAPPING CLASS VECTOR
# =========================================================
# Output akhir:
# 0 = tidak ada box terdeteksi di cell row
# 1 = class utama sesuai MATRIX_MODE
# 2 = class sisanya
# FB / FR tetap diabaikan untuk row2; hanya RB / RR yang dipakai.
# =========================================================

NUM_CLASSES = 60
IGNORE_VALUE = -99

# Ubah variabel ini langsung sesuai kebutuhan:
# "blue" -> RB = 1, RR = 2
# "red"  -> RR = 1, RB = 2
MATRIX_MODE = "red"

MODE_PRIMARY_PREFIXES = {
    "blue": ("RB","Rb"),
    "red": ("RR",),
}

VALID_ROW2_PREFIXES = ("RB","Rb", "RR")
COLOR_PREFIXES = {
    "blue": ("RB","Rb"),
    "red": ("RR",),
}

HSV_RANGES = {
    "blue": [
        ((90, 45, 40), (135, 255, 255)),
    ],
    "red": [
        ((0, 45, 40), (12, 255, 255)),
        ((168, 45, 40), (179, 255, 255)),
    ],
}

if MATRIX_MODE not in MODE_PRIMARY_PREFIXES:
    raise ValueError(
        f"MATRIX_MODE harus salah satu dari {tuple(MODE_PRIMARY_PREFIXES)}, "
        f"bukan {MATRIX_MODE!r}"
    )

class_to_value = np.full(NUM_CLASSES, IGNORE_VALUE, dtype=np.int32)
class_to_name = np.array([labels.get(i, f"class_{i}") for i in range(NUM_CLASSES)])

for cls_id, name in labels.items():
    name_upper = name.upper()

    if name_upper.startswith(VALID_ROW2_PREFIXES):
        if name_upper.startswith(MODE_PRIMARY_PREFIXES[MATRIX_MODE]):
            class_to_value[cls_id] = 1
        else:
            class_to_value[cls_id] = 2
    else:
        class_to_value[cls_id] = IGNORE_VALUE

STABLE_VALUES = tuple(
    int(value)
    for value in np.unique(class_to_value)
    if value not in (IGNORE_VALUE, EMPTY_VALUE)
)

CELL_VALUE_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 165, 255),
}

HSV_REJECTED_COLOR = (160, 160, 160)


# =========================================================
# PROFILER MOVING AVERAGE
# =========================================================

class MovingProfiler:
    def __init__(self, window=30):
        self.data = {
            "pre": deque(maxlen=window),
            "infer": deque(maxlen=window),
            "post": deque(maxlen=window),
            "render": deque(maxlen=window),
            "total": deque(maxlen=window),
        }
        self.lock = threading.Lock()

    def add(self, key, value_ms):
        with self.lock:
            if key in self.data:
                self.data[key].append(value_ms)

    def avg(self, key):
        with self.lock:
            arr = self.data.get(key, [])
            if len(arr) == 0:
                return 0.0
            return sum(arr) / len(arr)

    def text(self):
        return (
            f"pre {self.avg('pre'):.1f}ms | "
            f"infer {self.avg('infer'):.1f}ms | "
            f"post {self.avg('post'):.1f}ms | "
            f"render {self.avg('render'):.1f}ms | "
            f"total {self.avg('total'):.1f}ms"
        )


profiler = MovingProfiler(window=30)


# =========================================================
# GLOBAL STATE THREAD
# =========================================================

latest_frame = None
latest_frame_id = 0
latest_result = None

frame_lock = threading.Lock()
result_lock = threading.Lock()

running = True


# =========================================================
# AUTO EXPORT ONNX
# =========================================================

def export_to_onnx_if_needed(pt_path, onnx_path):
    if os.path.exists(onnx_path):
        print(f"[OK] ONNX sudah ada: {onnx_path}")
        return onnx_path

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model .pt tidak ditemukan: {pt_path}")

    if not AUTO_EXPORT_ONNX:
        print(f"[INFO] AUTO_EXPORT_ONNX=False, pakai .pt: {pt_path}")
        return pt_path

    print(f"[EXPORT] {pt_path} -> ONNX")
    print(f"[EXPORT] FP16: {USE_FP16}, DEVICE: {DEVICE}")

    model = YOLO(pt_path)

    exported_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        half=USE_FP16,
        simplify=True,
        opset=12,
        dynamic=False,
        device=DEVICE
    )

    if exported_path != onnx_path and os.path.exists(exported_path):
        try:
            os.replace(exported_path, onnx_path)
        except Exception:
            onnx_path = exported_path

    print(f"[OK] Export selesai: {onnx_path}")
    return onnx_path


# =========================================================
# LOAD MODEL
# =========================================================

def load_models():
    row_path = export_to_onnx_if_needed(ROW_PT_PATH, ROW_ONNX_PATH)
    box_path = export_to_onnx_if_needed(BOX_PT_PATH, BOX_ONNX_PATH)

    print("[LOAD] Loading ROW model:", row_path)
    row_model = YOLO(row_path, task="detect")

    print("[LOAD] Loading BOX model:", box_path)
    box_model = YOLO(box_path, task="detect")

    return row_model, box_model


# =========================================================
# CAMERA THREAD
# =========================================================

def camera_thread_func():
    global latest_frame, latest_frame_id, running

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka.")
        running = False
        return

    local_id = 0

    while running:
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.005)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        local_id += 1

        with frame_lock:
            latest_frame = frame
            latest_frame_id = local_id

    cap.release()


# =========================================================
# YOLO RESULT TO NUMPY
# =========================================================

def yolo_result_to_numpy(result):
    """
    Return:
    xyxy: Nx4 int32
    conf: N float32
    cls : N int32
    """

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )

    boxes = result.boxes

    xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
    conf = boxes.conf.cpu().numpy().astype(np.float32)
    cls = boxes.cls.cpu().numpy().astype(np.int32)

    return xyxy, conf, cls


def format_row_output(output):
    return " ".join(str(int(value)) for value in output)


# =========================================================
# INFERENCE CORE HELPERS
# =========================================================

class InferenceState:
    def __init__(self):
        self.row_bbox = None
        self.row_conf = 0.0
        self.cells = None
        self.row_frame_id = -1
        self.last_row_check_id = -1
        self.output_history = deque(maxlen=OUTPUT_HISTORY_SIZE)

    def has_cached_row(self, frame_id):
        if self.row_bbox is None or self.cells is None:
            return False
        return (frame_id - self.row_frame_id) <= ROW_CACHE_MAX_AGE

    def update_row(self, row_bbox, row_conf, cells, frame_id):
        self.row_bbox = row_bbox.copy()
        self.row_conf = float(row_conf)
        self.cells = cells.copy()
        self.row_frame_id = frame_id

    def clear_row(self):
        self.row_bbox = None
        self.row_conf = 0.0
        self.cells = None
        self.row_frame_id = -1
        self.output_history.clear()

    def stabilize_output(self, raw_output):
        self.output_history.append(raw_output.copy())

        if len(self.output_history) < OUTPUT_MIN_VOTES:
            return raw_output.copy()

        stacked = np.stack(self.output_history, axis=0)
        stable = np.full(raw_output.shape, EMPTY_VALUE, dtype=np.int32)

        for value in STABLE_VALUES:
            votes = np.sum(stacked == value, axis=0)
            stable[votes >= OUTPUT_MIN_VOTES] = value

        return stable


def make_empty_result(raw_output=None):
    if raw_output is None:
        raw_output = np.full((3,), EMPTY_VALUE, dtype=np.int32)

    return {
        "detected": False,
        "row_bbox": None,
        "row_conf": 0.0,
        "cells": None,
        "boxes": np.empty((0, 4), dtype=np.int32),
        "box_conf": np.empty((0,), dtype=np.float32),
        "box_cls": np.empty((0,), dtype=np.int32),
        "box_values": np.empty((0,), dtype=np.int32),
        "box_names": [],
        "hsv_overlap_ratios": np.empty((0,), dtype=np.float32),
        "hsv_cell_ratios": np.empty((0,), dtype=np.float32),
        "hsv_passed": np.empty((0,), dtype=bool),
        "raw_output": raw_output,
        "output": raw_output.copy(),
        "row_from_cache": False,
        "row_inferred": False
    }


def build_row_cells(row_bbox):
    x1, y1, x2, y2 = row_bbox
    x_edges = np.linspace(x1, x2, 4).astype(np.int32)

    return np.array([
        [x_edges[0], y1, x_edges[1], y2],
        [x_edges[1], y1, x_edges[2], y2],
        [x_edges[2], y1, x_edges[3], y2],
    ], dtype=np.int32)


def best_row_from_result(result, frame_shape):
    h, w = frame_shape[:2]
    row_xyxy, row_conf, _ = yolo_result_to_numpy(result)

    if len(row_xyxy) == 0:
        return None, 0.0, None

    best_idx = int(np.argmax(row_conf))
    x1, y1, x2, y2 = row_xyxy[best_idx]

    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None, 0.0, None

    row_bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
    cells = build_row_cells(row_bbox)

    return row_bbox, float(row_conf[best_idx]), cells


def class_color_family(cls_id):
    name_upper = class_to_name[int(cls_id)].upper()

    for family, prefixes in COLOR_PREFIXES.items():
        if name_upper.startswith(prefixes):
            return family

    return None


def hsv_mask_ratio(frame, rect, family):
    if family not in HSV_RANGES:
        return 0.0

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in rect]

    x1 = int(np.clip(x1, 0, w))
    y1 = int(np.clip(y1, 0, h))
    x2 = int(np.clip(x2, 0, w))
    y2 = int(np.clip(y2, 0, h))

    area = (x2 - x1) * (y2 - y1)
    if area < HSV_MIN_PATCH_AREA:
        return 0.0

    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES[family]:
        lower_arr = np.array(lower, dtype=np.uint8)
        upper_arr = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_arr, upper_arr))

    return float(cv2.countNonZero(mask)) / float(area)


def validate_box_hsv(frame, box, cell, family):
    if not USE_HSV_VALIDATION:
        return True, 1.0, 1.0

    bx1, by1, bx2, by2 = box
    cx1, cy1, cx2, cy2 = cell

    overlap_rect = np.array([
        max(bx1, cx1),
        max(by1, cy1),
        min(bx2, cx2),
        min(by2, cy2),
    ], dtype=np.int32)

    overlap_ratio = hsv_mask_ratio(frame, overlap_rect, family)
    cell_ratio = hsv_mask_ratio(frame, cell, family)

    passed = (
        overlap_ratio >= HSV_MIN_OVERLAP_MASK_RATIO and
        cell_ratio >= HSV_MIN_CELL_MASK_RATIO
    )

    return passed, overlap_ratio, cell_ratio


def mark_cells_by_overlap(output, box_xyxy, box_values, box_cls, cells, frame):
    hsv_overlap_ratios = np.zeros((len(box_xyxy),), dtype=np.float32)
    hsv_cell_ratios = np.zeros((len(box_xyxy),), dtype=np.float32)
    hsv_passed = np.zeros((len(box_xyxy),), dtype=bool)

    for idx, (box, value, cls_id) in enumerate(zip(box_xyxy, box_values, box_cls)):
        bx1, by1, bx2, by2 = box
        box_area = max(1, int((bx2 - bx1) * (by2 - by1)))

        ix1 = np.maximum(bx1, cells[:, 0])
        iy1 = np.maximum(by1, cells[:, 1])
        ix2 = np.minimum(bx2, cells[:, 2])
        iy2 = np.minimum(by2, cells[:, 3])

        inter_w = np.maximum(0, ix2 - ix1)
        inter_h = np.maximum(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        best_cell = int(np.argmax(inter_area))
        overlap_ratio = float(inter_area[best_cell]) / float(box_area)

        if inter_area[best_cell] > 0 and overlap_ratio >= MIN_BOX_CELL_OVERLAP_RATIO:
            family = class_color_family(cls_id)
            hsv_ok, hsv_overlap_ratio, hsv_cell_ratio = validate_box_hsv(
                frame,
                box,
                cells[best_cell],
                family
            )
            hsv_overlap_ratios[idx] = hsv_overlap_ratio
            hsv_cell_ratios[idx] = hsv_cell_ratio

            if not hsv_ok:
                continue

            hsv_passed[idx] = True
            output[best_cell] = int(value)

    return hsv_overlap_ratios, hsv_cell_ratios, hsv_passed


# =========================================================
# INFERENCE CORE
# =========================================================

def run_inference(frame, row_model, box_model, state, frame_id):
    """
    Output result dict:
    {
        row_bbox,
        row_conf,
        cells,
        boxes,
        raw_output: output 3 cell sebelum temporal smoothing,
        output
    }
    """

    t_total0 = time.perf_counter()

    # =====================================================
    # PREPROCESSING
    # =====================================================
    t0 = time.perf_counter()

    infer_frame = frame

    t1 = time.perf_counter()
    profiler.add("pre", (t1 - t0) * 1000)

    # =====================================================
    # INFERENCE ROW MODEL BERJADWAL + CACHE
    # =====================================================
    row_inferred = False
    row_infer_ms = 0.0

    should_infer_row = (
        not state.has_cached_row(frame_id) or
        frame_id - state.last_row_check_id >= ROW_INFER_EVERY_N
    )

    if should_infer_row:
        row_inferred = True
        state.last_row_check_id = frame_id

        t2 = time.perf_counter()
        row_results = row_model.predict(
            infer_frame,
            imgsz=IMGSZ,
            conf=ROW_CONF,
            device=DEVICE,
            half=USE_FP16,
            verbose=False
        )
        t3 = time.perf_counter()
        row_infer_ms = (t3 - t2) * 1000

        row_bbox, best_row_conf, cells = best_row_from_result(
            row_results[0],
            frame.shape
        )

        if row_bbox is not None:
            state.update_row(row_bbox, best_row_conf, cells, frame_id)
        elif not state.has_cached_row(frame_id):
            state.clear_row()

    t_post0 = time.perf_counter()

    if not state.has_cached_row(frame_id):
        output = np.full((3,), EMPTY_VALUE, dtype=np.int32)
        state.output_history.clear()

        t_post1 = time.perf_counter()

        profiler.add("infer", row_infer_ms)
        profiler.add("post", (t_post1 - t_post0) * 1000)
        profiler.add("total", (time.perf_counter() - t_total0) * 1000)

        result = make_empty_result(output)
        result["row_inferred"] = row_inferred
        return result

    row_bbox = state.row_bbox.copy()
    cells = state.cells.copy()
    best_row_conf = state.row_conf
    row_from_cache = not row_inferred or state.row_frame_id != frame_id

    x1, y1, x2, y2 = row_bbox
    roi = frame[y1:y2, x1:x2]

    t_post1 = time.perf_counter()

    # =====================================================
    # INFERENCE BOX MODEL HANYA DI ROI
    # =====================================================
    t4 = time.perf_counter()

    box_results = box_model.predict(
        roi,
        imgsz=IMGSZ,
        conf=BOX_CONF,
        device=DEVICE,
        half=USE_FP16,
        verbose=False
    )

    t5 = time.perf_counter()

    profiler.add("infer", row_infer_ms + ((t5 - t4) * 1000))

    # =====================================================
    # POSTPROCESS BOX + ASSIGNMENT BERBASIS OVERLAP
    # =====================================================
    t_post2 = time.perf_counter()

    box_xyxy, box_conf, box_cls = yolo_result_to_numpy(box_results[0])

    output = np.full((3,), EMPTY_VALUE, dtype=np.int32)

    box_values = np.empty((0,), dtype=np.int32)
    box_names = []
    hsv_overlap_ratios = np.empty((0,), dtype=np.float32)
    hsv_cell_ratios = np.empty((0,), dtype=np.float32)
    hsv_passed = np.empty((0,), dtype=bool)

    if len(box_xyxy) > 0:
        box_xyxy[:, [0, 2]] += x1
        box_xyxy[:, [1, 3]] += y1

        valid_cls_mask = (box_cls >= 0) & (box_cls < NUM_CLASSES)
        box_xyxy = box_xyxy[valid_cls_mask]
        box_conf = box_conf[valid_cls_mask]
        box_cls = box_cls[valid_cls_mask]

        if len(box_xyxy) > 0:
            box_values = class_to_value[box_cls]
            valid_value_mask = box_values != IGNORE_VALUE

            box_xyxy = box_xyxy[valid_value_mask]
            box_conf = box_conf[valid_value_mask]
            box_cls = box_cls[valid_value_mask]
            box_values = box_values[valid_value_mask]

            if len(box_xyxy) > 0:
                inside_roi = (
                    (box_xyxy[:, 2] > x1) & (box_xyxy[:, 0] < x2) &
                    (box_xyxy[:, 3] > y1) & (box_xyxy[:, 1] < y2)
                )

                box_xyxy = box_xyxy[inside_roi]
                box_conf = box_conf[inside_roi]
                box_cls = box_cls[inside_roi]
                box_values = box_values[inside_roi]

                if len(box_xyxy) > 0:
                    box_names = class_to_name[box_cls].tolist()
                    hsv_overlap_ratios, hsv_cell_ratios, hsv_passed = mark_cells_by_overlap(
                        output,
                        box_xyxy,
                        box_values,
                        box_cls,
                        cells,
                        frame
                    )

    t_post3 = time.perf_counter()

    stable_output = state.stabilize_output(output)

    profiler.add("post", ((t_post1 - t_post0) + (t_post3 - t_post2)) * 1000)
    profiler.add("total", (time.perf_counter() - t_total0) * 1000)

    return {
        "detected": True,
        "row_bbox": row_bbox,
        "row_conf": best_row_conf,
        "cells": cells,
        "boxes": box_xyxy,
        "box_conf": box_conf,
        "box_cls": box_cls,
        "box_values": box_values,
        "box_names": box_names,
        "hsv_overlap_ratios": hsv_overlap_ratios,
        "hsv_cell_ratios": hsv_cell_ratios,
        "hsv_passed": hsv_passed,
        "raw_output": output,
        "output": stable_output,
        "row_from_cache": row_from_cache,
        "row_inferred": row_inferred
    }


# =========================================================
# INFERENCE THREAD
# =========================================================

def inference_thread_func(row_model, box_model):
    global latest_result, running

    last_processed_id = -1
    state = InferenceState()

    while running:
        with frame_lock:
            if latest_frame is None:
                frame_copy = None
                frame_id = None
            else:
                frame_id = latest_frame_id
                frame_copy = latest_frame.copy()

        if frame_copy is None:
            time.sleep(0.005)
            continue

        if frame_id == last_processed_id:
            time.sleep(0.002)
            continue

        if frame_id % BOX_INFER_EVERY_N != 0:
            time.sleep(0.001)
            continue

        last_processed_id = frame_id

        result = run_inference(frame_copy, row_model, box_model, state, frame_id)

        with result_lock:
            latest_result = result


# =========================================================
# RENDER
# =========================================================

def render_frame(frame, result):
    t0 = time.perf_counter()

    h, w = frame.shape[:2]

    if result is None:
        cv2.putText(
            frame,
            "Waiting inference...",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        profiler.add("render", (time.perf_counter() - t0) * 1000)
        return frame

    output = result["output"]

    if not result["detected"] or result["row_bbox"] is None:
        cv2.putText(
            frame,
            "ROW2 NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"ROW2 OUTPUT: {format_row_output(output)}",
            (20, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            profiler.text(),
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        profiler.add("render", (time.perf_counter() - t0) * 1000)
        return frame

    row_bbox = result["row_bbox"]
    cells = result["cells"]
    boxes = result["boxes"]
    box_conf = result["box_conf"]
    box_values = result["box_values"]
    box_names = result["box_names"]
    hsv_overlap_ratios = result.get("hsv_overlap_ratios", [])

    x1, y1, x2, y2 = row_bbox
    row_label = "ROW2 CACHE" if result.get("row_from_cache") else "ROW2"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(
        frame,
        f"{row_label} {result['row_conf']:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )

    for i in range(3):
        cx1, cy1, cx2, cy2 = cells[i]
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

        cell_value = int(output[i])
        color = CELL_VALUE_COLORS.get(cell_value, (0, 255, 255))

        cv2.putText(
            frame,
            str(cell_value),
            (cx1 + 10, cy2 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = boxes[i]
        value = int(box_values[i])
        conf = float(box_conf[i])
        name = box_names[i]
        hsv_ratio = (
            float(hsv_overlap_ratios[i])
            if i < len(hsv_overlap_ratios)
            else 0.0
        )

        color = CELL_VALUE_COLORS.get(value, (255, 255, 255))

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

        cx = int((bx1 + bx2) / 2)
        cy = int((by1 + by2) / 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)

        cv2.putText(
            frame,
            f"{name}:{value} {conf:.2f} hsv {hsv_ratio:.2f}",
            (bx1, max(20, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    cv2.putText(
        frame,
        f"ROW2 OUTPUT: {format_row_output(output)}",
        (20, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        profiler.text(),
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    profiler.add("render", (time.perf_counter() - t0) * 1000)

    return frame


# =========================================================
# MAIN
# =========================================================

def main():
    global running

    print("======================================")
    print("ROW2 + BOX OPTIMIZED INFERENCE")
    print("======================================")
    print(f"CUDA_AVAILABLE : {CUDA_AVAILABLE}")
    print(f"DEVICE         : {DEVICE}")
    print(f"USE_FP16       : {USE_FP16}")
    print(f"BOX_INFER_EVERY_N   : {BOX_INFER_EVERY_N}")
    print(f"ROW_INFER_EVERY_N   : {ROW_INFER_EVERY_N}")
    print(f"ROW_CACHE_MAX_AGE   : {ROW_CACHE_MAX_AGE}")
    print(f"OUTPUT_HISTORY_SIZE : {OUTPUT_HISTORY_SIZE}")
    print(f"OUTPUT_MIN_VOTES    : {OUTPUT_MIN_VOTES}")
    print(f"MATRIX_MODE         : {MATRIX_MODE}")
    print(f"USE_HSV_VALIDATION  : {USE_HSV_VALIDATION}")
    print(f"HSV_OVERLAP_RATIO   : {HSV_MIN_OVERLAP_MASK_RATIO}")
    print(f"HSV_CELL_RATIO      : {HSV_MIN_CELL_MASK_RATIO}")
    print(
        "CLASS_TO_VALUE      : "
        + ", ".join(
            f"{class_to_name[i]}={int(class_to_value[i])}"
            for i in range(NUM_CLASSES)
            if int(class_to_value[i]) != IGNORE_VALUE
        )
    )
    print("======================================")

    row_model, box_model = load_models()

    cam_thread = threading.Thread(target=camera_thread_func, daemon=True)
    infer_thread = threading.Thread(
        target=inference_thread_func,
        args=(row_model, box_model),
        daemon=True
    )

    cam_thread.start()
    infer_thread.start()

    last_print = 0

    try:
        while running:
            with frame_lock:
                if latest_frame is None:
                    frame = None
                else:
                    frame = latest_frame.copy()

            if frame is None:
                time.sleep(0.005)
                continue

            with result_lock:
                result = latest_result

            rendered = render_frame(frame, result)

            now = time.time()

            if result is not None and now - last_print > 0.5:
                output = result["output"]
                print(
                    f"ROW2: {format_row_output(output)} | "
                    f"{profiler.text()}"
                )
                last_print = now

            if SHOW_WINDOW:
                cv2.imshow("Improved ROW2 Rack + Box Detection", rendered)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    running = False
                    break
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        running = False

    running = False
    time.sleep(0.2)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
