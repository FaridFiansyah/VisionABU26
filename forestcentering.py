from ultralytics import YOLO
from collections import deque
import torch
import cv2
import numpy as np
import threading
import time

INFER_EVERY_N   = 2
PROFILE_WINDOW  = 30
CAM_BUFFER_SIZE = 1     

SMOOTHING_ALPHA = 0.8
INPUT_SIZE = 320

CONFIDENCE_THRESHOLD = 0.3
MIN_AREA  = 3000
MIN_RATIO = 0.5
MAX_RATIO = 2.0
MAX_DET   = 2

LABEL_MAP = {
    0: "B",
    1: "FB",
    2: "FR",
    3: "R"
}

MODE_LABELS = {
    "blue": {"B"},
    "red":  {"R"}
}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE PROFILER
# ─────────────────────────────────────────────────────────────────────────────

class StageProfiler:
    stages = ["Capture", "Inference", "Post-Processing"]

    def __init__(self, window: int = PROFILE_WINDOW):
        self.window = window
        self.times = {stage: deque(maxlen=window) for stage in self.stages}

    def tick(self):
        return time.perf_counter()

    def record(self, stage: str, start: float):
        ms = (time.perf_counter() - start) * 1000
        if stage in self.times:
            self.times[stage].append(ms)
        return ms

    def avg(self, stage: str):
        buf = self.times.get(stage)
        if not buf:
            return 0.0
        return float(np.mean(list(buf), dtype=np.float32))

    def draw_overlay(self, frame: np.ndarray, origin=(10, 60)) -> np.ndarray:
        x, y = origin
        for i, s in enumerate(self.stages):
            cv2.putText(
                frame, f"{s}: {self.avg(s):.1f}ms",
                (x, y + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (200, 200, 0), 1, cv2.LINE_AA
            )

        fps = 1000.0 / max(self.avg("Inference"), 1)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

        return frame
    
class SharedState:
    def __init__(self):
        self._lock            = threading.Lock()
        self._latest_frame    = None
        self._latest_frame_id = 0      # counter integer
        self._latest_bbox     = None
        self._latest_result   = None
        self.stop_event       = threading.Event()

    def set_frame(self, frame: np.ndarray):
        with self._lock:
            self._latest_frame = frame
            self._latest_frame_id += 1

    def get_frame(self):
        with self._lock:
            return self._latest_frame, self._latest_frame_id

    def set_result(self, bbox, result):
        with self._lock:
            self._latest_bbox   = bbox
            self._latest_result = result

    def get_result(self):
        with self._lock:
            return self._latest_bbox, self._latest_result

    def request_stop(self):
        self.stop_event.set()

    def is_running(self):
        return not self.stop_event.is_set()

def best_confidence_index(confidences: np.ndarray) -> int:
    return int(np.argmax(confidences))


def filter_boxes(confs, xyxys, clsids, target_labels):
    """Filter bbox: label + area minimum + rasio w/h."""
    label_mask = np.array([LABEL_MAP.get(c, "") in target_labels for c in clsids])
    if not label_mask.any():
        return None, None

    confs = confs[label_mask]
    xyxys = xyxys[label_mask]

    w      = xyxys[:, 2] - xyxys[:, 0]
    h      = xyxys[:, 3] - xyxys[:, 1]
    areas  = w * h
    ratios = w / np.clip(h, 1e-6, None)

    quality_mask = (
        (areas  >= MIN_AREA)  &
        (ratios >= MIN_RATIO) &
        (ratios <= MAX_RATIO)
    )
    if not quality_mask.any():
        return None, None

    return confs[quality_mask], xyxys[quality_mask]


class PatternCentering:

    def __init__(
        self,
        frame_width=640,
        frame_height=480,
        center_box_width=120,
        center_box_height=120,
        threshold_x=15,
        threshold_y=15,
        infer_every_n=INFER_EVERY_N,
        enable_profiling=True,
        model_path="best11m5.onnx",
        mode="red",
    ):
        self.model = YOLO(model_path)

        # warmup
        dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, imgsz=INPUT_SIZE)
        print("[INFO] Model warmed up")

        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.center_x     = frame_width  // 2
        self.center_y     = frame_height // 2

        self.center_box_width  = center_box_width
        self.center_box_height = center_box_height
        self.threshold_x       = threshold_x
        self.threshold_y       = threshold_y

        self.device = self._get_device()

        self.target_labels = MODE_LABELS[mode.lower()]
        self.mode_label    = "B" if mode.lower() == "blue" else "R"

        self.infer_every_n    = infer_every_n
        self.enable_profiling = enable_profiling

        self.shared   = SharedState()
        self.profiler = StageProfiler() if enable_profiling else None

        # smoothing
        self.render_smooth_cx = None
        self.render_smooth_cy = None

        # extrapolation
        self._prev_render_cx = None
        self._prev_render_cy = None
        self._render_vel_x   = 0
        self._render_vel_y   = 0

        self._camera_thread = None
        self._infer_thread  = None

    # ──────────────────────────────────────────────────────────────────
    # DEVICE
    # ──────────────────────────────────────────────────────────────────

    def _get_device(self):
        if torch.cuda.is_available():
            print(f"[INFO] GPU Detected: {torch.cuda.get_device_name(0)}")
            return "cuda"
        print("[INFO] CPU Mode")
        return "cpu"

    # ──────────────────────────────────────────────────────────────────
    # CORE LOGIC
    # ──────────────────────────────────────────────────────────────────

    def detect(self, frame):
        """Kirim frame asli langsung ke model, tanpa resize manual."""
        results = self.model.predict(
            source=frame,
            verbose=False,
            imgsz=INPUT_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            max_det=MAX_DET
        )

        if not results or len(results[0].boxes) == 0:
            return None, None

        boxes  = results[0].boxes
        confs  = boxes.conf.cpu().numpy()
        xyxys  = boxes.xyxy.cpu().numpy()
        clsids = boxes.cls.cpu().numpy().astype(int)

        confs, xyxys = filter_boxes(confs, xyxys, clsids, self.target_labels)
        if confs is None:
            return None, None

        idx  = best_confidence_index(confs)
        conf = float(confs[idx])
        x1, y1, x2, y2 = xyxys[idx].astype(int)

        return (x1, y1, x2, y2), conf

    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def evaluate(self, bbox):
        cx, cy = self.get_center(bbox)

        dx = cx - self.center_x
        dy = cy - self.center_y

        x = -1 if dx < -self.threshold_x else (1 if dx > self.threshold_x else 0)
        y = -1 if dy < -self.threshold_y else (1 if dy > self.threshold_y else 0)

        left   = self.center_x - self.center_box_width  // 2
        right  = self.center_x + self.center_box_width  // 2
        top    = self.center_y - self.center_box_height // 2
        bottom = self.center_y + self.center_box_height // 2

        s = (left <= cx <= right) and (top <= cy <= bottom)

        return {"x": x, "y": y, "s": s, "center_x": cx, "center_y": cy}

    def draw(self, frame: np.ndarray, bbox: tuple, result: dict) -> np.ndarray:
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx = result["center_x"]
        cy = result["center_y"]

        if self.render_smooth_cx is None:
            self.render_smooth_cx = cx
            self.render_smooth_cy = cy
        else:
            self.render_smooth_cx = (
                0.8 * self.render_smooth_cx +
                0.2 * cx
            )

            self.render_smooth_cy = (
                0.8 * self.render_smooth_cy +
                0.2 * cy
            )

        draw_cx = int(self.render_smooth_cx)
        draw_cy = int(self.render_smooth_cy)

        conf = result.get("conf", 0.0)
        cv2.putText(
            frame, f"{self.mode_label} {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 2, cv2.LINE_AA
        )

        cv2.circle(frame, (draw_cx, draw_cy), 5, (0, 255, 255), -1)

        cv2.line(
            frame,
            (self.center_x, self.center_y),
            (draw_cx, draw_cy),
            (255, 255, 0), 2
        )

        cv2.line(frame, (self.center_x - 20, self.center_y), (self.center_x + 20, self.center_y), (255, 0, 0), 2)
        cv2.line(frame, (self.center_x, self.center_y - 20), (self.center_x, self.center_y + 20), (255, 0, 0), 2)

        left   = self.center_x - self.center_box_width  // 2
        right  = self.center_x + self.center_box_width  // 2
        top    = self.center_y - self.center_box_height // 2
        bottom = self.center_y + self.center_box_height // 2

        color = (0, 255, 0) if result["s"] else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.putText(
            frame,
            f"X:{result['x']}  Y:{result['y']}  S:{result['s']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 0), 2
        )

        return frame

    # ──────────────────────────────────────────────────────────────────
    # CAMERA THREAD
    # ──────────────────────────────────────────────────────────────────

    def camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   CAM_BUFFER_SIZE)

        print("[CameraThread] started")

        while self.shared.is_running():
            t0 = self.profiler.tick() if self.profiler else time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("[CameraThread] read failed, stopping.")
                self.shared.request_stop()
                break

            self.shared.set_frame(frame)

            if self.profiler:
                self.profiler.record("Capture", t0)

        cap.release()
        print("[CameraThread] stopped")

    def inference(self):
        last_frame_id = -1
        cached_bbox = None
        cached_result = None


        print("[InferenceThread] started")

        while self.shared.is_running():

            with self.shared._lock:
                frame_id = self.shared._latest_frame_id
                frame = self.shared._latest_frame

            if frame is None or frame_id == last_frame_id:
                time.sleep(0.001)
                continue


            if frame_id % self.infer_every_n != 0:
                self.shared.set_result(cached_bbox, cached_result)
                last_frame_id = frame_id
                continue

            t_inf = self.profiler.tick() if self.profiler else time.perf_counter()

            results = self.model.predict(
                source=frame,
                verbose=False,
                imgsz=INPUT_SIZE,          
                conf=CONFIDENCE_THRESHOLD,
                max_det=MAX_DET
            )

            if self.profiler:
                self.profiler.record("Inference", t_inf)

            last_frame_id = frame_id

            if not results or len(results[0].boxes) == 0:
                cached_bbox = None
                cached_result = None


                self.shared.set_result(None, None)
                continue

            # ==========================
            # POSTPROCESS
            # ==========================
            t_post = self.profiler.tick() if self.profiler else time.perf_counter()

            boxes = results[0].boxes

            confs = boxes.conf.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()
            clsids = boxes.cls.cpu().numpy().astype(int)

            confs, xyxys = filter_boxes(
                confs,
                xyxys,
                clsids,
                self.target_labels
            )

            if confs is None:
                cached_bbox = None
                cached_result = None
                self.shared.set_result(None, None)
                continue

            idx = np.argmax(confs)

            conf = float(confs[idx])

            x1, y1, x2, y2 = xyxys[idx].astype(int)

            bbox = (x1, y1, x2, y2)

            result = self.evaluate(bbox)
            result["conf"] = conf

            print(
                f"X : {result['x']} "
                f"Y : {result['y']} "
            )

            if self.profiler:
                self.profiler.record(
                    "Post-Processing",
                    t_post
                )

            cached_bbox = bbox
            cached_result = result

            self.shared.set_result(
                cached_bbox,
                cached_result
            )

        print("[InferenceThread] stopped")

    # ──────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────────────────

    def _start_threads(self):
        self._camera_thread = threading.Thread(target=self.camera,    name="CameraThread",    daemon=True)
        self._infer_thread  = threading.Thread(target=self.inference,  name="InferenceThread", daemon=True)
        self._camera_thread.start()
        self._infer_thread.start()

    def _stop_threads(self):
        self.shared.request_stop()
        if self._camera_thread:
            self._camera_thread.join(timeout=2.0)
        if self._infer_thread:
            self._infer_thread.join(timeout=2.0)

    # ──────────────────────────────────────────────────────────────────
    # RENDER THREAD
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        self._start_threads()
        print("[RenderThread] started. Press ESC to quit.")

        while self.shared.is_running():
            t_total = self.profiler.tick() if self.profiler else time.perf_counter()

            frame, _     = self.shared.get_frame()
            bbox, result = self.shared.get_result()

            if frame is None:
                time.sleep(0.001)
                continue

            display = frame.copy()

            t_render = self.profiler.tick() if self.profiler else time.perf_counter()
            if bbox is not None and result is not None:
                display = self.draw(display, bbox, result)

            if self.profiler:
                self.profiler.record("render", t_render)
                self.profiler.record("total",  t_total)
                display = self.profiler.draw_overlay(display)

            cv2.imshow("Pattern Centering", display)

            if cv2.waitKey(1) == 27:
                break

        self._stop_threads()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = PatternCentering()
    app.run()