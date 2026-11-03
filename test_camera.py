import cv2
import time
import os
import sys
import threading
from ultralytics import YOLO

# Category information matching Adiwiyata waste segregation rules
WASTE_CATEGORIES = {
    'plastic':       {'color': (14, 116, 144), 'bin': 'Yellow Bin (Plastik)',        'nature': 'Non-Biodegradable'},
    'metal':         {'color': (180, 83, 9),   'bin': 'Blue Bin (Logam/Kaleng)',      'nature': 'Non-Biodegradable'},
    'glass':         {'color': (21, 128, 61),  'bin': 'Green Bin (Kaca)',             'nature': 'Non-Biodegradable'},
    'e_waste':       {'color': (109, 40, 217), 'bin': 'E-Waste Bin (Elektronik)',    'nature': 'Non-Biodegradable'},
    'medical_waste': {'color': (28, 28, 185),  'bin': 'Red Bin (Limbah B3/Medis)',   'nature': 'Hazardous'},
    'paper':         {'color': (216, 78, 29),  'bin': 'Blue Bin (Kertas)',            'nature': 'Biodegradable'},
    'cardboard':     {'color': (146, 64, 14),  'bin': 'Blue Bin (Kardus)',            'nature': 'Biodegradable'},
    'organic_waste': {'color': (4, 120, 87),   'bin': 'Green Bin (Organik)',          'nature': 'Biodegradable'},
}
DEFAULT_CAT = {'color': (100, 100, 100), 'bin': 'General Bin', 'nature': 'Unknown'}

def normalize_class(name: str) -> str:
    n = name.strip().lower().replace(" ", "_")
    aliases = {
        "ewaste": "e_waste", 
        "organic": "organic_waste",
        "medical": "medical_waste", 
        "biodegradable": "organic_waste"
    }
    return aliases.get(n, n)

class AsyncWasteDetector:
    """
    Menjalankan inferensi YOLO di background thread agar video kamera
    tetap render mulus di 30 FPS tanpa freezing/stuttering pada CPU/iGPU.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.25, imgsz: int = 480):
        print(f"Memuat model YOLOv8m dari {model_path}...")
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.imgsz = imgsz
        
        self.latest_frame = None
        self.latest_detections = []
        self.inference_fps = 0.0
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def get_detections(self):
        with self.lock:
            return list(self.latest_detections), self.inference_fps

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _worker_loop(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None  # Consume frame

            if frame_to_process is None:
                time.sleep(0.01)
                continue

            t_start = time.time()
            # Inferensi YOLO dengan ukuran gambar yang dioptimalkan untuk CPU/iGPU
            results = self.model(frame_to_process, conf=self.conf, imgsz=self.imgsz, verbose=False)
            t_infer = time.time() - t_start
            infer_fps = 1.0 / t_infer if t_infer > 0 else 0

            dets = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        raw_cls = r.names.get(int(box.cls[0]), "unknown")
                        cls_name = normalize_class(raw_cls)
                        info = WASTE_CATEGORIES.get(cls_name, DEFAULT_CAT)
                        xyxy = [int(v) for v in box.xyxy[0].tolist()]
                        dets.append({
                            'class': cls_name,
                            'conf': conf,
                            'info': info,
                            'xyxy': xyxy
                        })

            with self.lock:
                self.latest_detections = dets
                self.inference_fps = infer_fps

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan di: {model_path}")
        sys.exit(1)

    # Inisialisasi detector background thread (imgsz=480 jauh lebih enteng dari 640 di CPU)
    detector = AsyncWasteDetector(model_path=model_path, conf_threshold=0.25, imgsz=480)

    camera_index = 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka webcam (index {camera_index}).")
        detector.stop()
        sys.exit(1)

    # Resolusi standar 640x480 agar ringan di CPU/iGPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "="*60)
    print(" LIVE WASTE DETECTION TEST (OPTIMIZED CPU/iGPU)")
    print(" Display: 30 FPS (Asynchronous Inference)")
    print(" Tekan 'q' di jendela kamera untuk keluar.")
    print(" Tekan 's' untuk menyimpan screenshot deteksi.")
    print("="*60 + "\n")

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)

            # Kirim frame terbaru ke background thread AI
            detector.update_frame(frame)

            # Ambil deteksi terakhir yang sudah selesai diproses AI
            detections, ai_fps = detector.get_detections()

            # Hitung Camera Display FPS
            curr_time = time.time()
            display_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            h, w, _ = frame.shape

            # Header overlay info
            cv2.rectangle(frame, (0, 0), (w, 38), (25, 25, 25), -1)
            cv2.putText(frame, f"Cam: {display_fps:.0f} FPS | AI: {ai_fps:.1f} FPS (CPU/iGPU Mode)", 
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

            # Render bounding boxes dari deteksi terakhir
            for d in detections:
                color = d['info']['color']
                x1, y1, x2, y2 = d['xyxy']
                cls_name = d['class']
                conf = d['conf']
                bin_name = d['info']['bin']

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label background
                label_txt = f"{cls_name.upper()} {conf*100:.0f}% -> {bin_name}"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 8, y1), color, -1)
                cv2.putText(frame, label_txt, (x1 + 4, max(14, y1 - 6)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Footer hint
            cv2.rectangle(frame, (0, h - 25), (w, h), (20, 20, 20), -1)
            cv2.putText(frame, "Tekan 'q': Keluar | 's': Simpan Foto", 
                        (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

            cv2.imshow("Waste Classification - Live Camera (Optimized)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[INFO] Tangkapan layar disimpan ke: {filename}")

    finally:
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
