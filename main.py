from flask import Flask, render_template, Response, redirect, url_for, request, jsonify
import cv2
import time
import threading
import numpy as np
import re
import json
import os
import logging
from datetime import datetime

class LoggerSetup:
    def __init__(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_filename = datetime.now().strftime('logs/events_%Y-%m-%d_%H-%M-%S.log')
        self.logger = logging.getLogger('event_logger')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    def log_event(self, event_type, message):
        self.logger.info(f'[{event_type}] {message}')

class ConfigManager:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(script_dir, 'config.json')
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

    def save_config(self, data):
        self.config.update(data)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

class CameraManager:
    def __init__(self):
        self.camera = cv2.VideoCapture(1)
        self.camera.set(cv2.CAP_PROP_FPS, 60)

    def get_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            time.sleep(0.1)
            return None
        return frame

class RaceManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.lock = threading.Lock()
        self.rundenzeiten = {}
        self.letzte_erfassung = {}
        self.beste_rundenzeiten = {}
        self.beste_geschwindigkeiten = {}
        self.delay_zeit = config.get('delay_zeit', 10)
        self.strecken_laenge = config.get('strecken_laenge', 120)
        self.autos_in_boundary = {}

    def rundenzeit_erfassen(self, auto_id):
        aktuelle_zeit = time.time()
        with self.lock:
            if auto_id in self.letzte_erfassung and (aktuelle_zeit - self.letzte_erfassung[auto_id] < self.delay_zeit):
                return None
            if auto_id in self.letzte_erfassung:
                rundenzeit = aktuelle_zeit - self.letzte_erfassung[auto_id]
                if auto_id not in self.rundenzeiten:
                    self.rundenzeiten[auto_id] = []
                self.rundenzeiten[auto_id].append(rundenzeit)
                self.logger.log_event('Rundenzeit', f'{auto_id} hat eine Rundenzeit von {rundenzeit:.2f} Sekunden.')
                geschwindigkeit = (self.strecken_laenge * 3600) / (rundenzeit * 1000)
                if auto_id not in self.beste_rundenzeiten or rundenzeit < self.beste_rundenzeiten[auto_id]:
                    self.beste_rundenzeiten[auto_id] = rundenzeit
                    self.beste_geschwindigkeiten[auto_id] = geschwindigkeit
                    self.logger.log_event('Beste Rundenzeit', f'{auto_id} hat eine neue beste Rundenzeit von {rundenzeit:.2f} Sekunden und eine Geschwindigkeit von {geschwindigkeit:.2f} km/h.')
            else:
                rundenzeit = None
            self.letzte_erfassung[auto_id] = aktuelle_zeit
        return self.rundenzeiten.get(auto_id, None)

class BaseVideoProcessor:
    def __init__(self, config, race_manager):
        self.config = config
        self.race_manager = race_manager
        self.lock = threading.Lock()
        self.analysed_frame = None

    def track_cars(self, camera_manager):
        raise NotImplementedError("Subclasses should implement this method")

    def gen_frames(self):
        while True:
            with self.lock:
                if self.analysed_frame is not None:
                    frame = self.analysed_frame.copy()
                else:
                    continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)

    def gen_web_frames(self):
        while True:
            with self.lock:
                if self.analysed_frame is not None:
                    frame = self.analysed_frame.copy()
                else:
                    continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.17)  # Reduce frame rate to approximately 10 FPS

class ColorBasedVideoProcessor(BaseVideoProcessor):
    def adjust_image(self, image):
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        alpha = self.config['alpha']
        beta = self.config['beta']
        saturation = self.config['saturation']
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], saturation - 50)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return result

    def hex_to_hsv(self, hex_color):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        return hsv

    def rgb_to_hex(self, rgb_color):
        match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', rgb_color)
        if match:
            r, g, b = map(int, match.groups())
            return f'#{r:02x}{g:02x}{b:02x}'
        return None

    def track_cars(self, camera_manager):
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                continue
            height, width, _ = frame.shape
            boundary_box = (0, 0, width, height)
            frame = self.adjust_image(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            x, y, w, h = boundary_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 5)
            detected_cars = []
            for car in self.config['carColors']:
                auto_id = car['name']
                hsv_color = self.hex_to_hsv(car['color'])
                lower_color = np.array([max(int(hsv_color[0]) - 10, 0), 100, 100], dtype=np.uint8)
                upper_color = np.array([min(int(hsv_color[0]) + 10, 255), 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_color, upper_color)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:
                        cx, cy, cw, ch = cv2.boundingRect(contour)
                        if any((cx, cy, cw, ch) in detected_cars for detected_cars in detected_cars):
                            continue
                        car_color = tuple(int(car['color'][i:i+2], 16) for i in (1, 3, 5))
                        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), car_color, 2)
                        cv2.putText(frame, auto_id, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, car_color, 2)
                        if (x < cx < x + w and y < cy < y + h):
                            rundenzeit = self.race_manager.rundenzeit_erfassen(auto_id)
                            if rundenzeit is not None:
                                letzte_rundenzeit = rundenzeit[-1]
                                print(f"{auto_id} hat eine Rundenzeit von {letzte_rundenzeit:.2f} Sekunden.")
                        detected_cars.append((cx, cy, cw, ch))
            with self.lock:
                self.analysed_frame = frame.copy()
            time.sleep(0.01)

from ultralytics import YOLO

class YoloBasedVideoProcessor(BaseVideoProcessor):
    def __init__(self, config, race_manager):
        super().__init__(config, race_manager)
        self.model = self.load_yolo_model()

    def load_yolo_model(self):
        return YOLO("yolov8n.pt")  # Oder "yolov8s.pt" für ein größeres Modell

    def track_cars(self, camera_manager):
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                continue

            results = self.model(frame)[0]  # YOLO-Vorhersagen abrufen

            for det in results.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])  # Bounding-Box-Koordinaten
                conf = det.conf[0].item()
                label = results.names[int(det.cls[0])]

                if conf > 0.5 and label in ["car", "truck", "bus", "motorcycle"]:  # Konfidenzschwelle setzen und nur Fahrzeuge tracken
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Erfassen der Rundenzeit
                    if label not in self.race_manager.rundenzeiten:
                        self.race_manager.rundenzeiten[label] = []
                    rundenzeit = self.race_manager.rundenzeit_erfassen(label)
                    if rundenzeit and len(rundenzeit) > 0:
                        letzte_rundenzeit = rundenzeit[-1]
                        print(f"{label} hat eine Rundenzeit von {letzte_rundenzeit:.2f} Sekunden.")

            with self.lock:
                self.analysed_frame = frame.copy()
            
            time.sleep(0.01)

class App:
    def __init__(self, tracking_type='color'):
        self.app = Flask(__name__)
        self.logger = LoggerSetup()
        self.config_manager = ConfigManager()
        self.camera_manager = CameraManager()
        self.race_manager = RaceManager(self.config_manager.config, self.logger)
        tracking_type = self.config_manager.config.get('trackingType', 'color')
        if tracking_type == 'color':
            self.video_processor = ColorBasedVideoProcessor(self.config_manager.config, self.race_manager)
        elif tracking_type == 'yolo':
            self.video_processor = YoloBasedVideoProcessor(self.config_manager.config, self.race_manager)
        self.setup_routes()

    def setup_routes(self):
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/help', 'help', self.help_page)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/rundenzeiten', 'get_rundenzeiten', self.get_rundenzeiten)
        self.app.add_url_rule('/config', 'config_page', self.config_page)
        self.app.add_url_rule('/get_config', 'get_config', self.get_config)
        self.app.add_url_rule('/save_config', 'save_config', self.save_config, methods=['POST'])
        self.app.add_url_rule('/add_car', 'add_car', self.add_car, methods=['POST'])
        self.app.add_url_rule('/leaderboard', 'leaderboard', self.leaderboard)
        self.app.add_url_rule('/car/<auto_id>', 'car_details', self.car_details)
        self.app.add_url_rule('/reset_data', 'reset_data', self.reset_data, methods=['POST'])
        self.app.add_url_rule('/get_best_lap_times', 'get_best_lap_times', self.get_best_lap_times)

    def index(self):
        return render_template('index.html', config=self.config_manager.config, autos=list(self.race_manager.rundenzeiten.keys()))

    def help_page(self):
        return render_template('help.html')

    def video_feed(self):
        return Response(self.video_processor.gen_web_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def get_rundenzeiten(self):
        zeiten = {}
        aktuelle_zeit = time.time()
        inactive_limit = self.config_manager.config['inactiveLimit']
        with self.race_manager.lock:
            for auto_id, runden in self.race_manager.rundenzeiten.items():
                letzte_rundenzeit = runden[-1] if runden else None
                letzte_erfassungszeit = self.race_manager.letzte_erfassung.get(auto_id, aktuelle_zeit)
                inaktiv_zeit = aktuelle_zeit - letzte_erfassungszeit
                zeiten[auto_id] = {
                    'letzte_runde': letzte_rundenzeit,
                    'inaktiv_zeit': inaktiv_zeit,
                    'status': 'Inaktiv' if inaktiv_zeit > inactive_limit else 'Aktiv'
                }
        return jsonify(zeiten)

    def config_page(self):
        return render_template('config.html')

    def get_config(self):
        return jsonify(self.config_manager.config)

    def save_config(self):
        data = request.get_json()
        self.config_manager.save_config({
            'inactiveLimit': int(data.get('inactiveLimit', self.config_manager.config.get('inactiveLimit', 10))),
            'alpha': float(data.get('alpha', self.config_manager.config.get('alpha', 1.3))),
            'beta': int(data.get('beta', self.config_manager.config.get('beta', 40))),
            'saturation': int(data.get('saturation', self.config_manager.config.get('saturation', 50))),
            'delay_zeit': int(data.get('delayTime', self.config_manager.config.get('delay_zeit', 10))),
            'strecken_laenge': int(data.get('trackLength', self.config_manager.config.get('strecken_laenge', 120))),
            'trackingType': data.get('trackingType', self.config_manager.config.get('trackingType', 'color')),
            'carColors': data.get('carColors', self.config_manager.config.get('carColors', []))
        })
        return jsonify(success=True)

    def add_car(self):
        data = request.get_json()
        color = data.get('color')
        if color and color.startswith('rgb'):
            color = self.video_processor.rgb_to_hex(color)
        if color and color.startswith('#') and len(color) == 7:
            new_car = {
                'name': f'auto_{len(self.config_manager.config["carColors"]) + 1}',
                'color': color
            }
            self.config_manager.config['carColors'].append(new_car)
            self.config_manager.save_config(self.config_manager.config)
            return jsonify(success=True, color=color)
        return jsonify(success=False), 400

    def leaderboard(self):
        return render_template('leaderboard.html')

    def get_best_lap_times(self):
        with self.race_manager.lock:
            lap_times = [{'name': auto_id, 'bestLapTime': best_time, 'bestSpeed': self.race_manager.beste_geschwindigkeiten[auto_id]} for auto_id, best_time in self.race_manager.beste_rundenzeiten.items()]
        return jsonify(lap_times)

    def car_details(self, auto_id):
        with self.race_manager.lock:
            runden = self.race_manager.rundenzeiten.get(auto_id, [])
            if not runden:
                return render_template('car_details.html', auto_id=auto_id, runden=[], avg_rundenzeit=None, avg_geschwindigkeit=None, gefahrene_strecke=None, rundenanzahl=0)
            total_rundenzeit = sum(runden)
            avg_rundenzeit = total_rundenzeit / len(runden)
            avg_geschwindigkeit = (self.race_manager.strecken_laenge * 3600) / (avg_rundenzeit * 1000)
            avg_rundenzeit = f"{avg_rundenzeit:.3f}"
            avg_geschwindigkeit = f"{avg_geschwindigkeit:.3f}"
            gefahrene_strecke = (len(runden) * self.race_manager.strecken_laenge) / 1000
            rundenanzahl = len(runden)
            runden = [f"{runde:.3f}" for runde in runden]
        return render_template('car_details.html', auto_id=auto_id, runden=runden, avg_rundenzeit=avg_rundenzeit, avg_geschwindigkeit=avg_geschwindigkeit, gefahrene_strecke=gefahrene_strecke, rundenanzahl=rundenanzahl)

    def reset_data(self):
        with self.race_manager.lock:
            self.race_manager.rundenzeiten = {}
            self.race_manager.letzte_erfassung = {}
            self.race_manager.beste_rundenzeiten = {}
            self.race_manager.beste_geschwindigkeiten = {}
            self.race_manager.autos_in_boundary = {}
        return redirect(url_for('index'))

    def run(self):
        tracking_thread = threading.Thread(target=self.video_processor.track_cars, args=(self.camera_manager,))
        tracking_thread.daemon = True
        tracking_thread.start()
        self.app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    app = App(tracking_type='yolo')  # or 'yolo' for YOLOv8 tracking
    app.run() 