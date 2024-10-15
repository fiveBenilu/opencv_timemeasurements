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

app = Flask(__name__)

# Logging setup
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a new log file for each session with a timestamp
log_filename = datetime.now().strftime('logs/events_%Y-%m-%d_%H-%M-%S.log')
logger = logging.getLogger('event_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filename)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Disable Flask's default logging to avoid logging web events
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def log_event(event_type, message):
    logger.info(f'[{event_type}] {message}')

# Variablen für Steuerung
lock = threading.Lock()

# Kamera einrichten
camera = cv2.VideoCapture(0)  # Stelle sicher, dass die richtige Kamera verwendet wird
camera.set(cv2.CAP_PROP_FPS, 60)

# Konfigurationsvariablen
config = {}

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the config.json file
config_path = os.path.join(script_dir, 'config.json')

# Open the config.json file
with open(config_path, 'r') as f:
    config = json.load(f)

# Rundenzeiten und Delays
rundenzeiten = {}
letzte_erfassung = {}
beste_rundenzeiten = {}
beste_geschwindigkeiten = {}
delay_zeit = config.get('delay_zeit', 10)  # Verzögerungszeit in Sekunden
strecken_laenge = config.get('strecken_laenge', 120)  # Länge der Strecke in Metern

# Boundary Box für die Erkennung (initial auf 0 gesetzt)
boundary_box = (0, 0, 0, 0)

# Überwachung, ob das Auto bereits in der Boundary Box ist
autos_in_boundary = {}

# Globale Variable für das analysierte Bild
analysed_frame = None

def rundenzeit_erfassen(auto_id):
    aktuelle_zeit = time.time()
    with lock:
        if auto_id in letzte_erfassung and (aktuelle_zeit - letzte_erfassung[auto_id] < delay_zeit):
            return None
        
        # Berechne die Rundenzeit nur, wenn das Auto bereits erfasst wurde
        if auto_id in letzte_erfassung:
            rundenzeit = aktuelle_zeit - letzte_erfassung[auto_id]
            if auto_id not in rundenzeiten:
                rundenzeiten[auto_id] = []
            rundenzeiten[auto_id].append(rundenzeit)  # speichere die letzte Rundenzeit
            log_event('Rundenzeit', f'{auto_id} hat eine Rundenzeit von {rundenzeit:.2f} Sekunden.')

            # Berechne die Durchschnittsgeschwindigkeit
            geschwindigkeit = (strecken_laenge * 3600) / (rundenzeit * 1000)  # km/h

            # Aktualisiere die beste Rundenzeit und Geschwindigkeit
            if auto_id not in beste_rundenzeiten or rundenzeit < beste_rundenzeiten[auto_id]:
                beste_rundenzeiten[auto_id] = rundenzeit
                beste_geschwindigkeiten[auto_id] = geschwindigkeit
                log_event('Beste Rundenzeit', f'{auto_id} hat eine neue beste Rundenzeit von {rundenzeit:.2f} Sekunden und eine Geschwindigkeit von {geschwindigkeit:.2f} km/h.')
        else:
            rundenzeit = None  # Keine Rundenzeit beim ersten Erkennen

        letzte_erfassung[auto_id] = aktuelle_zeit  # aktualisiere die Erkennungszeit

    return rundenzeiten.get(auto_id, None)

# Funktion zur Anpassung der Farben und des Weißabgleichs
def adjust_image(image):
    # Weißabgleich
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    # Farbkorrektur (Helligkeit, Kontrast und Sättigung)
    alpha = config['alpha']  # Kontraststeuerung (1.0-3.0)
    beta = config['beta']    # Helligkeitssteuerung (0-100)
    saturation = config['saturation']  # Sättigungssteuerung (0-100)

    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], saturation - 50)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result

# Funktion zur Umwandlung von Hex-Farbe zu HSV
def hex_to_hsv(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    return hsv

# Funktion zur Umwandlung von RGB zu Hex
def rgb_to_hex(rgb_color):
    match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', rgb_color)
    if match:
        r, g, b = map(int, match.groups())
        return f'#{r:02x}{g:02x}{b:02x}'
    return None

def track_cars():
    global boundary_box, analysed_frame
    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Aktualisiere die Boundary Box basierend auf der Frame-Größe
        height, width, _ = frame.shape
        boundary_box = (0, 0, width, height)  # Setze die Boundary Box auf die Größe des Frames

        # Bild anpassen (Farben und Weißabgleich)
        frame = adjust_image(frame)

        # Konvertiere das Bild in den HSV-Farbraum
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Zeichne die senkrechte Boundary Box
        x, y, w, h = boundary_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 5)

        detected_cars = []

        for car in config['carColors']:
            auto_id = car['name']
            hsv_color = hex_to_hsv(car['color'])
            lower_color = np.array([max(int(hsv_color[0]) - 10, 0), 100, 100], dtype=np.uint8)
            upper_color = np.array([min(int(hsv_color[0]) + 10, 255), 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_color, upper_color)

            # Verwende Dilation, um benachbarte kleine Bereiche zu verbinden
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Finde Konturen im maskierten Bild
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter für Rauschen

                    # Finde die Bounding Box der zusammengefassten Kontur
                    cx, cy, cw, ch = cv2.boundingRect(contour)

                    # Überprüfen, ob die erkannte Kontur bereits einem anderen Auto zugeordnet wurde
                    if any((cx, cy, cw, ch) in detected_cars for detected_cars in detected_cars):
                        continue

                    # Auto-Box zeichnen
                    car_color = tuple(int(car['color'][i:i+2], 16) for i in (1, 3, 5))
                    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), car_color, 2)
                    cv2.putText(frame, auto_id, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, car_color, 2)

                    # Überprüfung, ob das Auto in die senkrechte Boundary Box kommt
                    if (x < cx < x + w and y < cy < y + h):
                        # Auto ist in der Boundary Box
                        rundenzeit = rundenzeit_erfassen(auto_id)
                        if rundenzeit is not None:
                            letzte_rundenzeit = rundenzeit[-1]  # Nehmen Sie die letzte Rundenzeit aus der Liste
                            print(f"{auto_id} hat eine Rundenzeit von {letzte_rundenzeit:.2f} Sekunden.")
                        autos_in_boundary[auto_id] = True  # Markiere das Auto als in der Boundary Box
                    else:
                        # Auto ist nicht mehr in der Boundary Box, entferne es aus der Überwachung
                        autos_in_boundary[auto_id] = False

                    # Füge die erkannte Kontur zur Liste der erkannten Autos hinzu
                    detected_cars.append((cx, cy, cw, ch))

        # Speichere das analysierte Bild in der globalen Variable
        with lock:
            analysed_frame = frame.copy()

        # Kurze Pause, um CPU-Last zu reduzieren
        time.sleep(0.01)

def gen_frames():
    global analysed_frame
    while True:
        with lock:
            if analysed_frame is not None:
                frame = analysed_frame.copy()
            else:
                continue

        # Bild kodieren und als Byte-Stream zurückgeben
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Kurze Pause, um CPU-Last zu reduzieren
        time.sleep(0.01)

# Routen für Flask
@app.route('/')
def index():
    return render_template('index.html', config=config, autos=list(rundenzeiten.keys()))

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rundenzeiten')
def get_rundenzeiten():
    zeiten = {}
    aktuelle_zeit = time.time()
    inactive_limit = config['inactiveLimit']
    with lock:
        for auto_id, runden in rundenzeiten.items():
            letzte_rundenzeit = runden[-1] if runden else None
            letzte_erfassungszeit = letzte_erfassung.get(auto_id, aktuelle_zeit)
            inaktiv_zeit = aktuelle_zeit - letzte_erfassungszeit
            zeiten[auto_id] = {
                'letzte_runde': letzte_rundenzeit,
                'inaktiv_zeit': inaktiv_zeit,
                'status': 'Inaktiv' if inaktiv_zeit > inactive_limit else 'Aktiv'
            }
    return jsonify(zeiten)

@app.route('/config')
def config_page():
    return render_template('config.html')

@app.route('/get_config')
def get_config():
    return jsonify(config)

@app.route('/save_config', methods=['POST'])
def save_config():
    data = request.get_json()
    config['inactiveLimit'] = int(data.get('inactiveLimit', config.get('inactiveLimit', 10)))
    config['alpha'] = float(data.get('alpha', config.get('alpha', 1.3)))
    config['beta'] = int(data.get('beta', config.get('beta', 40)))
    config['saturation'] = int(data.get('saturation', config.get('saturation', 50)))
    config['delay_zeit'] = int(data.get('delayTime', config.get('delay_zeit', 10)))  # Verzögerungszeit in Sekunden
    config['strecken_laenge'] = int(data.get('trackLength', config.get('strecken_laenge', 120)))
    config['carColors'] = data.get('carColors', config.get('carColors', []))
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return jsonify(success=True)

@app.route('/add_car', methods=['POST'])
def add_car():
    data = request.get_json()
    color = data.get('color')
    if color and color.startswith('rgb'):
        color = rgb_to_hex(color)
    if color and color.startswith('#') and len(color) == 7:
        new_car = {
            'name': f'auto_{len(config["carColors"]) + 1}',
            'color': color
        }
        config['carColors'].append(new_car)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return jsonify(success=True, color=color)
    return jsonify(success=False), 400

# Route für das Leaderboard
@app.route('/leaderboard')
def leaderboard():
    return render_template('leaderboard.html')

# API-Endpunkt zum Abrufen der besten Rundenzeiten und Geschwindigkeiten
@app.route('/get_best_lap_times')
def get_best_lap_times():
    with lock:
        lap_times = [{'name': auto_id, 'bestLapTime': best_time, 'bestSpeed': beste_geschwindigkeiten[auto_id]} for auto_id, best_time in beste_rundenzeiten.items()]
    return jsonify(lap_times)

@app.route('/car/<auto_id>')
def car_details(auto_id):
    with lock:
        runden = rundenzeiten.get(auto_id, [])
        if not runden:
            return render_template('car_details.html', auto_id=auto_id, runden=[], avg_rundenzeit=None, avg_geschwindigkeit=None, gefahrene_strecke=None, rundenanzahl=0)

        # Berechne die durchschnittliche Rundenzeit und Geschwindigkeit
        total_rundenzeit = sum(runden)
        avg_rundenzeit = total_rundenzeit / len(runden)
        avg_geschwindigkeit = (strecken_laenge * 3600) / (avg_rundenzeit * 1000)  # km/h

        # Formatieren der Werte auf maximal 3 Nachkommastellen
        avg_rundenzeit = f"{avg_rundenzeit:.3f}"
        avg_geschwindigkeit = f"{avg_geschwindigkeit:.3f}"
        gefahrene_strecke = (len(runden) * strecken_laenge) / 1000  # in km
        rundenanzahl = len(runden)

        # Formatieren der Rundenzeiten auf maximal 3 Nachkommastellen
        runden = [f"{runde:.3f}" for runde in runden]

    return render_template('car_details.html', auto_id=auto_id, runden=runden, avg_rundenzeit=avg_rundenzeit, avg_geschwindigkeit=avg_geschwindigkeit, gefahrene_strecke=gefahrene_strecke, rundenanzahl=rundenanzahl)

@app.route('/reset_data', methods=['POST'])
def reset_data():
    global rundenzeiten, letzte_erfassung, beste_rundenzeiten, beste_geschwindigkeiten, autos_in_boundary
    with lock:
        rundenzeiten = {}
        letzte_erfassung = {}
        beste_rundenzeiten = {}
        beste_geschwindigkeiten = {}
        autos_in_boundary = {}
    return redirect(url_for('index'))

if __name__ == "__main__":
    # Starte den Tracking-Thread
    tracking_thread = threading.Thread(target=track_cars)
    tracking_thread.daemon = True
    tracking_thread.start()

    # Setze den Stream auf immer aktiv
    streaming = True

    app.run(host='0.0.0.0', port=5000, debug=False)