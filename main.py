from flask import Flask, render_template, Response, redirect, url_for, request, jsonify
import cv2
import time
import threading
import numpy as np

app = Flask(__name__)

# Variablen für Steuerung
streaming = False
lock = threading.Lock()

# Kamera einrichten
camera = cv2.VideoCapture(1)  # Stelle sicher, dass die richtige Kamera verwendet wird

# Auto-Farben (HSV und BGR)
farbdefinitionen = {
    'blaues_auto': ([90, 100, 50], [130, 255, 255], (255, 0, 0)),
    'rotes_auto': ([0, 100, 50], [10, 255, 255], (0, 0, 255)),
    'grünes_auto': ([40, 100, 50], [80, 255, 255], (0, 255, 0))
}

# Rundenzeiten und Delays
rundenzeiten = {}
letzte_erfassung = {}
delay_zeit = 3

# Boundary Box für die Erkennung
boundary_box = (500, 0, 150, 1080)  # (x, y, width, height)

# Überwachung, ob das Auto bereits in der Boundary Box ist
autos_in_boundary = {}

# Konfigurationsvariablen
config = {
    'inactiveLimit': 10,  # Standardwert für Inaktivitätsgrenze
    'alpha': 1.3,         # Standardwert für Kontrast
    'beta': 40,           # Standardwert für Helligkeit
    'saturation': 50      # Standardwert für Sättigung
}

# Funktion zur Erfassung der Rundenzeit
def rundenzeit_erfassen(auto_id):
    aktuelle_zeit = time.time()
    with lock:
        if auto_id in letzte_erfassung and (aktuelle_zeit - letzte_erfassung[auto_id] < delay_zeit):
            return None
        rundenzeit = aktuelle_zeit - letzte_erfassung.get(auto_id, aktuelle_zeit)  # Nutze die letzte Erfassungszeit
        rundenzeiten[auto_id] = rundenzeit  # speichere die letzte Rundenzeit
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

# Funktion für den Videostream mit Rundenzeiterfassung
def gen_frames():
    global streaming
    while True:
        with lock:
            if not streaming:
                break

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Bild anpassen (Farben und Weißabgleich)
        frame = adjust_image(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Zeichne die senkrechte Boundary Box
        x, y, w, h = boundary_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 5)

        for auto_id, (lower, upper, box_color) in farbdefinitionen.items():
            lower_color = tuple(lower)
            upper_color = tuple(upper)
            mask = cv2.inRange(hsv, lower_color, upper_color)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter für Rauschen
                    cx, cy, cw, ch = cv2.boundingRect(contour)

                    # Auto-Box zeichnen
                    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), box_color, 2)
                    cv2.putText(frame, auto_id, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

                    # Überprüfung, ob das Auto in die senkrechte Boundary Box kommt
                    if x < cx < x + w and y < cy < y + h and auto_id not in autos_in_boundary:
                        # Auto ist in der Boundary Box
                        rundenzeit = rundenzeit_erfassen(auto_id)
                        if rundenzeit is not None:
                            print(f"{auto_id} hat eine Rundenzeit von {rundenzeit:.2f} Sekunden.")
                            autos_in_boundary[auto_id] = True  # Markiere das Auto als in der Boundary Box
                    elif not (x < cx < x + w and y < cy < y + h) and auto_id in autos_in_boundary:
                        # Auto ist nicht mehr in der Boundary Box, entferne es aus der Überwachung
                        del autos_in_boundary[auto_id]

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Kurze Pause, um CPU-Last zu reduzieren
        time.sleep(0.01)

# Routen für Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_stream():
    global streaming, camera
    with lock:
        streaming = True
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)  # Stelle sicher, dass die richtige Kamera verwendet wird
    return redirect(url_for('index'))

@app.route('/stop')
def stop_stream():
    global streaming, camera
    with lock:
        streaming = False
        camera.release()
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rundenzeiten')
def get_rundenzeiten():
    zeiten = {}
    aktuelle_zeit = time.time()
    with lock:
        for auto_id, letzte_rundenzeit in rundenzeiten.items():
            letzte_erfassungszeit = letzte_erfassung.get(auto_id, aktuelle_zeit)
            inaktiv_zeit = aktuelle_zeit - letzte_erfassungszeit
            zeiten[auto_id] = {
                'letzte_runde': letzte_rundenzeit,
                'inaktiv_zeit': inaktiv_zeit
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
    config['inactiveLimit'] = int(data.get('inactiveLimit', 10))
    config['alpha'] = float(data.get('alpha', 1.3))
    config['beta'] = int(data.get('beta', 40))
    config['saturation'] = int(data.get('saturation', 50))
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)