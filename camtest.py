import cv2

# Liste zur Speicherung der gefundenen Kamerageräte
kameras = []

# Probiere die Kamera-Indices (0-10) aus
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Kamera mit Index {i} gefunden")
        kameras.append(i)
        cap.release()  # Kamera freigeben, nachdem sie gefunden wurde

if not kameras:
    print("Keine Kameras gefunden.")
else:
    print(f"Gefundene Kameras: {kameras}")