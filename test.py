import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import easyocr
import cv2
import matplotlib.pyplot as plt

# Initialisiere EasyOCR-Reader
reader = easyocr.Reader(['en'])

# Lade ein Testbild
image_path = 'test_image.jpg'  # Pfad zu deinem Testbild
image = cv2.imread(image_path)

# Überprüfe, ob das Bild geladen wurde
if image is None:
    print(f"Fehler: Bild {image_path} konnte nicht geladen werden.")
    exit()

# Erkenne Text im Bild
results = reader.readtext(image)

# Zeige das Bild und die erkannten Texte an
for (bbox, text, prob) in results:
    # Extrahiere die Bounding Box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    # Zeichne die Bounding Box und den erkannten Text auf das Bild
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Zeige das Bild mit den erkannten Texten an
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()