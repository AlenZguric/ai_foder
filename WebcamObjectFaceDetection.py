import face_recognition
import cv2
import numpy as np
from datetime import datetime

# Učitajte referentne slike za sve ukućane i enkodirajte ih
known_face_encodings = []
known_face_names = []

podatci= []

# Dodajte slike i imena
images_and_names = [
    ("alen.jpg", "Alen"),
    ("jasmina.jpg", "Jasmina"),
    ("ines.jpg", "Ines"),
    ("paula.jpg", "Paula"),
    ("lucia.jpg", "Lucia"),
     ("tomo.jpg", "Tomo")
]

for image_path, name in images_and_names:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# Pokrenite video stream s web kamere
cap = cv2.VideoCapture(0)

# Postavljanje rezolucije kamere za bolje rezultate na većoj udaljenosti
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Ne mogu pristupiti web kameri!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pretvori frame u RGB format za face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Smanjivanje rezolucije za proširenje područja pretraživanja
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    # Detekcija lica s HOG modelom (CPU) za veće područje
    face_locations = face_recognition.face_locations(small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Povećavanje koordinata nazad na originalnu veličinu
    face_locations = [(top * 2, right * 2, bottom * 2, left * 2) for (top, right, bottom, left) in face_locations]

    # Prođi kroz sva detektirana lica
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Usporedi lice s poznatim licima
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Nepoznato lice"

        # Ako postoji podudaranje, uzmi ime prvog pronađenog lica
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            if name == "Tomo":
                #vrijeme = datetime()
                #print(f"Uočeno lice: {name} - {vrijeme}")
                cv2.putText(frame, "Tomo je u blizzini",(left, top - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
                podatci.append(name)
                print(podatci)
                

        # Nacrtaj okvir oko lica i ispiši ime
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Crveni okvir
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Crveni tekst

    # Prikaz trenutnog frame-a
    cv2.imshow("Face Recognition Test", frame)

    # Pritisnite 'q' za izlaz
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
