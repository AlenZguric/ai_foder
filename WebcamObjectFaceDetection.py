import cv2
import mediapipe as mp
import face_recognition
import numpy as np

# Inicijalizacija Mediapipe modula za detekciju lica
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)

# Učitavanje referentne slike i enkodiranje lica
reference_image = face_recognition.load_image_file("reference_face.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Pokretanje video streama s web kamere
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ne mogu pristupiti web kameri!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ne mogu pročitati frame sa web kamere!")
        break

    # Promjena veličine frame-a na 800x600
    frame = cv2.resize(frame, (800, 600))

    # Pretvorba slike u RGB format za Mediapipe
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesiranje slike za detekciju lica
    results = faceDetection.process(frameRGB)

    if results.detections:
        for detection in results.detections:
            # Dohvat granica detekcije lica
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Izrezivanje lica iz trenutnog okvira
            face_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            try:
                face_encoding = face_recognition.face_encodings(face_img)[0]  # Kodiraj izrezano lice
                matches = face_recognition.compare_faces([reference_encoding], face_encoding)  # Usporedi lice s referentnim
                if matches[0]:  # Ako lice odgovara referentnom
                    # Ispis prepoznavanja lica
                    cv2.putText(frame, "Prepoznato: Alen Zgurić", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, bbox, (0, 255, 0), 2)  # Pravokutnik oko prepoznatog lica
                else:
                    # Oznaka za neprepoznato lice
                    cv2.putText(frame, "Nepoznato lice", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.rectangle(frame, bbox, (255, 0, 0), 2)  # Pravokutnik oko nepoznatog lica
            except:
                continue  # Preskoči ako enkodiranje lica ne uspije

    # Prikaz slike s detekcijama
    cv2.imshow("Webcam Object and Face Detection", frame)

    # Prekini petlju pritiskom na tipku 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Oslobodi resurse
cap.release()
cv2.destroyAllWindows()
