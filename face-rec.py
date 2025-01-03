import cv2
import mediapipe as mp
import face_recognition
import numpy as np
from djitellopy import Tello

# Parametri slike
width, height = 640, 480

# PID parametri
xPID, yPID, zPID = [0.21, 0, 0.1], [0.27, 0, 0.1], [0.0021, 0, 0.1]
xTarget, yTarget, zTarget = width // 2, height // 2, 11500
areaTolerance = 2000  # Tolerancija za udaljenost
pErrorX, pErrorY, pErrorZ = 0, 0, 0

# Mediapipe Face Detection
mpFaces = mp.solutions.face_detection
Faces = mpFaces.FaceDetection(min_detection_confidence=0.5, model_selection=1)

# Tello dron
my_drone = Tello()
my_drone.connect()
print(my_drone.get_battery())
my_drone.streamon()

# Učitajte referentnu sliku
reference_image = face_recognition.load_image_file("reference_face.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# PID kontroler funkcija
def PIDController(PID, target, current, prevError, integral, limit=[-100, 100]):
    error = target - current
    P = PID[0] * error
    integral += PID[1] * error
    derivative = PID[2] * (error - prevError)
    output = P + integral + derivative
    return np.clip(output, limit[0], limit[1]), error, integral

while True:
    img = my_drone.get_frame_read().frame
    img = cv2.resize(img, (width, height))
    xVal, yVal, zVal = 0, 0, 0

    # Pretvori sliku natrag u BGR format prije prikaza
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Prikaz slike
    cv2.imshow("Drone Face Tracking", img)

    # Detekcija lica
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    results = Faces.process(imgRGB)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cx, cy = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
            area = bbox[2] * bbox[3]

            # Prikaz svih lica
            cv2.rectangle(img, bbox, (255, 0, 0), 2)
            cv2.putText(img, "Nepoznato lice", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Izrežite lice iz trenutnog kadra
            face_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            try:
                face_encoding = face_recognition.face_encodings(face_img)[0]
                matches = face_recognition.compare_faces([reference_encoding], face_encoding)
                if matches[0]:  # Ako lice odgovara referentnom
                    # Ispis prepoznavanja
                    cv2.putText(img, "Alen Zgurić", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # PID kontrola za X, Y, i Z osi
                    xVal, pErrorX, _ = PIDController(xPID, xTarget, cx, pErrorX, 0)
                    yVal, pErrorY, _ = PIDController(yPID, yTarget, cy, pErrorY, 0)

                    if abs(area - zTarget) > areaTolerance:
                        zVal, pErrorZ, _ = PIDController(zPID, zTarget, area, pErrorZ, 0)
                    else:
                        zVal = 0  # Unutar tolerancije, ne pomiče se

                    # Prikaz praćenja lica
                    cv2.rectangle(img, bbox, (0, 255, 0), 2)
                    cv2.putText(img, f"Tracking Face", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break  # Prati samo referentno lice
            except:
                continue

    # Slanje naredbi dronu
    my_drone.send_rc_control(int(xVal), int(zVal), int(yVal), 0)

    # Prikaz slike
    cv2.imshow("Drone Face Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        my_drone.land()
        break

cv2.destroyAllWindows()
