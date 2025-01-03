import cv2  # OpenCV za obradu slike
import mediapipe as mp  # Mediapipe za detekciju lica
import face_recognition  # Knjižnica za prepoznavanje lica
import numpy as np  # Za matematičke operacije i obradu podataka
from djitellopy import Tello  # Za komunikaciju s Tello dronom

# Parametri slike (širina i visina prozora za prikaz slike)
width, height = 640, 480

# PID parametri za kontrolu drona na X, Y i Z osima
xPID, yPID, zPID = [0.21, 0, 0.1], [0.27, 0, 0.1], [0.0021, 0, 0.1]
xTarget, yTarget, zTarget = width // 2, height // 2, 11500  # Ciljne vrijednosti za centriranje lica
areaTolerance = 2000  # Tolerancija za udaljenost između drona i objekta
pErrorX, pErrorY, pErrorZ = 0, 0, 0  # Početne vrijednosti pogrešaka za PID

# Mediapipe Face Detection (detekcija lica)
mpFaces = mp.solutions.face_detection  # Inicijalizacija Mediapipe modula za detekciju lica
Faces = mpFaces.FaceDetection(min_detection_confidence=0.5, model_selection=1)  # Model za detekciju lica s povjerenjem 0.5

# Tello dron
my_drone = Tello()  # Inicijalizacija Tello drona
my_drone.connect()  # Povezivanje s dronom
print(my_drone.get_battery())  # Ispis trenutnog stanja baterije drona
my_drone.streamon()  # Uključivanje prijenosa video zapisa s kamere drona

# Učitajte referentnu sliku (slika osobe za prepoznavanje)
reference_image = face_recognition.load_image_file("reference_face.jpg")  # Učitaj sliku iz datoteke
reference_encoding = face_recognition.face_encodings(reference_image)[0]  # Kodiraj lice iz referentne slike

# PID kontroler funkcija za upravljanje dronom
# PID koristi proporcionalni, integralni i derivativni dio za precizno praćenje cilja
def PIDController(PID, target, current, prevError, integral, limit=[-100, 100]):
    error = target - current  # Izračun pogreške
    P = PID[0] * error  # Proporcionalni dio
    integral += PID[1] * error  # Integralni dio
    derivative = PID[2] * (error - prevError)  # Derivativni dio
    output = P + integral + derivative  # Konačni izlaz PID kontrolera
    return np.clip(output, limit[0], limit[1]), error, integral  # Ograniči izlaz na definirane vrijednosti

while True:
    img = my_drone.get_frame_read().frame  # Dohvati trenutni okvir s kamere drona
    img = cv2.resize(img, (width, height))  # Promijeni veličinu slike na definirane dimenzije
    xVal, yVal, zVal = 0, 0, 0  # Početne vrijednosti za kontrolu kretanja drona

    # Pretvori sliku natrag u BGR format prije prikaza (ako je potrebna konverzija boja)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Prikaz slike u prozoru
    cv2.imshow("Drone Face Tracking", img)

    # Detekcija lica na slici (konverzija u RGB format za Mediapipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # Promjena formata slike za Mediapipe
    results = Faces.process(imgRGB)  # Procesiraj sliku za detekciju lica

    if results.detections:  # Ako su pronađena lica
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box  # Dohvati granice lica
            ih, iw, _ = img.shape  # Visina i širina slike
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)  # Prilagodi granice lica stvarnim dimenzijama slike
            cx, cy = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2  # Koordinate centra lica
            area = bbox[2] * bbox[3]  # Površina lica (širina x visina)

            # Prikaz svih lica na slici
            cv2.rectangle(img, bbox, (255, 0, 0), 2)  # Nacrtaj pravokutnik oko detektiranog lica
            cv2.putText(img, "Nepoznato lice", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Označi lice kao nepoznato

            # Izrežite lice iz trenutnog kadra
            face_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]  # Izoliraj lice za prepoznavanje
            try:
                face_encoding = face_recognition.face_encodings(face_img)[0]  # Kodiraj izrezano lice
                matches = face_recognition.compare_faces([reference_encoding], face_encoding)  # Usporedi lice s referentnim
                if matches[0]:  # Ako lice odgovara referentnom
                    # Ispis prepoznavanja lica
                    cv2.putText(img, "Alen Zgurić", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # PID kontrola za X, Y, i Z osi
                    xVal, pErrorX, _ = PIDController(xPID, xTarget, cx, pErrorX, 0)  # Kontrola na X osi
                    yVal, pErrorY, _ = PIDController(yPID, yTarget, cy, pErrorY, 0)  # Kontrola na Y osi

                    if abs(area - zTarget) > areaTolerance:  # Provjeri udaljenost cilja
                        zVal, pErrorZ, _ = PIDController(zPID, zTarget, area, pErrorZ, 0)  # Kontrola na Z osi
                    else:
                        zVal = 0  # Unutar tolerancije, dron se ne pomiče

                    # Prikaz praćenja lica
                    cv2.rectangle(img, bbox, (0, 255, 0), 2)  # Oboji pravokutnik u zeleno za prepoznato lice
                    cv2.putText(img, f"Tracking Face", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Ispis statusa praćenja
                    break  # Prekini petlju nakon što pronađe referentno lice
            except:
                continue  # Preskoči greške u prepoznavanju

    # Slanje naredbi dronu (kontrola kretanja)
    my_drone.send_rc_control(int(xVal), int(zVal), int(yVal), 0)  # Šalji kontrolne vrijednosti za X, Y, i Z osi

    # Prikaz slike u prozoru
    cv2.imshow("Drone Face Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Ako pritisnemo "q", izađi iz petlje
        my_drone.land()  # Spusti dron
        break

cv2.destroyAllWindows()  # Zatvori sve prozore