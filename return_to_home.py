import cv2
from djitellopy import Tello
import numpy as np
from time import sleep, time
import KeyPressFunc as kp
import pygame

# Inicijalizacija
pygame.init()
pygame.display.set_mode((1, 1))
kp.init()

# Postavljanje drona
tello = Tello()
tello.connect()
pocetno_vrijeme = time()  # Inicijalizacija početnog vremena
tello.streamon()

# Memorija za pohranu podataka
memorija = []
senzorski_podaci = []
trenutna_naredba = None
start_vrijeme = None

# Funkcija za završetak trenutne naredbe i spremanje u memoriju
def zavrsi_naredbu():
    global trenutna_naredba, start_vrijeme, memorija
    if trenutna_naredba is not None:
        trajanje = int((time() - start_vrijeme) * 1000)  # Trajanje u milisekundama
        trenutna_naredba["trajanje"] = trajanje
        memorija.append(trenutna_naredba)
        trenutna_naredba = None
        start_vrijeme = None

# Funkcija za spremanje nove naredbe
def spremi_naredbu(lr, fb, ud, yv):
    global trenutna_naredba, start_vrijeme
    if trenutna_naredba is None or trenutna_naredba["lr"] != lr or trenutna_naredba["fb"] != fb \
            or trenutna_naredba["ud"] != ud or trenutna_naredba["yv"] != yv:
        zavrsi_naredbu()
        trenutna_naredba = {"lr": lr, "fb": fb, "ud": ud, "yv": yv}
        start_vrijeme = time()

# Funkcija za dohvaćanje unosa s tipkovnice
def get_keyboard_input(drone):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if kp.getKey('LEFT'): lr = -speed
    elif kp.getKey('RIGHT'): lr = speed

    if kp.getKey('UP'): fb = speed
    elif kp.getKey('DOWN'): fb = -speed

    if kp.getKey('w'): ud = speed
    elif kp.getKey('s'): ud = -speed

    if kp.getKey('a'): yv = -speed
    elif kp.getKey('d'): yv = speed

    if kp.getKey('q'):
        drone.land()
        print("Dron je sletio.")
    if kp.getKey('e'):
        drone.takeoff()
        global pocetno_vrijeme
        pocetno_vrijeme = time()
        print("Dron je poletio i uključena je prednja kamera.")
        tello.set_video_direction(tello.CAMERA_FORWARD)  # Uključivanje prednje kamere

    spremi_naredbu(lr, fb, ud, yv)
    return [lr, fb, ud, yv]

# Funkcija za bilježenje senzorskih podataka
def biljezi_senzorske_podatke():
    global senzorski_podaci
    try:
        stanje = tello.get_current_state()
        senzorski_podaci.append({
            "pitch": stanje.get("pitch", 0),
            "roll": stanje.get("roll", 0),
            "yaw": stanje.get("yaw", 0),
            "visina": stanje.get("h", 0),
        })
    except Exception as e:
        print(f"Greška pri bilježenju senzorskih podataka: {e}")

# Funkcija za povratak na početnu točku
def return_to_home():
    global memorija
    zavrsi_naredbu()  # Završava posljednju aktivnu naredbu

    if not memorija:
        print("Nema spremljenih podataka za povratak!")
        return

    print("Započinjem povratak na početnu točku...")

    for komanda in reversed(memorija):
        lr = -komanda["lr"]
        fb = -komanda["fb"]
        ud = -komanda["ud"]
        yv = -komanda["yv"]
        trajanje = komanda["trajanje"] / 1000

        tello.send_rc_control(lr, fb, ud, yv)
        sleep(trajanje)

    print("Izvršen povratak prema memoriranim podacima. Pokrećem detekciju kruga...")
    detect_orange_circle_and_land()

# Funkcija za detekciju narančastog kruga i centriranje prije slijetanja
def detect_orange_circle_and_land():
    print("Tražim narančasti krug za slijetanje...")
    tello.set_video_direction(tello.CAMERA_DOWNWARD)  # Prebacivanje na donju kameru
    print("Uključena donja kamera za detekciju kruga.")

    while True:
        downward_frame = tello.get_frame_read().frame
        downward_frame = cv2.cvtColor(downward_frame, cv2.COLOR_RGB2BGR)
        downward_frame = cv2.resize(downward_frame, (320, 240))

        hsv_frame = cv2.cvtColor(downward_frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

            okvir_x = downward_frame.shape[1] // 2
            okvir_y = downward_frame.shape[0] // 2

            if radius > 10:
                cv2.circle(downward_frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(downward_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                if abs(x - okvir_x) > 5:
                    lr = -20 if x > okvir_x else 20
                    tello.send_rc_control(lr, 0, 0, 0)
                elif abs(y - okvir_y) > 5:
                    fb = -20 if y > okvir_y else 20
                    tello.send_rc_control(0, fb, 0, 0)
                else:
                    print("Krug centriran. Zaustavljam i slijećem...")
                    tello.send_rc_control(0, 0, 0, 0)
                    sleep(1.5)
                    tello.land()
                    return

        cv2.imshow("Maska za narančasti krug", mask)
        cv2.imshow("Donja Kamera", downward_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Glavna petlja
try:
    while True:
        frame = tello.get_frame_read().frame
        biljezi_senzorske_podatke()

        lr, fb, ud, yv = get_keyboard_input(tello)
        tello.send_rc_control(lr, fb, ud, yv)

        cv2.imshow("Tello Navigation", frame)
        if kp.getKey('m'):
            cv2.destroyWindow("Tello Navigation")
            return_to_home()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Dogodila se greška: {e}")

finally:
    try:
        zavrsi_naredbu()
        tello.streamoff()
    except:
        print("Stream je već isključen ili nije moguće isključiti.")
    cv2.destroyAllWindows()
