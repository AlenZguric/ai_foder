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
tello.streamon()

# Memorija za pohranu naredbi i senzorskih podataka
memorija = []
senzorska_memorija = []

# Početak snimanja podataka iz akcelerometra i žiroskopa
last_recorded_time = time()
def snimiSensorPodatke():
    """
    Kontinuirano snima podatke iz akcelerometra i žiroskopa.
    Optimizirano za smanjivanje broja spremljenih podataka.
    """
    global senzorska_memorija, last_recorded_time
    while True:
        current_time = time()
        if current_time - last_recorded_time >= 0.2:  # Snimaj podatke svakih 200 ms
            acc_x = tello.get_acceleration_x()
            acc_y = tello.get_acceleration_y()
            acc_z = tello.get_acceleration_z()
            roll = tello.get_roll()
            pitch = tello.get_pitch()
            yaw = tello.get_yaw()

            senzorska_memorija.append({
                "timestamp": current_time,
                "acc": (acc_x, acc_y, acc_z),
                "orientation": (roll, pitch, yaw)
            })
            last_recorded_time = current_time
        sleep(0.1)

def getKeyBoardInput():
    """
    Omogućuje unos kontrola putem tipkovnice za ručno upravljanje dronom.
    Također pohranjuje naredbe u memoriju.
    """
    ld, nN, gd, yv = 0, 0, 0, 0
    speed = 50
    global memorija

    if kp.getKey('LEFT'): ld = -speed
    elif kp.getKey('RIGHT'): ld = speed

    if kp.getKey('UP'): nN = speed
    elif kp.getKey('DOWN'): nN = -speed

    if kp.getKey('w'): gd = speed
    elif kp.getKey('s'): gd = -speed

    if kp.getKey('a'): yv = -speed
    elif kp.getKey('d'): yv = speed

    if kp.getKey('r'): tello.land()
    if kp.getKey('e'): tello.takeoff()

    # Spremi naredbe u memoriju ako nisu sve nule
    if any([ld, nN, gd, yv]):
        trenutna_komanda = {
            "ld": ld,
            "nN": nN,
            "gd": gd,
            "yv": yv,
            "timestamp": time()
        }
        memorija.append(trenutna_komanda)

    return [ld, nN, gd, yv]

def autonomniLet():
    """
    Ponavlja memorirane naredbe za autonomni let koristeći i podatke iz senzora.
    """
    global memorija, senzorska_memorija
    if not memorija:
        print("Nema spremljenih naredbi za autonomni let!")
        return

    print("Započinjem autonomni let...")

    # Rotacija za 180 stupnjeva prije izvođenja naredbi
    print("Rotiram dron za 180 stupnjeva...")
    tello.rotate_clockwise(180)
    sleep(2)  # Kratka pauza nakon rotacije

    pocetno_vrijeme = memorija[0]["timestamp"]

    for naredba in memorija:
        ld = naredba["ld"]
        nN = naredba["nN"]
        gd = naredba["gd"]
        yv = naredba["yv"]
        trenutak = naredba["timestamp"]
        vrijeme_cekanja = trenutak - pocetno_vrijeme
        sleep(vrijeme_cekanja)  # Pričekaj odgovarajući interval
        tello.send_rc_control(ld, nN, gd, yv)
        pocetno_vrijeme = trenutak

        # Provjera trenutnih senzorskih podataka i usporedba s ciljem
        trenutni_acc_x = tello.get_acceleration_x()
        trenutni_acc_y = tello.get_acceleration_y()
        trenutni_acc_z = tello.get_acceleration_z()
        trenutni_roll = tello.get_roll()
        trenutni_pitch = tello.get_pitch()
        trenutni_yaw = tello.get_yaw()

        print(f"Trenutni Acc: ({trenutni_acc_x}, {trenutni_acc_y}, {trenutni_acc_z}), Orientation: ({trenutni_roll}, {trenutni_pitch}, {trenutni_yaw})")

    # Reprodukcija senzorskih podataka za precizno pozicioniranje
    print("Reprodukcija senzorskih podataka za povratak...")
    for podatak in reversed(senzorska_memorija):
        acc = podatak["acc"]
        orientation = podatak["orientation"]
        print(f"Snimljeni Acc: {acc}, Snimljena Orientation: {orientation}")
        sleep(0.2)

    # Zaustavi dron na 1.5 sekundi prije slijetanja
    tello.send_rc_control(0, 0, 0, 0)
    print("Zaustavljanje prije slijetanja...")
    sleep(1.5)

    # Slijetanje
    print("Slijećem...")
    tello.land()
    print("Autonomni let završen!")

# Pokreni snimanje senzorskih podataka u pozadini
import threading
sensor_thread = threading.Thread(target=snimiSensorPodatke, daemon=True)
sensor_thread.start()

# Glavna petlja
try:
    while True:
        # Dohvati trenutni okvir s drona
        frame = tello.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ručno upravljanje
        vals = getKeyBoardInput()
        tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

        # Provjera za autonomni mod
        if kp.getKey('m'):
            autonomniLet()

        # Prikaz trenutnog okvira
        cv2.imshow("Tello Navigation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Dogodila se greška: {e}")

finally:
    # Zaustavljanje streama i čišćenje
    try:
        tello.streamoff()
    except:
        print("Stream je već isključen ili nije moguće isključiti.")
    cv2.destroyAllWindows()
