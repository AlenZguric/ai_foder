import cv2
from djitellopy import Tello
import numpy as np
from time import sleep
import KeyPressFunc as kp
import pygame  # Ako već nije importirano
import threading

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
pygame.init()
pygame.display.set_mode((1, 1))
kp.init()

# Load class labels (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize Tello drone
tello = Tello()

# Connect to Tello
tello.connect()

# Start video stream
tello.streamon()

# Global variable to manage control state
control_active = False

def getKeyBoardInput():
    ld, nN, gd, yv = 0, 0, 0, 0
    speed = 50  # Brzina kretanja

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

    return [ld, nN, gd, yv]

def center_and_land_on_object(box, frame_width, frame_height, visina, label):
    global control_active
    x, y, w, h = box
    center_x = x + w // 2
    center_y = y + h // 2

    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    offset_x = center_x - frame_center_x
    offset_y = center_y - frame_center_y

    lr = -20 if offset_x > 20 else (20 if offset_x < -20 else 0)
    fb = -20 if offset_y > 20 else (20 if offset_y < -20 else 0)

    if visina > 50:  # Adjust vertical movement based on height
        ud = -20
    elif visina < 30:
        ud = 20
    else:
        ud = 0

    tello.send_rc_control(lr, fb, ud, 0)
    sleep(0.1)

    # Check if centered
    if lr == 0 and fb == 0 and ud == 0:
        print(f"{label.capitalize()} centered. Landing...")
        tello.land()
        control_active = False

def process_frame(frame, frame_width, frame_height, visina):
    global control_active
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == "cell phone" and not control_active:
                control_active = True
                threading.Thread(target=center_and_land_on_object, args=([x, y, w, h], frame_width, frame_height, visina, label)).start()

# Procesiranje okvira s drona
try:
    while True:
        tello.set_video_direction(tello.CAMERA_DOWNWARD)  # Prebacivanje na donju kameru
        frame = tello.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame.shape

        baterija = tello.get_battery()
        visina = tello.get_distance_tof()

        # Draw center point
        cv2.circle(frame, (width // 2, height // 2), 5, (0, 0, 255), -1)

        # Prikaz stanja baterije
        cv2.putText(frame, f"Battery: {baterija} %", (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Visina: {visina} cm", (width - 400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Kontrola drona putem tipkovnice
        vals = getKeyBoardInput()
        tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])
        sleep(0.05)

        # Obrada okvira
        threading.Thread(target=process_frame, args=(frame, width, height, visina)).start()

        # Prikaz rezultata
        cv2.imshow("Tello Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Prekid programa.")
finally:
    tello.streamoff()
    cv2.destroyAllWindows()
