import cv2
from djitellopy import Tello
import numpy as np
from time import sleep
import KeyPressFunc as kp  # Provjeri je li modul dostupan
import pygame  # Ako već nije importirano


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

# Funkcija za unos s tipkovnice
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

# Procesiranje okvira s drona
while True:
    # Dohvati okvir s drona
    frame = tello.get_frame_read().frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = frame.shape

    baterija = tello.get_battery()
    visina = tello.get_distance_tof()

    # Prikaz stanja baterije
    cv2.putText(frame, f"Battery: {baterija}%", (width - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Visina: {visina} cm", (width - 400, 30),

                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Kontrola drona putem tipkovnice
    vals = getKeyBoardInput()
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    sleep(0.05)
    
    # Obrada za YOLO detekciju
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
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
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
            
        '''
                '''

    # Prikaz rezultata
    cv2.imshow("Tello Object Detection", frame)
    
    # Prekini petlju ako je pritisnut 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Isključi stream i zatvori prozore
tello.streamoff()
cv2.destroyAllWindows()
