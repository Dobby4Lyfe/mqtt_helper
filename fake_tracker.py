from mqtt_helper import Mqtt, STATE
from enum import Enum
import math
import time
import random
import json

mqtt = Mqtt()



WIDTH = 800
HEIGHT = 600

def dist(x,y):
    return math.dist([x,y],[WIDTH/2, HEIGHT/2])

def calc(x,y):
    distance = [x - WIDTH/2, y  - HEIGHT/2]
    norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
    direction = [distance[0] / norm, distance[1] / norm]  
    return direction  

def send(x,y):
    msg = {
        'x' : x,
        'y' : y
    }
    mqtt.publish('/face/1', json.dumps(msg))

while 1:
    mqtt.publish('/state', STATE['SCANNING'])
    time.sleep(random.randint(2,5))
    
    mqtt.publish('/state', STATE['FOUND'])
    for i in range(random.randrange(10,50)):
        x = random.randint(0,WIDTH)
        y = random.randint(0,HEIGHT)
        send(x,y)
        time.sleep(0.1)


    mqtt.publish('/state', STATE['LOCK'])
    time.sleep(random.randint(2,5))
    mqtt.publish('/state', STATE['LOST'])
    time.sleep(random.randint(2,5))
    
