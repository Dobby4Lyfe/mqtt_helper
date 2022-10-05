import paho.mqtt.client as mqtt
from enum import Enum
import json

STATE = {
    'SCANNING': 0,
    'FOUND' : 1,
    'LOCK': 2,
    'LOST' : 3
}
STATE_MAP = list(STATE.keys())

class Mqtt:

    def __init__(self, hostname='localhost', port=1883, keepalive=60):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.callbacks = {}
        self.client.connect(hostname, port, keepalive)
        self.client.loop_start()
       

    def subscribe(self, topic, callback):

        self.callbacks[topic] = callback
        self.client.subscribe(topic)

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        for topic in self.callbacks.keys():
            self.client.subscribe(topic)

    def on_message(self, client, userdata, msg):
        print(msg.topic+ " " + msg.payload.decode())

        try:
            payload = json.decode(msg.payload)
        except:
            payload = msg.payload.decode()
        
        if msg.topic == '/state':
            payload = int(payload)

        # loop through callbacks and 
        if msg.topic in self.callbacks:
            self.callbacks[msg.topic](payload)
        

    def publish(self, topic, payload):
        print(f"Publish: {topic} {payload}")
        self.client.publish(topic, payload)


