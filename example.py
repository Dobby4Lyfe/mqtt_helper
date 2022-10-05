from mqtt_helper import Mqtt, STATE_MAP
import time

def dobby_said(msg):
    print(f"Dobby said: {msg}")

def change_state(state):

    print(f"New State: {STATE_MAP[state]}")

dobby = Mqtt()

dobby.subscribe('/dobby', dobby_said)
dobby.subscribe('/state', change_state)


count = 0



while 1:
    # print(count)
    dobby.publish('/dobby', f'hello {count}')
    count += 1
    time.sleep(10)
