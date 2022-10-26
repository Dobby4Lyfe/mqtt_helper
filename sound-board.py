#%%
from mqtt_helper import Mqtt
import pygame
import time

pygame.mixer.init()

mqtt = Mqtt()

CURRENT_SCENE = 0

SOUND_MAPPING = {
  0: ['samples/loop_amen.wav',1,False],
  1: ['samples/ambi_drone.wav',1,False],
  2: ['samples/bass_voxy_c.wav',1,False],
  3: ['samples/hello.ogg',1,False],

  4: ['samples/pen-beat.wav',0.5,True],
  5: ['samples/pen.wav',1,False],
  6: ['samples/pineapple.wav',1,False],
  7: ['samples/ppap.wav',1,False],
 
}

sounds = []
is_loop = []
sounds_playing = []
for key,data in SOUND_MAPPING.items():
  soundfile, volume, loop = data
  sounds.append(0)
  is_loop.append(False)
  sounds_playing.append(False)
  sounds[key] =  pygame.mixer.Sound(soundfile)
  sounds[key].set_volume(volume)
  is_loop[key] = loop


def playSound(sound_id):
    print("playing ", sound_id)
    if sounds_playing[sound_id]:
        sounds[sound_id].stop()
        

    if is_loop[sound_id]:
        if not sounds_playing[sound_id]:
            sounds[sound_id].play(loops = -1)
            sounds_playing[sound_id] = True
    else:
        sounds[sound_id].play()
        sounds_playing[sound_id] = True

def stopSound(sound_id):
    sounds[sound_id].stop()
    sounds_playing[sound_id] = False

def stopAllSounds():
    for sound_id in SOUND_MAPPING.keys():
        sounds[sound_id].stop()
        sounds_playing[sound_id] = False

def onSound(msg):
  topic = msg['topic']
  val = int(msg['payload'])
  sound = int(topic.split('/')[2])
  print(sound, val)
  if val == 1:
    playSound(sound)
  else:
    stopSound(sound)

for i in SOUND_MAPPING.keys():
  mqtt.subscribe(f'/sounds/{i}', onSound) 

while 1:
  time.sleep(1)
