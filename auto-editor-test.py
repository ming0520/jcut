'''auto-editor.py'''
#python auto-editor.py lib.mp4 --frame_margin 8 --silent_threshold 0.03
# external python libraries
from pydub import AudioSegment
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
from PIL import Image # pip install pillow
import numpy as np
import librosa


# internal python libraries
import math
import os
import argparse
import subprocess
import sys
import time
from re import search
from datetime import timedelta
from shutil import move, rmtree, copyfile
from multiprocessing import Process

import time

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv, -minv)

def _seconds(value):
    if isinstance(value, str):  # value seems to be a timestamp
        _zip_ft = zip((3600, 60, 1, 1/framerate), value.split(':'))
        return sum(f * float(t) for f,t in _zip_ft)
    elif isinstance(value, (int, float)):  # frames
        return value / framerate
    else:
        return 0

def _timecode(seconds):
    return '{h:02d}:{m:02d}:{s:02d}.{f:02d}' \
            .format(h=int(seconds/3600),
                    m=int(seconds/60%60),
                    s=int(seconds%60),
                    f=round((seconds-int(seconds))*framerate))


def _frames(seconds):
    return seconds * framerate

def timecode_to_frames(timecode, start=None):
    return _frames(_seconds(timecode) - _seconds(start))

def frames_to_timecode(frames, start=None):
    return _timecode(_seconds(frames) + _seconds(start))


#parameter for ffmpeg to convert the file
BITRATE_AUDIO = "160k"
AUDIO_CHANEL = str(2)
AUDIO_RATE = str(44100)
AUDIO_NAME = 'audio'
AUDIO_OUTPUT_FORMAT = ".wav"
AUDIO_OUTPUT = f'{AUDIO_NAME}{AUDIO_OUTPUT_FORMAT}'
FRAME_RATE = 30.0

#parameter for changing the file
SILENT_SPEED = 999999
VIDEO_SPEED = 1.0
LOUDNESS_THRESHOLD = 2.00
SILENT_THRESHOLD = 0.04
FRAME_MARGIN = 10
VERBOSE = True
FADE_SIZE = 400

FRAME_SPREADAGE = FRAME_MARGIN

#parameter for folder
TEMP = '.TEMP'
CACHE = '.CACHE'
version = '20w22b'

INPUT_FILE = 'example.mp4'

#check is it the audio file exits
if os.path.exists(f'{AUDIO_OUTPUT}'):
    os.remove(f'{AUDIO_OUTPUT}')
    print("Removed " + f'{AUDIO_OUTPUT}')

#convert to wav file
cmd = ['ffmpeg','-i',INPUT_FILE, '-b:a', BITRATE_AUDIO, '-ac', AUDIO_CHANEL, '-ar', AUDIO_RATE, '-vn', f'{AUDIO_OUTPUT}']
if(not VERBOSE):
    cmd.extend(['-nostats', '-loglevel', '0'])
subprocess.call(cmd)

start_time = time.time()
sampleRate, audioData = wavfile.read(f'{AUDIO_OUTPUT}')

frameRate = FRAME_RATE
audioSampleCount = len(audioData) # equal audio.shape[0]
maxAudioVolume = getMaxVolume(audioData)

# audioSampleCount = audioData.shape[0]
samplesPerFrame = sampleRate / frameRate
audioFrameCount = int(math.ceil(audioSampleCount / samplesPerFrame))
hasLoudAudio = np.zeros((audioFrameCount))

# print(f'audioSampleCount={audioSampleCount} maxAudioVolume={maxAudioVolume} hasLoudAudio={hasLoudAudio} samplesPerFrame={samplesPerFrame} audioFrameCount={audioFrameCount}')

for i in range(audioFrameCount):
    start = int(i * samplesPerFrame)
    end = min(int((i+1) * samplesPerFrame), audioSampleCount)
    audiochunks = audioData[start:end]
    maxchunksVolume = getMaxVolume(audiochunks) / maxAudioVolume
    if(maxchunksVolume >= LOUDNESS_THRESHOLD):
        hasLoudAudio[i] = 2
    elif(maxchunksVolume >= SILENT_THRESHOLD):
        hasLoudAudio[i] = 1
    # print(f'start={start} end= {end} maxCunksVolume={maxchunksVolume} hasLoudAudio[{i}]={hasLoudAudio[i]}')  


chunks = [[0, 0, 0]]
shouldIncludeFrame = np.zeros((audioFrameCount))
# print(f'Total chunks={audioFrameCount}')

for i in range(audioFrameCount):
    chunksOp = False
    start = int(max(0, i-FRAME_SPREADAGE))
    end = int(min(audioFrameCount, i+1+FRAME_SPREADAGE))
    shouldIncludeFrame[i] = min(1,np.max(hasLoudAudio[start:end]))
    
    if(i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]):
        chunks.append([chunks[-1][1], i, shouldIncludeFrame[i-1]])
        chunksOp = True
    # print(f'start={start} end={end} shouldIncludeFrame[{i}]={shouldIncludeFrame[i]} chunksOp={chunksOp}')

chunks.append([chunks[-1][1], audioFrameCount, shouldIncludeFrame[i-1]])
chunks = chunks[1:]
print(chunks)
# print(f'Processing time for librosa {time.time()-start_time}')

framerate=30.0

timecode = [[0,0,0]]
framecode = [[0,0,0]]
for i in chunks:
    # startTime = frames_to_timecode(int(i[0]))
    # endTime = frames_to_timecode(int(i[1]))
    # isInclude = i[2]
    # framecode.append([startTime,endTime,isInclude])
    framecode.append([i[0],i[1],i[2]])
    if i[2] > 0:
        timecode.append([frames_to_timecode(int(i[0]),start=None),frames_to_timecode(int(i[1]),start=None),i[2]])

for i in timecode:
    if i[2] > 0:
        print(f'{INPUT_FILE},{i[0]},{i[1]}')
between = []
counter = 0
for i in framecode:
    if i[2] > 0:
            between.append(f'between(n\,{i[0]},{i[1]})')

# Process the video by using select filter in ffmpeg
# betweens = '+'.join(between)
# slt = '\"select=\'' + betweens + '\'' + ',setpts=N/FRAME_RATE/TB\"'
# aslt = '\"aselect=\'' + betweens + '\'' + ',asetpts=N/SR/TB\"'
# sltFilter = ['ffmpeg','-i',f'{INPUT_FILE}', '-vf', f'{slt}','-af', f'{aslt}', 'selectFILTER.mp4']
# subprocess.call(sltFilter)

#export time code to text
f= open("chunks.txt","w+")
for i in timecode:
    if i[2] > 0:
        f.write(f'file {INPUT_FILE}\ninpoint {i[0]}\noutpoint {i[1]}\n')
        # print(f'{INPUT_FILE},{i[0]},{i[1]}')

if os.path.exists(f'{INPUT_FILE}_CONCATED.mp4'):
    os.remove(f'{INPUT_FILE}_CONCATED.mp4')
    print("Removed " + f'{INPUT_FILE}_CONCATED.mp4')

concat = ['ffmpeg','-y','-f','concat','-safe','0','-i','chunks.txt','-async','1', '-framerate',f'{FRAME_RATE}','-c:v','copy','-c:a','aac','-movflags','+faststart',f'{INPUT_FILE}_CONCATED.mp4']
subprocess.call(concat)