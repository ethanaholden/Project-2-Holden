#Project 2, Ethan Holden
import wave
import sounddevice as sd
import librosa
from scipy.io.wavfile import write
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import librosa.display
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
import glob
import time
import IPython.display as ipd
from audio2numpy import open_audio
import pydub
import soundfile as sf
import math
import pyaudio
from scipy.signal import butter,filtfilt,cheby1
''
#Record the voice sample
duration = 5
sr = 44100
'''
recording = sd.rec(int(duration*sr), samplerate=sr, channels=1)
print("recording...............")
sd.wait()
write("sound.wav",sr,recording)
'''
data, sampling_rate = librosa.load("sound.wav", sr = 44100)
recording, sampling_rate = librosa.load("sound.wav", sr = 44100)
print(data.shape)
print(recording.shape)

''
#produce sin wave 
sr = 44100

def sin(freq, sample_rate, duration):
    #a = np.linspace(0, duration, sample_rate*duration, endpoint=False)
    x = np.arange(0, duration,duration/(sample_rate*duration))
    frequencies = x*freq
    y = np.sin((2*np.pi)*frequencies)
    #print (a.shape)
    print (y.shape)
    #print (x)
    return y
tone = sin(5000, 44100, 5)
#tone = np.rot90(np.array(sin(5000, 44100, 5), ndmin=2))
print(tone.shape)

'''
plt.plot(t,x)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()
'''
''
#Q1
sd.play(tone,44100)
sf.write('teamholden-sinetone.wav', tone,sr, subtype='PCM_24')

freq = librosa.amplitude_to_db(np.abs(librosa.stft(tone)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(freq, x_axis='time', y_axis='linear', ax=ax, sr=sr, fmin=1000, fmax=8000)
ax.set(title='Spectrogram of Recording.')
fig.colorbar(img, ax=ax)
plt.show()

#Q2
#First a reference to make sure my math was right
#chirpsig = librosa.chirp(fmin=0,fmax=8000,duration=5,linear=True,sr=sr)

#Making the signal with a linearly increasing frequency  and normalizing it
N = int(sr * duration)
t = np.linspace(0, duration, N)
f_min = 0
f_max = 4000 #halving it to simplify math
f_t = f_min + (f_max - f_min) * t / duration
chirp = np.sin(2 * np.pi * f_t * t)
chirp = chirp / np.max(np.abs(chirp))

sd.play(chirp,sr)
sf.write('teamholden-chirp.wav', chirp,sr, subtype='PCM_24')

freq = librosa.amplitude_to_db(np.abs(librosa.stft(chirp)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(freq, x_axis='time', y_axis='linear', ax=ax, sr=sr, fmin=1000, fmax=8000)
ax.set(title='Spectrogram of Recording.')
fig.colorbar(img, ax=ax)
plt.show()
''
#Q3
note1 = sin(440, sr, 0.5)
note2 = sin(494, sr, 0.75)
note3 = sin(392, sr, .9)
note4 = sin(196, sr, 0.75)
note5 = sin(294, sr, 1.75)
closeenc = np.append(np.append(np.append(np.append(note1,note2),note3),note4),note5)
sd.play(closeenc,sr)
print (closeenc.shape)
sf.write('teamholden-cetk.wav', closeenc,sr, subtype='PCM_24')

freq = librosa.amplitude_to_db(np.abs(librosa.stft(closeenc)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(freq, x_axis='time', y_axis='linear', ax=ax, sr=sr, fmin=1000, fmax=8000)
ax.set(title='Spectrogram of Recording.')
fig.colorbar(img, ax=ax)
plt.show()
''
#Q4

print(recording.shape,"recording")
#recording = recording.flatten()
recording = data.flatten()
print(recording.shape,'flatrec')
print(tone.shape)
speechchirp = np.add(recording,tone)
#speechchirp = np.rot90(speechchirp,3)
print(speechchirp.shape)
sd.play(speechchirp)
sf.write('teamholden-speechchirp.wav', speechchirp,sr, subtype='PCM_24')

freq = librosa.amplitude_to_db(np.abs(librosa.stft(speechchirp)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(freq, x_axis='time', y_axis='linear', ax=ax, sr=sr, fmin=1000, fmax=8000)
ax.set(title='Spectrogram of Recording.')
fig.colorbar(img, ax=ax)
plt.show()
print('done')

#Q5
b, a = cheby1(4, 5, 3000, btype='low', analog=False, fs=44100)
filteredspeechsine = filtfilt(b,a,speechchirp)
sd.play(filteredspeechsine)
sf.write('teamholden-filteredspeechsine.wav', filteredspeechsine,sr, subtype='PCM_24')

freq = librosa.amplitude_to_db(np.abs(librosa.stft(filteredspeechsine)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(freq, x_axis='time', y_axis='linear', ax=ax, sr=sr, fmin=1000, fmax=8000)
ax.set(title='Spectrogram of Recording.')
fig.colorbar(img, ax=ax)
plt.show()

#Q6
stereospeechsine = np.rot90(np.stack([recording,speechchirp]),3)
print(stereospeechsine.shape)
sd.play(stereospeechsine)
sf.write('teamholden-stereospeechsine.wav', stereospeechsine,sr, subtype='PCM_24')
'''
freq = librosa.amplitude_to_db(np.abs(librosa.stft(recording)), ref=np.max)
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
img = librosa.display.specshow(freq, x_axis='time', y_axis='linear', ax=ax, sr=sr, fmin=1000, fmax=8000)
ax.set(title='Spectrogram of Recording.')
fig.colorbar(img, ax=ax)
plt.show()
'''
freq = librosa.amplitude_to_db(np.abs(librosa.stft(recording)), ref=np.max)
freq2 = librosa.amplitude_to_db(np.abs(librosa.stft(speechchirp)), ref=np.max)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

img = librosa.display.specshow(freq, x_axis='time', y_axis='linear', ax=ax[0], sr=sr)
img2 = librosa.display.specshow(freq2, x_axis='time', y_axis='linear', ax=ax[1], sr=sr)

ax[0].set(title='Left Ear')
ax[0].label_outer()
ax[1].label_outer()
ax[1].set(title='Right Ear')
fig.colorbar(img, ax=[ax[0],ax[1]])
plt.show()
''
