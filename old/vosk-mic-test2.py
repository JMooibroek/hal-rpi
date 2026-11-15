import os
import sys
import json
import contextlib
import pyaudio
import io
from vosk import Model, KaldiRecognizer

model = Model(lang="en-us")

sample_rate = 16000
chunk_size = 8192
format = pyaudio.paInt16
channels = 1

p = pyaudio.PyAudio()
stream = p.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)
recognizer = KaldiRecognizer(model, sample_rate)

print("\nSpeak...\n")

while True:
	data = stream.read(chunk_size)
	if recognizer.AcceptWaveform(data):
		result_json = json.loads(recognizer.Result())
		text = result_json.get('text', '')
		if text:
			print("\r" + text, end='\n')
	else:
		partial_json = json.loads(recognizer.PartialResult())
		partial = partial_json.get('partial', '')
		sys.stdout.write('\r' + partial)
		sys.stdout.flush()
