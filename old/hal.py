# vosk imports
import os
import sys
import json
import contextlib
import pyaudio
import io
import time
from vosk import Model, KaldiRecognizer

# wakeword import
import numpy as np
import openwakeword.model

# vosk variables/defs
model = Model(lang="en-us")
sample_rate = 16000
chunk_size_vosk = 8192
format = pyaudio.paInt16
channels = 1
audio = pyaudio.PyAudio()
recognizer = KaldiRecognizer(model, sample_rate)


def speech_to_text(stream):
	silence = 0
	text_to_return = ""
	print("Running Speech Recognition")
	while True:
		data = stream.read(chunk_size_vosk)
		if recognizer.AcceptWaveform(data):
			result_json = json.loads(recognizer.Result())
			text = result_json.get('text', '')
			if text:
				text_to_return += text
				print("\r" + text, end='\n')
				silence = 0
		else:
			partial_json = json.loads(recognizer.PartialResult())
			partial = partial_json.get('partial', '')
			if partial:
				sys.stdout.write('\r' + partial)
				sys.stdout.flush()
			else:
				silence += 1
				
		if text_to_return and silence > 5:
			return text_to_return
		if silence > 10:
			return ""

# wakeword variables/defs
framework = 'onnx'
model_path = ['ww/hal_v1.onnx']
chunk_size_ww = 1024

owwModel = openwakeword.model.Model(wakeword_models=model_path, inference_framework=framework)
n_models = len(owwModel.models.keys())

def wakeword(stream):
	while True:
		audio = np.frombuffer(stream.read(chunk_size_ww), dtype=np.int16)
		# Feed to openWakeWord model
		prediction = owwModel.predict(audio)
		for mdl in owwModel.prediction_buffer.keys():
			# Add scores in formatted table
			scores = list(owwModel.prediction_buffer[mdl])
			if scores[-1] > 0.5:
				return True

if __name__ == "__main__":
	stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size_ww)
	last_time = time.time()
	print("listening for wake word")
	while True:
		if wakeword(stream):
			if time.time() - last_time > 2.0: # Debounce
				stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size_vosk)
				print("Awake")
				speech_to_text(stream)
				print("Listening for wake word")
				last_time = time.time()
