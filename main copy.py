print("Doing IMPORTant work")
import socket
import pyaudio
import time
import wave
# import noisereduce as nr # should delete if not used
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, VADIterator

# wakeword import
import numpy as np
import openwakeword.model

print("Starting")

SERVER_IP = '192.168.178.22'
SERVER_PORT = 12345
CHUNK_STREAM = 1024
SILERO_CHUNK = 512
SILENCE_THRESHOLD = 40 # works when there is little noise. og 15
# IDEA: check the volume at the start, and set that as threshold.
SPEAKER_SAMPLE_RATE = 22050
WAKE_WORD_SENSITIVITY = 0.4

MIC_SAMPLE_RATE = 16000
format = pyaudio.paInt16
channels = 1
audio = pyaudio.PyAudio()

def receive_audio(socket, speaker):
	print("Receiving audio")
	
	while True: # Receiving audio
		data = socket.recv(CHUNK_STREAM)
			
		if data.endswith(b'END'):
			print("Done.")
			
			break
		else:
			speaker.write(data)

# Silero
print("Loading Silero")
silero_model = load_silero_vad()

def int2float(sound):
	abs_max = np.abs(sound).max()
	sound = sound.astype('float32')
	if abs_max > 0:
		sound *= 1/32768
	sound = sound.squeeze()
	return sound

end_silence = int(3.0 * MIC_SAMPLE_RATE / SILERO_CHUNK)
nothing_silence = int(6.0 * MIC_SAMPLE_RATE / SILERO_CHUNK)
def send_audio(socket, speaker):
		vad_iterator = VADIterator(silero_model, sampling_rate=MIC_SAMPLE_RATE)
		mic = audio.open(format=format, channels=channels, rate=MIC_SAMPLE_RATE, input=True, frames_per_buffer=SILERO_CHUNK) 

		silent_chunks = 0
		speech_started = False
		socket.sendall(b'START')
		while True:
			data = mic.read(SILERO_CHUNK) # og. CHUNK_STREAM
			audio_data = np.frombuffer(data, dtype=np.int16)			
			speech_dict = vad_iterator(int2float(audio_data))
			
			if not speech_dict:
				silent_chunks += 1
				if silent_chunks >= end_silence:
					if speech_started:
						socket.sendall(b'END')
						print("end\n")
						receive_audio(conn, speaker)
						# LED to done listening
						break
					elif silent_chunks >= nothing_silence:
						socket.sendall(b'NOTHING')
						print("Nothing heard")
						break
			else:
				speech_started = True
				silent_chunks = 0
			#TODO: breaks here if server disconnected
			socket.sendall(data)
		mic.close()


# wakeword variables/defs
framework = 'onnx'
model_path = ['ww/hal_v1.onnx']

print("Loading wakeword model")
owwModel = openwakeword.model.Model(wakeword_models=model_path, inference_framework=framework)
n_models = len(owwModel.models.keys())

def wakeword(mic):
	while True:
		audio = np.frombuffer(mic.read(CHUNK_STREAM), dtype=np.int16)
		# Feed to openWakeWord model
		prediction = owwModel.predict(audio)
		for mdl in owwModel.prediction_buffer.keys():
			scores = list(owwModel.prediction_buffer[mdl])
			if scores[-1] > WAKE_WORD_SENSITIVITY:
				return True
			if scores[-1] > 0.1:
				print("Not loud enough.")
				# TODO: Maybe do something with the LED

if __name__ == "__main__":
	audio_mic = audio.open(format=format, channels=channels, rate=MIC_SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_STREAM)
	audio_speaker = audio.open(format=format, channels=channels, rate=SPEAKER_SAMPLE_RATE, output=True, frames_per_buffer=CHUNK_STREAM)
	
	not_connected = True
	while not_connected:
		try:
			print("Connecting...")
			conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			conn.connect((SERVER_IP, SERVER_PORT))
			not_connected = False
		except:
			print("Can't connect")
			time.sleep(5)
	last_time = time.time()
	print("Waiting for wake word")
	while True:
		if wakeword(audio_mic):
			if time.time() - last_time > 2.0: # Debounce
				print("Listening...")
				send_audio(conn, audio_speaker)
				audio_mic = audio.open(format=format, channels=channels, rate=MIC_SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_STREAM)
				print("Waiting for wake word")
				last_time = time.time()
