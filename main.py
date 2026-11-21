import time
from piper import PiperVoice, SynthesisConfig
import pyaudio
# from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, VADIterator
import openwakeword.model 
import numpy
from cactus import Whistle
import listen

CHUNK_STREAM = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
MIC_SAMPLE_RATE = 16000
SPEAKER_SAMPLE_RATE = 22050
SILERO_CHUNK = 512
SILENCE_THRESHOLD = 40
WAKE_WORD_SENSITIVITY = 0.4

audio = pyaudio.PyAudio()
audio_speaker = audio.open(format=FORMAT, channels=CHANNELS, rate=SPEAKER_SAMPLE_RATE, output=True, frames_per_buffer=CHUNK_STREAM)
mic = audio.open(format=FORMAT, channels=CHANNELS, rate=MIC_SAMPLE_RATE, input=True, frames_per_buffer=SILERO_CHUNK) 

# silero_model = load_silero_vad()
voice = PiperVoice.load("voice/denoised-hal.onnx") # diyengineering
whistle = Whistle() # weights="path/to/your/weights"

end_silence = int(3.0 * MIC_SAMPLE_RATE / SILERO_CHUNK)
nothing_silence = int(6.0 * MIC_SAMPLE_RATE / SILERO_CHUNK)

owwModel = openwakeword.model.Model(wakeword_models=["wakewords/hey_hal.tflite", "wakewords/ok_hal.tflite"])

syn_config = SynthesisConfig(
    volume=2.0,  # half as loud
    length_scale=1.2,  # speed
    noise_scale=0.0,  # audio variation
    noise_w_scale=0.0,  # speaking variation
    normalize_audio=False, # use raw audio from voice
)

def speak(text: str):
	for chunk in voice.synthesize(text, syn_config=syn_config):
		audio_speaker.write(chunk.audio_int16_bytes)

def wakeword():
	while True:
		audio = numpy.frombuffer(mic.read(CHUNK_STREAM), dtype=numpy.int16)
		# Feed to openWakeWord model
		owwModel.predict(audio)
		for model in owwModel.prediction_buffer.keys():
			scores = list(owwModel.prediction_buffer[model])
			if scores[-1] > WAKE_WORD_SENSITIVITY:
				return True
			if scores[-1] > 0.1:
				print("Not loud enough.")
				# TODO: Maybe do something with the LED

if __name__ == "__main__":
	print("Start.\n")
	last_time = time.time()
	while True:
			if wakeword():
				if time.time() - last_time > 2.0: # Debounce
					text = listen.listen_until_silence(whistle, keywords=["Jamaro", "Mooibroek", "Safran", "Petra", "Barry", "HAL"])
					speak(text)
					last_time = time.time()