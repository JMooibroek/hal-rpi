import json
import queue
import sys
import sounddevice as sd

from vosk import Model, KaldiRecognizer

device = 1

device_info = sd.query_devices(device, "input")
samplerate = int(device_info["default_samplerate"])
model = Model(lang="en-us")
rec = KaldiRecognizer(model, samplerate)

q = queue.Queue()

def callback(indata, frames, time, status):
#	if status:
#		print(status, file=sys.stderr)
	q.put(bytes(indata))

try:
	with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=device, dtype="int16", channels=1, callback=callback):
		print("#"*80)
		print("Press CTRL-C to stop")
		print("#"*80)

		while True:
			data = q.get()
			if rec.AcceptWaveform(data):
				res = json.loads(rec.Result())
				print(res["text"])
#			else:
#				print(rec.PartialResult())
except KeyboardInterrupt:
	print("\nDone.")
