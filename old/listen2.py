# Imports
import pyaudio
import numpy as np
from openwakeword.model import Model

FRAMEWORK = 'onnx'
MODEL_PATH = ['ww/hal_v1.onnx', 'ww/do_you_read_me__hal.onnx', 'ww/hey__hal.onnx']

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
owwModel = Model(wakeword_models=MODEL_PATH, inference_framework=FRAMEWORK)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#"*20)
    print("WakeWord Online")

    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])
            if scores[-1] > 0.5:
                print("Detected")
                break
