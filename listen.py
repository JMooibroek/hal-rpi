import queue
import threading

import numpy
import pyaudio
import torch
from silero_vad import load_silero_vad

from cactus.agent.whistle import SAMPLE_RATE


def listen_until_silence(whistle, language=None, keywords=None):
    vad = load_silero_vad()
    vad.eval()
    vad.reset_states()

    audio_queue = queue.Queue()
    stop_event = threading.Event()
    errors = []
    transcript_parts = []

    def audio_chunks():
        blocks = []
        held = 0

        while True:
            data = audio_queue.get()

            if data is None:
                if blocks:
                    yield numpy.concatenate(blocks)
                return

            blocks.append(data)
            held += len(data)

            # Give whistle.stream roughly one second of audio at a time.
            if held >= SAMPLE_RATE:
                yield numpy.concatenate(blocks)
                blocks = []
                held = 0

    def transcribe():
        try:
            committed = ""

            for step in whistle.stream(
                audio_chunks(),
                language=language,
                keywords=keywords,
            ):
                text = step["text"]
                if text:
                    transcript_parts.append(text)
                    print((" " if committed else "") + text, end="", flush=True)
                    committed += (" " if committed else "") + text
        except (RuntimeError, OSError) as error:
            errors.append(error)
            stop_event.set()

    worker = threading.Thread(target=transcribe, daemon=True)
    pa = pyaudio.PyAudio()
    stream = None

    try:
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=512,
        )

        print("Listening...")

        worker.start()
        speech_seen = False
        silent_samples = 0
        silence_limit = 3 * SAMPLE_RATE

        while not stop_event.is_set():
            # Silero VAD expects 512 samples per frame at 16 kHz.
            raw = stream.read(512, exception_on_overflow=False)
            frame = numpy.frombuffer(raw, dtype=numpy.float32).copy()
            audio_queue.put(frame)

            with torch.inference_mode():
                probability = vad(torch.from_numpy(frame), SAMPLE_RATE).item()

            if probability >= 0.5:
                speech_seen = True
                silent_samples = 0
            elif speech_seen:
                silent_samples += len(frame)
                if silent_samples >= silence_limit:
                    break

    except OSError as error:
        raise RuntimeError(f"Microphone error: {error}") from error
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()

        pa.terminate()
        audio_queue.put(None)

        if worker.is_alive():
            worker.join()

        print()

    if errors:
        raise RuntimeError(errors[0])
    
    return " ".join(transcript_parts)


# if __name__ == "__main__":
#     whistle = models.Whistle(weights="path/to/your/weights")
#     listen_until_silence(whistle)