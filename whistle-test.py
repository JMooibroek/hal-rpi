import queue
import threading

import numpy
import sounddevice
import soxr

from cactus import Whistle
# import whistle as models
from cactus.agent.whistle import SAMPLE_RATE


def listen_forever(whistle, language=None, keywords=None):
    audio_queue = queue.Queue()
    errors = []

    def audio_chunks():
        block = []
        held = 0

        while True:
            data = audio_queue.get()

            if data is None:
                if block:
                    yield numpy.concatenate(block)
                return

            block.append(data)
            held += len(data)

            if held >= SAMPLE_RATE:
                yield numpy.concatenate(block)
                block = []
                held = 0

    def transcribe():
        try:
            for step in whistle.stream(
                audio_chunks(),
                language=language,
                keywords=keywords,
            ):
                text = step["text"]
                if text:
                    print((" " if transcribe.has_text else "") + text, end="", flush=True)
                    transcribe.has_text = True
        except (RuntimeError, OSError) as error:
            errors.append(error)

    transcribe.has_text = False

    try:
        rate = int(sounddevice.query_devices(kind="input")["default_samplerate"])
        resampler = (
            soxr.ResampleStream(
                rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ"
            )
            if rate != SAMPLE_RATE
            else None
        )

        def capture(data, *_):
            mono = data[:, 0]
            audio_queue.put(
                resampler.resample_chunk(mono) if resampler else mono.copy()
            )

        worker = threading.Thread(target=transcribe, daemon=True)

        with sounddevice.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            callback=capture,
        ):
            print("Listening — press Ctrl+C to stop.")
            worker.start()

            try:
                while worker.is_alive():
                    worker.join(timeout=0.2)
            except KeyboardInterrupt:
                pass

        if resampler:
            audio_queue.put(
                resampler.resample_chunk(
                    numpy.zeros(0, dtype=numpy.float32), last=True
                )
            )
        audio_queue.put(None)
        worker.join()
        print()

        if errors:
            raise RuntimeError(errors[0])

    except sounddevice.PortAudioError as error:
        raise RuntimeError(f"No microphone: {error}") from None


if __name__ == "__main__":
    whistle = Whistle() # weights="path/to/your/weights"
    listen_forever(whistle, keywords=["Jamaro", "Mooibroek", "Safran", "Petra", "Barry", "HAL"])