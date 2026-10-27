import atexit
import io
import os
import threading
import wave
import time

import numpy as np

from flask import Flask, jsonify, request
from edge_impulse_linux.runner import ImpulseRunner


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# EDGE IMPULSE MODEL
# ============================================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelfile.eim"
)

print("======================================")
print("Loading Edge Impulse cloud model...")
print("Model:", MODEL_PATH)
print("======================================")


runner = ImpulseRunner(MODEL_PATH)

model_info = runner.init()


# ============================================================
# MODEL INFORMATION
# ============================================================

model_parameters = model_info["model_parameters"]

LABELS = model_parameters["labels"]

EXPECTED_FREQUENCY = int(
    model_parameters["frequency"]
)

EXPECTED_SAMPLES = int(
    model_parameters["input_features_count"]
)


print()
print("MODEL LOADED SUCCESSFULLY")
print("Labels             :", LABELS)
print("Sampling frequency :", EXPECTED_FREQUENCY, "Hz")
print("Samples required   :", EXPECTED_SAMPLES)
print()


# Prevent simultaneous access to EIM runner
runner_lock = threading.Lock()


# ============================================================
# ACTION FOR EACH KEYWORD
# ============================================================

ACTION_MAP = {

    "down": "Display DOWN image",

    "go": "Display GO image",

    "left": "Display LEFT image",

    "off": "Display OFF image",

    "on": "Display ON image"
}


# ============================================================
# HOME PAGE / HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "status":
            "Cloud Keyword Spotting Server Running",

        "model_location":
            "Cloud",

        "labels":
            LABELS,

        "sampling_frequency":
            EXPECTED_FREQUENCY,

        "required_samples":
            EXPECTED_SAMPLES
    })


# ============================================================
# CLOUD PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    request_start = time.perf_counter()


    # --------------------------------------------------------
    # CHECK AUDIO FILE
    # --------------------------------------------------------

    if "audio" not in request.files:

        return jsonify({

            "error":
                "No audio file received"

        }), 400


    audio_file = request.files["audio"]

    audio_bytes = audio_file.read()


    try:

        # ====================================================
        # OPEN WAV AUDIO
        # ====================================================

        with wave.open(
            io.BytesIO(audio_bytes),
            "rb"
        ) as wav_file:

            channels = wav_file.getnchannels()

            sample_width = wav_file.getsampwidth()

            sample_rate = wav_file.getframerate()

            frame_count = wav_file.getnframes()

            raw_audio = wav_file.readframes(
                frame_count
            )


        # ====================================================
        # VALIDATE AUDIO
        # ====================================================

        if channels != 1:

            return jsonify({

                "error":
                    "Expected mono audio"

            }), 400


        if sample_width != 2:

            return jsonify({

                "error":
                    "Expected 16-bit PCM audio"

            }), 400


        if sample_rate != EXPECTED_FREQUENCY:

            return jsonify({

                "error":
                    f"Expected {EXPECTED_FREQUENCY} Hz, "
                    f"received {sample_rate} Hz"

            }), 400


        # ====================================================
        # PCM -> INT16
        # ====================================================

        samples = np.frombuffer(
            raw_audio,
            dtype="<i2"
        )


        print(
            "Received samples:",
            len(samples)
        )


        if len(samples) != EXPECTED_SAMPLES:

            return jsonify({

                "error":
                    f"Expected {EXPECTED_SAMPLES} samples, "
                    f"received {len(samples)} samples"

            }), 400


        # ====================================================
        # INT16 -> FLOAT ARRAY
        # ====================================================

        features = samples.astype(
            np.float32
        ).tolist()


        # ====================================================
        # RUN MODEL IN CLOUD
        # ====================================================

        inference_start = time.perf_counter()


        with runner_lock:

            result = runner.classify(
                features
            )


        inference_end = time.perf_counter()


        # ====================================================
        # GET CLASSIFICATION
        # ====================================================

        if "classification" not in result["result"]:

            return jsonify({

                "error":
                    "Model did not return classification"

            }), 500


        classification = (
            result["result"]["classification"]
        )


        # ====================================================
        # FIVE PROBABILITY SCORES
        # ====================================================

        scores = {}

        for label in LABELS:

            scores[label] = float(
                classification[label]
            )


        # ====================================================
        # HIGHEST PROBABILITY
        # ====================================================

        detected_keyword = max(
            scores,
            key=scores.get
        )


        confidence = scores[
            detected_keyword
        ]


        # ====================================================
        # CLOUD ACTION
        # ====================================================

        action = ACTION_MAP.get(
            detected_keyword,
            "No action"
        )


        # ====================================================
        # TIMING
        # ====================================================

        edge_impulse_timing = result.get(
            "timing",
            {}
        )


        cloud_inference_ms = (
            inference_end -
            inference_start
        ) * 1000.0


        total_request_ms = (
            time.perf_counter() -
            request_start
        ) * 1000.0


        # ====================================================
        # CLOUD TERMINAL OUTPUT
        # ====================================================

        print()
        print("======================================")
        print("       CLOUD ML PREDICTION")
        print("======================================")

        for label in LABELS:

            print(
                f"{label:<10}: "
                f"{scores[label]:.5f}"
            )


        print("--------------------------------------")

        print(
            "Detected Keyword :",
            detected_keyword
        )

        print(
            "Confidence       :",
            confidence
        )

        print(
            "Cloud Action     :",
            action
        )

        print(
            "Inference Time   :",
            round(cloud_inference_ms, 2),
            "ms"
        )

        print("======================================")
        print()


        # ====================================================
        # RETURN RESULT TO WINDOWS PC
        # ====================================================

        return jsonify({

            "keyword":
                detected_keyword,

            "confidence":
                confidence,

            "scores":
                scores,

            "action":
                action,

            "edge_impulse_timing":
                edge_impulse_timing,

            "cloud_inference_ms":
                round(
                    cloud_inference_ms,
                    2
                ),

            "total_server_ms":
                round(
                    total_request_ms,
                    2
                )
        })


    except Exception as error:

        print(
            "Cloud inference error:",
            error
        )

        return jsonify({

            "error":
                str(error)

        }), 500


# ============================================================
# CLEANUP
# ============================================================

def shutdown_runner():

    try:

        runner.stop()

    except Exception:

        pass


atexit.register(
    shutdown_runner
)


# ============================================================
# LOCAL / CLOUD START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(
        host="0.0.0.0",
        port=port
    )