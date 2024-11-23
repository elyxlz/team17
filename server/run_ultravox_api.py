import logging
import uuid
from fastapi.middleware.cors import CORSMiddleware
import shutil
from fastapi import FastAPI, File, UploadFile
from ultravox.data.data_sample import VoiceSample
from ultravox.inference.ultravox_infer import UltravoxInference

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text-to-Speech & Speech-to-Text API", version="1.0")

logger.info("Initializing Ultravox Inference...")
ultravox = UltravoxInference(
    model_path="./submodules/ultravox/ultravox-v0_3-llama-3_2-1b",
    conversation_mode=True,
    device="mps",
    data_type="float16",
)
logger.info("Ultravox Inference initialized successfully.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/speech-to-text-file/")
async def stt_endpoint_file(audio_file: UploadFile = File(...)):
    """
    Converts an uploaded audio file to text.
    """
    logger.info("Received audio file for speech-to-text conversion.")

    uuid_ = str(uuid.uuid4())
    # Copy the file to a temporary location
    temp_file_path = f"/tmp/{uuid_}.wav"
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(audio_file.file, temp_file)
        print(f"Copied file to {temp_file_path}")

    try:
        logger.info("Creating VoiceSample from the provided audio buffer.")
        sample = VoiceSample.from_prompt_and_file(
            path=temp_file_path,
            prompt="<|audio|>",
        )

        logger.info("Running inference on the audio sample.")
        voice_output = ultravox.infer(
            sample=sample,
        )

        logger.info("Inference completed successfully. Returning response.")
        logger.info(f"Response: {voice_output.text}")
        return voice_output.text

    except Exception as e:
        logger.error(
            f"Error occurred during speech-to-text conversion: {e}", exc_info=True
        )
        raise


@app.post("/reset-ultravox/")
def reset_ultravox():
    global ultravox
    logger.info("Resetting Ultravox Inference...")
    ultravox = UltravoxInference(
        model_path="./submodules/ultravox/ultravox-v0_3-llama-3_2-1b",
        conversation_mode=True,
        device="mps",
        data_type="float16",
    )
    logger.info("Ultravox Inference has been reset.")
