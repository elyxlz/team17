import logging
import shutil
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile
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


class STTResponse(BaseModel):
    text: str


@app.post("/speech-to-text-buf/", response_model=STTResponse)
async def stt_endpoint_buf(audio_buf: UploadFile = File(...)):
    """
    Converts an uploaded audio file to text.
    """
    logger.info("Received audio buffer for speech-to-text conversion.")

    try:
        # Read the uploaded audio buffer
        audio_data = await audio_buf.read()
        logger.info(f"Audio buffer read successfully. Size: {len(audio_data)} bytes.")

        # Create a VoiceSample from the provided buffer
        logger.info("Creating VoiceSample from the provided audio buffer.")
        sample = VoiceSample.from_prompt_and_buf(
            buf=audio_data,
            prompt="<|audio|>",
        )

        # Perform inference on the audio sample
        logger.info("Running inference on the audio sample.")
        voice_output = ultravox.infer(
            sample=sample,
        )

        logger.info("Inference completed successfully. Returning response.")
        return STTResponse(text=voice_output.text)

    except Exception as e:
        logger.error(
            f"Error occurred during speech-to-text conversion: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred during speech-to-text processing.",
        )


@app.post("/speech-to-text-file/", response_model=STTResponse)
async def stt_endpoint_file(audio_file: UploadFile = File(...)):
    """
    Converts an uploaded audio file to text.
    """
    logger.info("Received audio file for speech-to-text conversion.")

    # Copy the file to a temporary location
    temp_file_path = f"/tmp/{audio_file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(audio_file.file, temp_file)

    print(audio_file)
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
        return STTResponse(text=voice_output.text)

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
