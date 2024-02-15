import time
import numpy as np
from tqdm.notebook import tqdm

model_ids = {
    "Distil-Whisper": [
        "distil-whisper/distil-large-v2",
        "distil-whisper/distil-medium.en",
        "distil-whisper/distil-small.en"
    ],
    "Whisper": [
        "openai/whisper-large-v3",
        "openai/whisper-large-v2",
        "openai/whisper-large",
        "openai/whisper-medium",
        "openai/whisper-small",
        "openai/whisper-base",
        "openai/whisper-tiny",
        "openai/whisper-medium.en",
        "openai/whisper-small.en",
        "openai/whisper-base.en",
        "openai/whisper-tiny.en",
    ]
}

'''
model_type = widgets.Dropdown(
    options=model_ids.keys(),
    value="Distil-Whisper",
    description="Model type:",
    disabled=False,
)
'''

model_type='Distil-Whisper'

'''
model_id = widgets.Dropdown(
    options=model_ids[model_type.value],
    value=model_ids[model_type.value][0],
    description="Model:",
    disabled=False,
)
'''

model_id='distil-whisper/distil-small.en'

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

#processor = AutoProcessor.from_pretrained(model_id.value)
processor = AutoProcessor.from_pretrained(model_id)

#pt_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id.value)
pt_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
pt_model.eval()

from pathlib import Path
from optimum.intel.openvino import OVModelForSpeechSeq2Seq


#model_path = Path(model_id.value.replace('/', '_'))
model_path = Path(model_id.replace('/', '_'))
ov_config = {"CACHE_DIR": ""}

if not model_path.exists():
    print ("Model path does not exist.")
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
        #model_id.value, ov_config=ov_config, export=True, compile=False, load_in_8bit=False
        model_id, ov_config=ov_config, export=True, compile=False, load_in_8bit=False
    )
    ov_model.half()
    ov_model.save_pretrained(model_path)
else:
    print ("Model path is ", model_path)
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_path, ov_config=ov_config, compile=False
    )

import openvino as ov
#import ipywidgets as widgets

core = ov.Core()

'''
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)
'''

#device: CPU, GPU, AUTO
device='AUTO'

#ov_model.to(device.value)
ov_model.to(device)
ov_model.compile()

from transformers import pipeline

ov_model.generation_config = pt_model.generation_config

pipe = pipeline(
    "automatic-speech-recognition",
    model=ov_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
)

'''
to_quantize = widgets.Checkbox(
    value=True,
    description='Quantization',
    disabled=False,
)
'''

to_quantize=True


from datasets import load_dataset

def extract_input_features(sample):
    input_features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features
    return input_features

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = dataset[0]
input_features = extract_input_features(sample)

from itertools import islice
from optimum.intel.openvino.quantization import InferRequestWrapper


def collect_calibration_dataset(ov_model: OVModelForSpeechSeq2Seq, calibration_dataset_size: int):
    # Overwrite model request properties, saving the original ones for restoring later
    original_encoder_request = ov_model.encoder.request
    original_decoder_with_past_request = ov_model.decoder_with_past.request
    encoder_calibration_data = []
    decoder_calibration_data = []
    ov_model.encoder.request = InferRequestWrapper(original_encoder_request, encoder_calibration_data)
    ov_model.decoder_with_past.request = InferRequestWrapper(original_decoder_with_past_request,
                                                             decoder_calibration_data)

    calibration_dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    for sample in tqdm(islice(calibration_dataset, calibration_dataset_size), desc="Collecting calibration data",
                       total=calibration_dataset_size):
        input_features = extract_input_features(sample)
        ov_model.generate(input_features)

    ov_model.encoder.request = original_encoder_request
    ov_model.decoder_with_past.request = original_decoder_with_past_request

    return encoder_calibration_data, decoder_calibration_data

import gc
import shutil
import nncf

CALIBRATION_DATASET_SIZE = 50
quantized_model_path = Path(f"{model_path}_quantized")


def quantize(ov_model: OVModelForSpeechSeq2Seq, calibration_dataset_size: int):
    if not quantized_model_path.exists():
        encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(
            ov_model, calibration_dataset_size
        )
        print("Quantizing encoder")
        quantized_encoder = nncf.quantize(
            ov_model.encoder.model,
            nncf.Dataset(encoder_calibration_data),
            subset_size=len(encoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.50)
        )
        ov.save_model(quantized_encoder, quantized_model_path / "openvino_encoder_model.xml")
        del quantized_encoder
        del encoder_calibration_data
        gc.collect()

        print("Quantizing decoder with past")
        quantized_decoder_with_past = nncf.quantize(
            ov_model.decoder_with_past.model,
            nncf.Dataset(decoder_calibration_data),
            subset_size=len(decoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.95)
        )
        ov.save_model(quantized_decoder_with_past, quantized_model_path / "openvino_decoder_with_past_model.xml")
        del quantized_decoder_with_past
        del decoder_calibration_data
        gc.collect()

        # Copy the config file and the first-step-decoder manually
        shutil.copy(model_path / "config.json", quantized_model_path / "config.json")
        shutil.copy(model_path / "openvino_decoder_model.xml", quantized_model_path / "openvino_decoder_model.xml")
        shutil.copy(model_path / "openvino_decoder_model.bin", quantized_model_path / "openvino_decoder_model.bin")

    quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(quantized_model_path, ov_config=ov_config, compile=False, use_cache=True)
    #quantized_ov_model.to(device.value)
    quantized_ov_model.to(device)
    quantized_ov_model.compile()
    return quantized_ov_model


ov_quantized_model = quantize(ov_model, CALIBRATION_DATASET_SIZE)

from transformers.pipelines.audio_utils import ffmpeg_read
import gradio as gr
import urllib.request

urllib.request.urlretrieve(
    url="https://huggingface.co/spaces/distil-whisper/whisper-vs-distil-whisper/resolve/main/assets/example_1.wav",
    filename="example_1.wav",
)

BATCH_SIZE = 16
MAX_AUDIO_MINS = 30  # maximum audio input in minutes


#generate_kwargs = {"language": "en", "task": "transcribe"} if not model_id.value.endswith(".en") else {}
generate_kwargs = {"language": "en", "task": "transcribe"} if not model_id.endswith(".en") else {}
ov_pipe = pipeline(
    "automatic-speech-recognition",
    model=ov_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    generate_kwargs=generate_kwargs,
)
ov_pipe_forward = ov_pipe._forward

if to_quantize:
    ov_quantized_model.generation_config = ov_model.generation_config
    ov_quantized_pipe = pipeline(
        "automatic-speech-recognition",
        model=ov_quantized_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        generate_kwargs=generate_kwargs,
    )
    ov_quantized_pipe_forward = ov_quantized_pipe._forward


def transcribe(inputs, quantized=False):
    pipe = ov_quantized_pipe if quantized else ov_pipe
    pipe_forward = ov_quantized_pipe_forward if quantized else ov_pipe_forward

    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please record or upload an audio file before submitting your request."
        )

    with open(inputs, "rb") as f:
        inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    audio_length_mins = len(inputs) / pipe.feature_extractor.sampling_rate / 60

    if audio_length_mins > MAX_AUDIO_MINS:
        raise gr.Error(
            f"To ensure fair usage of the Space, the maximum audio length permitted is {MAX_AUDIO_MINS} minutes."
            f"Got an audio of length {round(audio_length_mins, 3)} minutes."
        )

    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    def _forward_ov_time(*args, **kwargs):
        global ov_time
        start_time = time.time()
        result = pipe_forward(*args, **kwargs)
        ov_time = time.time() - start_time
        ov_time = round(ov_time, 2)
        return result

    pipe._forward = _forward_ov_time
    ov_text = pipe(inputs.copy(), batch_size=BATCH_SIZE)["text"]
    return ov_text, ov_time


with gr.Blocks() as demo:
    gr.HTML(
        """
                <div style="text-align: center; max-width: 700px; margin: 0 auto;">
                  <div
                    style="
                      display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem;
                    "
                  >
                    <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                      OpenVINO Distil-Whisper demo
                    </h1>
                  </div>
                </div>
            """
    )
    audio = gr.components.Audio(type="filepath", label="Audio input")
    with gr.Row():
        button = gr.Button("Transcribe")
        if to_quantize:
            button_q = gr.Button("Transcribe quantized")
    with gr.Row():
        infer_time = gr.components.Textbox(
            label="OpenVINO Distil-Whisper Transcription Time (s)"
        )
        if to_quantize:
            infer_time_q = gr.components.Textbox(
                label="OpenVINO Quantized Distil-Whisper Transcription Time (s)"
            )
    with gr.Row():
        transcription = gr.components.Textbox(
            label="OpenVINO Distil-Whisper Transcription", show_copy_button=True
        )
        if to_quantize:
            transcription_q = gr.components.Textbox(
                label="OpenVINO Quantized Distil-Whisper Transcription", show_copy_button=True
            )
    button.click(
        fn=transcribe,
        inputs=audio,
        outputs=[transcription, infer_time],
    )
    if to_quantize:
        button_q.click(
            fn=transcribe,
            inputs=[audio, gr.Number(value=1, visible=False)],
            outputs=[transcription_q, infer_time_q],
        )
    gr.Markdown("## Examples")
    gr.Examples(
        [["./example_1.wav"]],
        audio,
        outputs=[transcription, infer_time],
        fn=transcribe,
        cache_examples=False,
    )
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/
try:
    demo.launch(debug=True)
except Exception:
    demo.launch(share=True, debug=True)
