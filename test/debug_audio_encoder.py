from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
from dotenv import load_dotenv
import os
load_dotenv()
# dummy dataset, however you can swap this with an dataset on the 🤗 hub or bring your own
librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", token=os.getenv("HUGGINGFACE_TOKEN"))

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz", token=os.getenv("HUGGINGFACE_TOKEN"))
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz", token=os.getenv("HUGGINGFACE_TOKEN"))

# cast the audio data to the correct sampling rate for the model
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[0]["audio"]["array"]

# Select a batch of samples (e.g., first 4 samples)
batch = librispeech_dummy[:4]  # You can increase the batch size as needed
print(audio_sample.shape)
# Pre-process the inputs for the batch
inputs = processor(
    raw_audio=audio_sample, #[sample[i]["audio"]["array"] for i in range(4)],
    sampling_rate=processor.sampling_rate,
    return_tensors="pt"
)


import torch
# Assuming 'model' is your PyTorch model
device = next(model.parameters()).device
print(device)
# Assuming 'model' is your PyTorch model
device = (processor).device
print(device)


# explicitly encode then decode the audio inputs
encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]

# or the equivalent with a forward pass
audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values

# you can also extract the discrete codebook representation for LM tasks
# output: concatenated tensor of all the representations
audio_codes = model(inputs["input_values"], inputs["padding_mask"]).audio_codes