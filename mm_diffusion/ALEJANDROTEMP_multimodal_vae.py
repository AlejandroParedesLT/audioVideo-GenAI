import torch as th
import torch.nn as nn
from einops import rearrange
# Import environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Option 1: For Stable Video Diffusion VAE
def load_pretrained_video_vae(vae_path=None):
    """
    Load the Stable Video Diffusion VAE.
    """
    from diffusers import AutoencoderKL
    
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=th.float16,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )
    
    class VideoVAEEncoder(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, x):
            # x: [B, F, C, H, W]
            B, F, C, H, W = x.shape
            x = rearrange(x, 'b f c h w -> (b f) c h w')

            with th.no_grad():
                encode_output = self.vae.encode(x)
                latent_dist = encode_output.latent_dist
                latents = latent_dist.sample() * 0.18215  # SD scaling factor

            latents = rearrange(latents, '(b f) c h w -> b f c h w', b=B, f=F)
            return latents
    
    class VideoVAEDecoder(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            
        def forward(self, z):
            # z: [B, F, C, H, W]
            B, F, C, H, W = z.shape
            z = rearrange(z, 'b f c h w -> (b f) c h w')
            
            # Scale latents
            z = z / 0.18215
            
            # Decode
            with th.no_grad():
                x = self.vae.decode(z).sample
            
            # Reshape back to include frames
            x = rearrange(x, '(b f) c h w -> b f c h w', b=B, f=F)
            print(f"Encoding batch: {x}")
            return x
    
    encoder = VideoVAEEncoder(vae)
    decoder = VideoVAEDecoder(vae)
    
    return encoder, decoder


# Option 2: For VQGAN from Taming Transformers
def load_vqgan_video_vae(vqgan_path=None):
    """
    Load a pretrained VQGAN for video frames.
    """
    import sys
    sys.path.append("taming-transformers")  # Adjust as needed
    from taming.models.vqgan import VQModel
    
    vqgan = VQModel.load_from_checkpoint(vqgan_path)
    vqgan.eval()
    
    class VQGANVideoEncoder(nn.Module):
        def __init__(self, vqgan):
            super().__init__()
            self.vqgan = vqgan
            
        def forward(self, x):
            # x: [B, F, C, H, W]
            B, F, C, H, W = x.shape
            x = rearrange(x, 'b f c h w -> (b f) c h w')
            
            # VQGAN encode (this uses VQ so it's different from VAE)
            with th.no_grad():
                z = self.vqgan.encode(x)[0]  # Get latent without quantization
            
            # Reshape back to include frames
            z = rearrange(z, '(b f) c h w -> b f c h w', b=B, f=F)
            return z
    
    class VQGANVideoDecoder(nn.Module):
        def __init__(self, vqgan):
            super().__init__()
            self.vqgan = vqgan
            
        def forward(self, z):
            # z: [B, F, C, H, W]
            B, F, C, H, W = z.shape
            z = rearrange(z, 'b f c h w -> (b f) c h w')
            
            # VQGAN decode (need to handle the VQ part)
            with th.no_grad():
                # This is what differs from a standard VAE - need VQ lookup
                z_q = self.vqgan.quantize.get_codebook_entry(
                    z.flatten(2).transpose(1, 2), 
                    shape=(z.shape[0], z.shape[2], z.shape[3])
                )
                x = self.vqgan.decode(z_q)
            
            # Reshape back to include frames
            x = rearrange(x, '(b f) c h w -> b f c h w', b=B, f=F)
            return x
    
    encoder = VQGANVideoEncoder(vqgan)
    decoder = VQGANVideoDecoder(vqgan)
    
    return encoder, decoder


import torch as th
import torch.nn as nn

# Option 1: EnCodec from Meta
def load_encodec_audio_vae(model_path=None):
    """
    Load Meta's EnCodec model for audio.
    """

    if model_path:
        import torchaudio.models as models
        model = models.encodec.EnCodec.encodec_model_24khz()
        model.load_state_dict(th.load(model_path))
    else:       
        from transformers import AutoProcessor, EncodecModel 
        model_id = "facebook/encodec_24khz"
        model = EncodecModel.from_pretrained(model_id, token=os.getenv("HUGGINGFACE_TOKEN"))  # Use your Hugging Face token)
        processor = AutoProcessor.from_pretrained(model_id, token=os.getenv("HUGGINGFACE_TOKEN"))
    
    model.eval()
    class EnCodecEncoder(nn.Module):
        def __init__(self, model, processor):
            super().__init__()
            self.model = model
            self.processor = processor
            
        def forward(self, wav):
            # wav: [B, 1, T] or [B, T]
            with th.no_grad():
                device = next(self.model.parameters()).device  # Get device of the model
                ##################################
                # Process each sample in the batch individually
                batch_size = wav.shape[0]
                all_latents = []
                for i in range(batch_size):
                    sample = wav[i:i+1]  # Keep batch dimension for consistency
                    print(f"Sample shape: {sample.shape}")
                    # Convert [1, 1, T] to [1, T] as processor expects
                    if sample.dim() == 3 and sample.shape[0] == 1 and sample.shape[1] == 1:
                        sample = sample.squeeze(0).squeeze(0)  # Convert from [1, T] to [T]
                    # sample = sample.squeeze(0)
                    # sample = sample.squeeze(0)
                    print(f"Sample shape before processor: {sample.shape}")
                    
                    # Process the audio with the processor
                    inputs = self.processor(sample.detach().cpu(), sampling_rate=24_000, return_tensors="pt")
                    gpu_inputs = { k: v.to(device) for k,v in inputs.items() }
                    # encoded = self.model.encode(**inputs)
                    encoded = self.model.encode(**gpu_inputs)
                    #audio_values = self.model.decode(encoded.audio_codes, encoded.audio_scales, inputs["padding_mask"])[0]
                    # Access the codebooks from the model's quantizer
                    print(encoded.last_hidden_state)
                    continuous_latents = []
                    print(dir(encoded))
                    print(dir(self.model.quantizer))
                    for q_idx, codes in enumerate(encoded.audio_codes):
                        # Get the codebook for this quantizer
                        # Note: The exact implementation depends on the model structure
                        # Direct access to the codebook embeddings
                        codebook = self.model.quantizer.codebooks[q_idx]
                        
                        # Map indices to embeddings
                        embeddings = codebook[codes]  # This should retrieve the embeddings for the given indices
                        continuous_latents.append(embeddings)
                    
                    # Stack across quantizer dimension
                    stacked_latents = th.stack(continuous_latents, dim=1)
                    all_latents.append(stacked_latents)

                    # Stack across quantizer dimension
                    # Result: [1, num_quantizers, T, D]
                    stacked_latents = th.stack(continuous_latents, dim=1)
                    print(stacked_latents)
                    all_latents.append(stacked_latents)
                    print(all_latents)
                    # audio_scales = encoded.audio_scales
                    # audio_codes = encoded.audio_codes 
                    # print(f"Audio Values: {audio_scales}")
                    # print(f"Audio Codes: {audio_codes}")                    
                    # # Stack them into a single Tensor: [1, Q, T', C]
                    # latent = th.stack(audio_codes, dim=1)
                    # all_latents.append(latent)
                    
                # Concatenate all processed samples
            # Combine all batches
            latents = th.cat(all_latents, dim=0)
            return latents
                # # batch_size = wav.size(0)
                # wav = wav.squeeze(1)  # Now wav is [B, T] because the processor expects that
                # print(f"Processed shape: {wav.shape}")
                # # Process the audio with the processor
                # inputs = self.processor(wav, sampling_rate=24_000, return_tensors="pt")
                # # inputs = self.processor(wav.detach().cpu(), sampling_rate=24_000, return_tensors="pt")
                # encoded = self.model.encode(**inputs)
                # # THIS is the quantized latent: a list of codebook outputs
                # codes_list = encoded.quantized   # List[Tensor] length = num_quantizers
                # # stack them into a single Tensor: [B, Q, T', C]
                # latents = th.stack(codes_list, dim=1)
            # return latents
    
    class EnCodecDecoder(nn.Module):
        def __init__(self, model, processor):
            super().__init__()
            self.model = model
            self.processor=processor
            
        def forward(self, z):
            # z: [B, C, L]
            with th.no_grad():
                # We need to convert back to EnCodec's format
                # This is simplified - would need correct reshaping in practice
                codes = self.model.quantizer(z)
                decoded = self.model.decode([codes])
            
            return decoded
    
    encoder = EnCodecEncoder(model, processor)
    decoder = EnCodecDecoder(model, processor)
    
    return encoder, decoder


def load_audio_encoder_simplex(model_path=None):
    from music2latent import EncoderDecoder
    encdec = EncoderDecoder()
    return encdec

# Option 2: SoundStream from Google (via HF)
def load_soundstream_audio_vae():
    """
    Load Google's SoundStream model.
    """
    from transformers import AutoProcessor, SoundStreamModel
    
    model = SoundStreamModel.from_pretrained("google/soundstream")
    processor = AutoProcessor.from_pretrained("google/soundstream")
    model.eval()
    
    class SoundStreamEncoder(nn.Module):
        def __init__(self, model, processor):
            super().__init__()
            self.model = model
            self.processor = processor
            
        def forward(self, x):
            # x: [B, C, L]
            with th.no_grad():
                # Process the audio
                inputs = self.processor(x, sampling_rate=16000, return_tensors="pt")
                encoded = self.model.encode(inputs["input_values"])
                print(encoded)
                # Get the continuous latents
                latents = encoded.last_hidden_state
                print(latents)
                # Reshape as needed
                latents = latents.view(latents.size(0), -1)
            return latents
            
    class SoundStreamDecoder(nn.Module):
        def __init__(self, model, processor):
            super().__init__()
            self.model = model
            self.processor = processor
            
        def forward(self, z):
            # z: [B, C, L]
            with th.no_grad():
                # Decode the latents
                decoded = self.model.decode(z)
                # Reshape as needed
                decoded = decoded.view(decoded.size(0), -1)
            return decoded
    encoder = SoundStreamEncoder(model, processor)
    decoder = SoundStreamDecoder(model, processor)
    return encoder, decoder