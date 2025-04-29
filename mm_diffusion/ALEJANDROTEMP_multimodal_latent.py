import torch as th
import torch.nn as nn
from einops import rearrange

# Import the original model components
from .multimodal_unet import MultimodalUNet
from .ALEJANDROTEMP_multimodal_vae import load_pretrained_video_vae, load_encodec_audio_vae, load_audio_encoder_simplex #load_soundstream_audio_vae #VideoEncoder, AudioEncoder, VideoDecoder, AudioDecoder




class LatentMultimodalDiffusion(nn.Module):
    """
    Latent Diffusion Model for joint audio-video generation.
    """
    def __init__(
        self,
        video_size,  # [frames, channels, height, width]
        audio_size,  # [channels, length]
        model_channels,
        video_latent_channels=4,   # Usually 4 for VAEs
        audio_latent_channels=8,   # Depends on audio VAE
        video_vae_scale_factor=8,  # How much the VAE downsamples (usually 8 for video)
        audio_vae_scale_factor=16, # How much the VAE downsamples audio
        video_vae_path=None,       # Path to pretrained video VAE 
        audio_vae_path=None,       # Path to pretrained audio VAE
        pretrained_unet_path=None, # Path to the pretrained UNet (optional)
        freeze_vaes=True,          # Whether to freeze VAE parameters
        **unet_kwargs              # Arguments for the UNet model
    ):
        super().__init__()
        
        # Calculate latent dimensions
        video_latent_size = [
            video_size[0],  # frames stay the same
            video_latent_channels,
            video_size[2] // video_vae_scale_factor,
            video_size[3] // video_vae_scale_factor
        ]
        
        audio_latent_size = [
            audio_latent_channels,
            audio_size[1] // audio_vae_scale_factor
        ]
        self.video_encoder, self.video_decoder = load_pretrained_video_vae()
        # self.audio_encoder, self.audio_decoder = load_audio_encoder_simplex()
        from music2latent import EncoderDecoder
        self.music_encoderdecoder = EncoderDecoder()
        
        # Initialize the UNet in latent space
        self.diffusion_model = MultimodalUNet(
            video_size=video_latent_size,
            audio_size=audio_latent_size,
            model_channels=model_channels,
            # video_out_channels=video_latent_channels,
            # audio_out_channels=audio_latent_channels,
            **unet_kwargs
        )
        
        # Load pretrained UNet weights if provided
        if pretrained_unet_path:
            self.diffusion_model.load_state_dict_(th.load(pretrained_unet_path))
            
        self.video_latent_channels = video_latent_channels
        # self.audio_latent_channels = audio_latent_channels

    def encode(self, video, audio):
        """
        Encode video and audio into latent space.
        """
        # Normalize inputs to [-1, 1] if needed
        # video = video * 2 - 1 if video.max() <= 1 else video
        # audio = audio * 2 - 1 if audio.max() <= 1 else audio
        video = video.half()
        # print(audio)
        # audio = audio.half()
        # Encode
        with th.no_grad():
            video_latent = self.video_encoder(video)
            audio = audio.squeeze()
            audio_latent = []
            for i in range(audio.shape[0]):
                #print(audio[i].shape)
                aux = audio[i].detach().cpu().numpy()
                audio_temp = self.music_encoderdecoder.encode(aux)
                print(audio_temp.shape)
                print(audio_temp)
                audio_temp = audio_temp.squeeze(0) 
                audio_latent.append(audio_temp)
            audio_latent = th.stack(audio_latent)
            
        return video_latent, audio_latent
    
    def decode(self, video_latent, audio_latent):
        """
        Decode latents back to pixel/sample space.
        """
        with th.no_grad():
            video = self.video_decoder(video_latent)
            audio = self.music_encoderdecoder.decode(audio)
            
            # audio = audio_latent
            # # Normalize outputs to [0, 1] if needed
            # video = (video + 1) / 2
            # audio = (audio + 1) / 2
            
        return video, audio
    
    def forward(self, video, audio, timesteps, label=None):
        """
        Forward pass for training.
        """
        # Encode inputs to latent space
        video_latent, audio_latent = self.encode(video, audio)
        print('dimensions of video and audio latent')
        print(video_latent.shape)
        print(audio_latent.shape)
        print('dimensions of video and audio')
        # Apply diffusion model in latent space
        video_latent_pred, audio_latent_pred = self.diffusion_model(
            video_latent, audio_latent, timesteps, label
        )
        # print(video_latent, audio_latent)
        return video_latent_pred, audio_latent_pred
    
    def sample(self, video_latent, audio_latent, timesteps, label=None):
        """
        Sample new latents and decode them.
        """
        # Apply diffusion model in latent space
        video_latent_pred, audio_latent_pred = self.diffusion_model(
            video_latent, audio_latent, timesteps, label
        )
        
        # Decode back to pixel/sample space
        video, audio = self.decode(video_latent_pred, audio_latent_pred)
        
        return video, audio


# # Define additional necessary components: VAE encoders and decoders
# class VideoEncoder(nn.Module):
#     """
#     Video VAE Encoder (placeholder, use a pretrained one in practice)
#     """
#     def __init__(self, in_channels, out_channels, scale_factor):
#         super().__init__()
#         # In practice, load a pretrained encoder like from Stable Video Diffusion
#         self.scale_factor = scale_factor
#         self.encoder = nn.Sequential(
#             # Downsampling layers would go here
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#         )
    
#     def forward(self, x):
#         # x shape: [B, F, C, H, W]
#         b, f, c, h, w = x.shape
#         x = rearrange(x, 'b f c h w -> (b f) c h w')
        
#         # Apply 2D encoder (frame by frame)
#         z = self.encoder(x)
        
#         # Reshape back to include frames
#         z = rearrange(z, '(b f) c h w -> b f c h w', b=b, f=f)
#         return z


# class VideoDecoder(nn.Module):
#     """
#     Video VAE Decoder (placeholder, use a pretrained one in practice)
#     """
#     def __init__(self, in_channels, out_channels, scale_factor):
#         super().__init__()
#         self.scale_factor = scale_factor
#         self.decoder = nn.Sequential(
#             # Upsampling layers would go here
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#         )
    
#     def forward(self, z):
#         # z shape: [B, F, C, H, W]
#         b, f, c, h, w = z.shape
#         z = rearrange(z, 'b f c h w -> (b f) c h w')
        
#         # Apply 2D decoder (frame by frame)
#         x = self.decoder(z)
        
#         # Reshape back to include frames
#         x = rearrange(x, '(b f) c h w -> b f c h w', b=b, f=f)
#         return x


class AudioEncoder(nn.Module):
    """
    Audio VAE Encoder (placeholder, use a pretrained one in practice)
    """
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.encoder = nn.Sequential(
            # Example simple encoder (in practice use pretrained)
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
    
    def forward(self, x):
        # x shape: [B, C, L]
        return self.encoder(x)


class AudioDecoder(nn.Module):
    """
    Audio VAE Decoder (placeholder, use a pretrained one in practice)
    """
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.decoder = nn.Sequential(
            # Example simple decoder (in practice use pretrained)
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, z):
        # z shape: [B, C, L]
        return self.decoder(z)