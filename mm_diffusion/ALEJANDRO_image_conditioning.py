from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL

def encode_image(image_tensor, linear_proj, T, A):
    """
    Encodes an image tensor using a pretrained VAE and prepares it for video and audio conditioning.

    Args:
        image_tensor (torch.Tensor): The input image tensor of shape [1, 3, H, W].
        linear_proj (torch.nn.Module): A linear projection module for audio conditioning.
        T (int): The length of the video in frames.
        A (int): The length of the audio in frames.

    Returns:
        tuple: A tuple containing:
            - z_vid_cond (torch.Tensor): The video conditioning tensor of shape [1, 4, T, H', W'].
            - z_aud_cond (torch.Tensor): The audio conditioning tensor of shape [1, C_aud, A].
    """
    # Load the pretrained VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")

    # preprocess to tensor [1,3,H,W], then:
    z_img = vae.encode(image_tensor).latent_dist.sample()  
    # import torch                                                                                                                                                                                                                                  

    # pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", vae=vae ) #, torch_dtype=torch.float16
    # pipe = pipe.to("cuda")

    # prompt = "A fantasy landscape with mountains and a river"
    # image = pipe(prompt).images[0]  # The generated image is in the first element of the list
    # image.save("fantasy_landscape.png")  # Save the image to a file
    # #         save_interval=1000,


    # Suppose video length T; audio length A
    # Video: tile along time
    z_vid_cond = z_img.unsqueeze(2).repeat(1,1,T,1,1)  # [1,4,T,H',W']

    # # Audio: global-pool → linear → repeat or upsample to [1, C_aud, A]
    # z_aud_vec  = z_img.mean(dim=[2,3])                  # [1,4]
    # z_aud_cond = linear_proj(z_aud_vec).unsqueeze(-1).repeat(1,1,A)
    return z_vid_cond #, z_aud_cond

