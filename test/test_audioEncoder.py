import torch as th
import unittest
from mm_diffusion.ALEJANDROTEMP_multimodal_vae import load_pretrained_video_vae, load_encodec_audio_vae

class TestEnCodecAudioVAE(unittest.TestCase):
    def setUp(self):
        """
        Setup test case, initialize encoder and decoder.
        Here we load the EnCodec model.
        """
        self.encoder, self.decoder = load_encodec_audio_vae()

    def test_encoder_single_audio(self):
        """
        Test the encoder with a single audio sample.
        The expected shape of latents should be [1, num_quantizers, latent_steps, codes_per_step].
        """
        # Create a mock audio tensor with shape [1, T] for a single sample
        wav = th.randn(1, 16000)  # Let's assume T = 16000 for a 1-second audio sample

        latents = self.encoder(wav)  # Latent encoding for the single sample

        # Check if the latents are in the expected shape (B, Q, L, C)
        # Here, Q is the number of quantizers and L and C are the latent dimensions
        self.assertEqual(latents.dim(), 4)
        self.assertEqual(latents.shape[0], 1)  # Batch size is 1
        self.assertGreater(latents.shape[1], 0)  # At least one quantizer should exist
        self.assertGreater(latents.shape[2], 0)  # Latent steps (L) should be > 0
        self.assertGreater(latents.shape[3], 0)  # Codes per step (C) should be > 0

    def test_encoder_batch_audio(self):
        """
        Test the encoder with a batch of audio samples.
        The expected shape of latents should be [B, num_quantizers, latent_steps, codes_per_step].
        """
        # Create a mock audio tensor with shape [B, T] for a batch of samples
        wav_batch = th.randn(4, 16000)  # Batch of 4 audio samples, each of length 16000

        latents = self.encoder(wav_batch)  # Latent encoding for the batch

        # Check if the latents are in the expected shape (B, Q, L, C)
        self.assertEqual(latents.dim(), 4)
        self.assertEqual(latents.shape[0], 4)  # Batch size is 4
        self.assertGreater(latents.shape[1], 0)  # At least one quantizer should exist
        self.assertGreater(latents.shape[2], 0)  # Latent steps (L) should be > 0
        self.assertGreater(latents.shape[3], 0)  # Codes per step (C) should be > 0

    def test_decoder(self):
        """
        Test the decoder to check if it can properly decode latents.
        """
        # Create some mock latents [B, num_quantizers, latent_steps, codes_per_step]
        latents = th.randn(4, 8, 100, 256)  # Example shape [B, Q, L, C]

        decoded_audio = self.decoder(latents)  # Decode latents back to audio

        # Ensure the decoded audio has the expected shape
        self.assertEqual(decoded_audio.dim(), 3)  # [B, C, T] (Batch, Channels, Time)
        self.assertEqual(decoded_audio.shape[0], 4)  # Batch size is 4
        self.assertGreater(decoded_audio.shape[2], 0)  # Time length (T) should be > 0

if __name__ == "__main__":
    unittest.main()
