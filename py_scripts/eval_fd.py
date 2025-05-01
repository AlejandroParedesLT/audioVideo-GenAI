import torch
import os
import numpy as np
from scipy.linalg import sqrtm
from pathlib import Path
from tqdm import tqdm
import torchaudio
import torchvision
from torchvision import transforms
import torch.nn.functional as F

# --- Custom Feature Extractors for Audio (VGGish) and Video (I3D) ---
# Ensure you have the necessary models for feature extraction
from torchvision.models.video import r3d_18  # Using 3D ResNet (as an I3D alternative)
from vggish_pytorch import VGGish  # Make sure VGGish is available (or another method)

# --- Directory paths ---
GEN_DIR = Path("./results")
REAL_DIR = Path("./data10/concerts_audiovideo_dataset/unittest")

# --- GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Audio Feature Extraction (VGGish or equivalent) ---
def extract_vggish_embeddings(audio_path):
    """Extract VGGish embeddings from the audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    model = VGGish().to(device)
    model.eval()
    
    # Preprocess and normalize audio
    waveform = waveform.to(device)
    embedding = model(waveform)  # Assuming VGGish returns embeddings directly
    return embedding.cpu().detach().numpy()

# --- Video Feature Extraction (I3D or 3D ResNet) ---
def extract_i3d_features(video_path):
    """Extract I3D features from the video file."""
    # Load and preprocess the video
    video_frames = extract_video_frames(video_path)
    video_tensor = torch.stack([torch.tensor(frame).to(device) for frame in video_frames]).unsqueeze(0).float()
    
    model = r3d_18(pretrained=True).to(device)
    model.eval()
    
    # Forward pass through I3D model
    with torch.no_grad():
        features = model(video_tensor)
    
    return features.cpu().detach().numpy()

def extract_video_frames(video_path):
    """Extract frames from video using OpenCV and preprocess."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))  # Resizing to fit the I3D input size
        frames.append(frame)
    cap.release()
    return frames

# --- Helper Functions to Compute Metrics ---
def compute_stats(features):
    """Compute mean and covariance of feature set."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Fréchet Distance between two distributions."""
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def compute_kld_gaussians(mu1, sigma1, mu2, sigma2):
    """Compute Kullback-Leibler Divergence between two distributions."""
    d = mu1.shape[0]
    sigma2_inv = np.linalg.inv(sigma2)
    trace_term = np.trace(sigma2_inv @ sigma1)
    mean_term = (mu2 - mu1).T @ sigma2_inv @ (mu2 - mu1)
    log_det_term = np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))
    return 0.5 * (trace_term + mean_term - d + log_det_term)

# --- Metric Calculation ---
def collect_files(base_dir, extensions):
    """Collect files from directory based on extensions."""
    return sorted([f for f in base_dir.rglob("*") if f.suffix in extensions])

def compute_fad(generated_audio, real_audio):
    """Compute Fréchet Audio Distance (FAD)."""
    gen_features = [extract_vggish_embeddings(str(f)) for f in generated_audio]
    real_features = [extract_vggish_embeddings(str(f)) for f in real_audio]
    mu_gen, sigma_gen = compute_stats(np.vstack(gen_features))
    mu_real, sigma_real = compute_stats(np.vstack(real_features))
    return compute_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)

def compute_fvd_kvd(generated_videos, real_videos):
    """Compute Fréchet Video Distance (FVD) and Kullback-Leibler Video Distance (KVD)."""
    gen_features = [extract_i3d_features(str(f)) for f in generated_videos]
    real_features = [extract_i3d_features(str(f)) for f in real_videos]
    mu_gen, sigma_gen = compute_stats(np.vstack(gen_features))
    mu_real, sigma_real = compute_stats(np.vstack(real_features))
    fvd = compute_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    kvd = compute_kld_gaussians(mu_real, sigma_real, mu_gen, sigma_gen)
    return fvd, kvd

def main():
    print("Loading files...")
    generated_audio = collect_files(GEN_DIR, {".wav"})
    real_audio = collect_files(REAL_DIR, {".wav"})
    generated_videos = collect_files(GEN_DIR, {".mp4", ".avi"})
    real_videos = collect_files(REAL_DIR, {".mp4", ".avi"})

    assert len(generated_audio) == len(real_audio), "Mismatch in audio files"
    assert len(generated_videos) == len(real_videos), "Mismatch in video files"

    print("Computing FAD...")
    fad = compute_fad(generated_audio, real_audio)
    print(f"FAD ↓: {fad:.4f}")

    print("Computing FVD and KVD...")
    fvd, kvd = compute_fvd_kvd(generated_videos, real_videos)
    print(f"FVD ↓: {fvd:.4f}")
    print(f"KVD ↓: {kvd:.4f}")

if __name__ == "__main__":
    main()
