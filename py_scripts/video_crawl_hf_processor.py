import os
import io
import subprocess
import random
import tempfile
from huggingface_hub import HfApi
from datasets import Dataset
import yt_dlp
import shutil
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_TOKEN"))


def main():
    base_dir = './data10/full_concerts/' 
    huggingface_repo_id = os.getenv('HUGGINGFACE_REPO_ID')   # Replace with your actual repo ID
    hf_token = os.getenv("HUGGINGFACE_TOKEN") # Replace with your HF token
    
    # Create a temporary directory for storing intermediate files
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Define the path to the text file containing video IDs
    video_file = f'{base_dir}concerts.txt'

    # Read video IDs from the file
    with open(video_file, 'r') as file:
        video_ids = file.readlines()

    # Remove any trailing whitespace or newline characters from video IDs
    video_ids = [video_id.strip() for video_id in video_ids]

    # Shuffle the video IDs to ensure randomness
    random.shuffle(video_ids)

    # Split the video IDs: 90% train, 5% eval, 5% test
    train_size = int(0.9 * len(video_ids))
    eval_size = int(0.05 * len(video_ids))

    train_ids = video_ids[:train_size]
    eval_ids = video_ids[train_size:train_size + eval_size]
    test_ids = video_ids[train_size + eval_size:]

    # Process in batches of 100
    batch_size = 100
    
    # Process train, eval, and test sets
    process_batch(train_ids, "train", batch_size, temp_dir, huggingface_repo_id, hf_token)
    process_batch(eval_ids, "validation", batch_size, temp_dir, huggingface_repo_id, hf_token)
    process_batch(test_ids, "test", batch_size, temp_dir, huggingface_repo_id, hf_token)
    
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    print(f"Removed temporary directory: {temp_dir}")
    print("Processing complete.")

def process_batch(video_ids, split_type, batch_size, temp_dir, repo_id, token):
    """Process videos in batches and upload to HuggingFace."""
    
    for i in range(0, len(video_ids), batch_size):
        batch = video_ids[i:i+batch_size]
        batch_num = i//batch_size
        print(f"Processing {split_type} batch {batch_num + 1}/{len(video_ids)//batch_size + 1}")
        
        # Process batch
        batch_data = process_videos(batch, split_type, temp_dir)
        
        # Upload batch to HuggingFace
        if batch_data:
            # Use a valid split name format (use underscore instead of hyphen)
            split_name = f"{split_type}_batch{batch_num}"
            upload_to_huggingface(batch_data, split_name, repo_id, token)

def process_videos(video_ids, split_type, temp_dir):
    """Download and process a batch of videos, storing data in memory."""
    batch_data = []
    
    for video_id in video_ids:
        print(f"Processing video ID: {video_id}")
        chunks = download_and_process_video(video_id, temp_dir)
        if chunks:
            for chunk_idx, chunk_data in enumerate(chunks):
                batch_data.append({
                    "video_id": video_id,
                    "chunk_id": chunk_idx + 1,
                    "split": split_type,
                    "video_data": chunk_data
                })
    
    return batch_data

def get_video_duration(video_path):
    """Get the duration of a video using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None

def download_and_process_video(video_id, temp_dir):
    """Download video to memory and process it into chunks."""
    # Create a temporary file for initial download
    temp_video_path = os.path.join(temp_dir, f"{video_id}.mp4")
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestvideo+bestaudio',
        'merge_output_format': 'mp4',
        'outtmpl': temp_video_path,
    }
    
    try:
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://youtube.com/watch?v={video_id}'])
        
        # Get the video duration
        duration = get_video_duration(temp_video_path)
        if duration is None or duration <= 10:
            print(f"Skipping {video_id}: Video too short or duration check failed.")
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return None
        
        print(f"Processing video ID: {video_id} (Duration: {duration:.2f} seconds)")
        
        # Process video into chunks and store in memory
        chunks = []
        start_time = 10  # Start at 10 seconds
        chunk_index = 1
        
        while start_time < duration:
            chunk_temp_path = os.path.join(temp_dir, f"{video_id}_chunk{chunk_index}.mp4")
            
            # Extract chunk using ffmpeg
            subprocess.run([
                'ffmpeg', '-i', temp_video_path,
                '-ss', str(start_time),
                '-t', '34',
                '-c', 'copy',
                chunk_temp_path
            ], check=True)
            
            # Read chunk into memory
            with open(chunk_temp_path, 'rb') as f:
                chunk_data = f.read()
                chunks.append(chunk_data)
            
            # Remove the temp chunk file
            os.remove(chunk_temp_path)
            
            print(f"Processed chunk {chunk_index} for {video_id} (in memory)")
            start_time += 34
            chunk_index += 1
        
        # Remove the original video file
        os.remove(temp_video_path)
        return chunks
        
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        # Clean up any temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return None

def upload_to_huggingface(batch_data, split_name, repo_id, token):
    """Upload a batch of videos to HuggingFace datasets."""
    print(f"Uploading {split_name} with {len(batch_data)} chunks to HuggingFace")
    
    # Create dataset from batch data
    dataset_dict = {
        "video_id": [item["video_id"] for item in batch_data],
        "chunk_id": [item["chunk_id"] for item in batch_data],
        "split": [item["split"] for item in batch_data],
        "video_data": [item["video_data"] for item in batch_data]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Push to hub
    dataset.push_to_hub(
        repo_id,
        token=token,
        split=split_name,  # Using valid split name format now (with underscores)
        private=True  # Set to False if you want a public dataset
    )
    
    print(f"Successfully uploaded {split_name} to HuggingFace")

if __name__ == "__main__":
    main()