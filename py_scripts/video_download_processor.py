import os
import subprocess
import random

def main():
    i = 0
    base_dir = './data10/full_concerts/'

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

    # Create directories for train, eval, and test
    os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'eval'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)

    # Function to get video duration using ffprobe
    def get_video_duration(video_path):
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

    # Define a function to download and process videos
    def download_and_process_video(video_id, category_dir):
        print(f"Downloading video ID: {video_id}")
        
        # Download the video using yt-dlp
        video_path = f"{category_dir}/{video_id}.mp4"
        os.system(f'yt-dlp -f bestvideo+bestaudio --merge-output-format mp4 -o "{video_path}" "http://youtube.com/watch?v={video_id}"')

        if not os.path.exists(video_path):
            print(f"Video ID: {video_id} was not downloaded successfully.")
            return

        # Get the total video duration
        duration = get_video_duration(video_path)
        if duration is None or duration <= 10:
            print(f"Skipping {video_id}: Video too short.")
            return

        print(f"Processing video ID: {video_id} (Duration: {duration:.2f} seconds)")

        # Start cropping at 10 seconds and extract 34-second chunks
        start_time = 10
        chunk_index = 1

        while start_time < duration:
            end_time = min(start_time + 34, duration)
            chunk_output = f"{category_dir}/{video_id}_chunk{chunk_index}.mp4"

            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-t', '34',
                '-c', 'copy',
                chunk_output
            ])

            print(f"Saved chunk {chunk_index} for {video_id}: {chunk_output}")
            start_time += 34
            chunk_index += 1

        # Remove the original full video after processing
        os.remove(video_path)
        print(f"Deleted original video {video_id}.mp4")

    # Download and process videos for each category (train, eval, test)
    for video_id in train_ids:
        download_and_process_video(video_id, os.path.join(base_dir, 'train'))

    for video_id in eval_ids:
        download_and_process_video(video_id, os.path.join(base_dir, 'eval'))

    for video_id in test_ids:
        download_and_process_video(video_id, os.path.join(base_dir, 'test'))

    print("Processing complete.")
    # if i == 1000:
    #     print("No videos to process.")

if __name__ == "__main__":
    main()