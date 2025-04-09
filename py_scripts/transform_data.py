import os
import io
import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

def parse_example(example):
    """Parse a YouTube-8M TFRecord example."""
    context_features = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.VarLenFeature(tf.int64),
    }
    
    # Make 'audio' optional since some records don't have it
    sequence_features = {
        "rgb": tf.io.FixedLenSequenceFeature([], tf.string),
    }
    
    # Parse the example
    try:
        context, sequence = tf.io.parse_single_sequence_example(
            example,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return context, sequence
    except tf.errors.OpError:
        # Try parsing with different feature configurations
        try:
            # Try with both rgb and audio
            sequence_features = {
                "rgb": tf.io.FixedLenSequenceFeature([], tf.string),
                "audio": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
            }
            
            context, sequence = tf.io.parse_single_sequence_example(
                example,
                context_features=context_features,
                sequence_features=sequence_features
            )
            return context, sequence
        except tf.errors.OpError as e:
            # If still failing, re-raise the exception
            raise e

def decode_rgb_features(features):
    """Decode RGB features to numpy arrays."""
    decoded_frames = []
    for feature in features:
        # YouTube-8M RGB frames are typically stored as 1024-dim features
        try:
            # First try decoding as uint8 (for raw frame data)
            feature_vector = tf.io.decode_raw(feature, tf.uint8)
            feature_vector = tf.reshape(feature_vector, (-1,))
            
            # Convert to numpy
            frame = feature_vector.numpy()
            
            # Check if this looks like a valid frame (should have dimensions divisible by 3 for RGB)
            if frame.shape[0] % 3 != 0:
                raise ValueError("Feature doesn't seem to be a valid RGB frame")
            
            # For visualization, reshape to an image-like format
            # YouTube-8M typically uses 1024-dim features, which we can reshape to 32x32 RGB
            size = int(np.sqrt(frame.shape[0] // 3))
            if size * size * 3 <= frame.shape[0]:
                frame = frame[:size*size*3].reshape(size, size, 3)
                decoded_frames.append(frame)
        except (tf.errors.OpError, ValueError):
            try:
                # Try decoding as float32 (for feature vectors)
                feature_vector = tf.io.decode_raw(feature, tf.float32)
                feature_vector = tf.reshape(feature_vector, (-1,))
                
                # Convert to numpy and normalize for visualization
                frame = feature_vector.numpy()
                
                # If it's a feature vector (typically 1024-dim for YouTube-8M),
                # create a visualization by reshaping and scaling
                size = int(np.sqrt(frame.shape[0]))
                if size * size <= frame.shape[0]:
                    # Normalize to 0-255 range for visualization
                    frame = frame[:size*size]
                    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
                    frame = frame.reshape(size, size).astype(np.uint8)
                    # Convert grayscale to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    decoded_frames.append(frame)
            except:
                # Skip frames that can't be decoded
                continue
    
    return decoded_frames

def convert_tfrecord_to_mp4(tfrecord_path, output_dir):
    """Convert a TFRecord file to MP4 videos."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    videos_processed = 0
    errors = 0
    
    # Process each example
    for i, example in enumerate(tqdm(dataset, desc=f"Processing {os.path.basename(tfrecord_path)}")):
        try:
            context, sequence = parse_example(example)
            video_id = context["id"].numpy().decode('utf-8')
            
            # Decode RGB features
            rgb_features = sequence["rgb"].values
            frames = decode_rgb_features(rgb_features)
            
            if frames and len(frames) > 0:
                # Create output video file
                output_path = os.path.join(output_dir, f"{video_id}.mp4")
                
                # Get frame dimensions
                height, width, _ = frames[0].shape
                
                # Create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))
                
                # Write frames to video
                for frame in frames:
                    out.write(frame)
                
                # Release resources
                out.release()
                
                videos_processed += 1
                if videos_processed % 10 == 0:
                    print(f"Saved {videos_processed} videos so far")
        except Exception as e:
            errors += 1
            if errors < 5:  # Only show the first few errors to avoid overwhelming output
                print(f"Error processing example {i}: {e}")
            elif errors == 5:
                print("Suppressing further error messages...")
    
    print(f"Completed processing {tfrecord_path}")
    print(f"Successfully processed {videos_processed} videos with {errors} errors")

def process_directory(input_dir, output_dir):
    """Process all TFRecord files in a directory."""
    input_path = Path(input_dir)
    tfrecord_files = list(input_path.glob("*.tfrecord"))
    
    print(f"Found {len(tfrecord_files)} TFRecord files in {input_dir}")
    
    for tfrecord_file in tfrecord_files:
        convert_tfrecord_to_mp4(str(tfrecord_file), output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YouTube-8M TFRecord files to MP4 videos")
    parser.add_argument("--input", required=True, help="Input TFRecord file or directory")
    parser.add_argument("--output", required=True, help="Output directory for MP4 files")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        convert_tfrecord_to_mp4(args.input, args.output)
    
    print("Conversion complete!")