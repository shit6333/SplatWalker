import cv2
import os
from glob import glob

def images_to_video(image_folder, output_path, fps=30):
    # Get all image paths and sort them by filename
    image_paths = sorted(glob(os.path.join(image_folder, '*.png')))

    if not image_paths:
        raise ValueError("No PNG images found in the specified folder.")

    # Read the first image to get the frame size
    first_frame = cv2.imread(image_paths[0])
    height, width, _ = first_frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: could not read image {img_path}. Skipping.")
            continue
        out.write(frame)  # Write frame to video

    out.release()
    print(f"Video saved to: {output_path}")

# Example usage:
if __name__ == "__main__":
    images_to_video('/mnt/HDD3/miayan/omega/RL/gaussian-splatting/demo_test_frames', './video/checkpoints_room_step15_aes3_var_penalty_endbouns/room_video1.mp4', fps=2)
