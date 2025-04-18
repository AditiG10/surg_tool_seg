import os
import cv2


def extract_frames(video_dir: str, target_fps: int = 1) -> None:
    """
    Extract frames from video_left.avi inside each video_* folder in the given directory.

    Args:
        video_dir (str): Path to the parent directory containing video folders (e.g., data/train or data/test).
        target_fps (int): Number of frames to extract per second from the video. Default is 1.
    """
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Directory not found: {video_dir}")

    folders = [
        f for f in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, f))
    ]

    if not folders:
        print(f"No folders found in {video_dir}.")
        return

    print(f"\nProcessing {len(folders)} video folders in '{video_dir}'...\n")

    print("\nFound folders:")
    for f in os.listdir("data/test"):
        print("-", f)

    for folder in sorted(folders):
        folder_path = os.path.join(video_dir, folder)
        video_path = os.path.join(folder_path, "video_left.avi")
        output_dir = os.path.join(folder_path, "frames")

        if not os.path.exists(video_path):
            print(f"[skip] {folder}: video_left.avi not found.")
            continue

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if not cap.isOpened() or video_fps == 0:
            print(f"[error] {folder}: Unable to read video.")
            continue

        frame_interval = int(video_fps // target_fps)
        frame_count, saved_count = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_name = f"{frame_count:09d}.png"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"[done] {folder}: {saved_count} frames saved to 'frames/'")


if __name__ == "__main__":
    extract_frames("data/train")
    extract_frames("data/test")
