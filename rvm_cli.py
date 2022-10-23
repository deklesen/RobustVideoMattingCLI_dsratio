import torch
import argparse
from multiprocessing import freeze_support
if __name__ == '__main__':
	freeze_support()
	model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda() # or "resnet50"
	convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

	parser = argparse.ArgumentParser()
	parser.add_argument("--input_path", type=str, required=True, help="Input Path")
	parser.add_argument("--output_alpha", type=str, required=True, help="Output Alpha Path")
	parser.add_argument("--output_composed", type=str, help="Output Composed Path")
	parser.add_argument("--output_raw_pred", type=str,  help="Output Raw Prediction Path")
	parser.add_argument("--save_frames", action='store_true',  help="Save Frames Instead of Videos")

	args = parser.parse_args()

	convert_video(
    model,                           # The loaded model, can be on any device (cpu or cuda).
    input_source=args.input_path,        # A video file or an image sequence directory.
    downsample_ratio=None,           # [Optional] If None, make downsampled max size be 512px.
    output_type='png_sequence' if args.save_frames else 'video',             # Choose "video" or "png_sequence"
    output_composition=args.output_composed,    # File path if video; directory path if png sequence.
    output_alpha=args.output_alpha,          # [Optional] Output the raw alpha prediction.
    output_foreground=args.output_raw_pred,     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
    num_workers=1,                   # Only for image sequence input. Reader threads.
    progress=True                    # Print conversion progress.
)
