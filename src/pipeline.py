from moviepy.editor import *
from IPython.display import HTML
from predict_model import *

def process_video(input_path, output_path):
	input_file = VideoFileClip(input_path)
	output_clip = input_file.fl_image(process_frame)
	output_clip.write_videofile(output_path, audio=False)

process_video("../data/full_size/test_videos/test_video_1.mp4", "../results/test_videos/video_annotated_1_annotated.mp4")
