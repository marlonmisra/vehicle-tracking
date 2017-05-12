from process_frame import *
from moviepy.editor import *
from IPython.display import HTML
from process_frame import *

def process_video(input_path, output_path):
	input_file = VideoFileClip(input_path)
	output_clip = input_file.fl_image(process_frame)
	output_clip.write_videofile(output_path, audio=False)

process_video("test_video.mp4", "video_annotated.mp4")
