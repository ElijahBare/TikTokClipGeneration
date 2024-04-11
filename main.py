import openai
import uuid
import tiktoken
import json
import youtube_transcript_api
from pytube import YouTube
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import cv2
import numpy as np
import subprocess
import os
import re

# Set up OpenAI API client
client = openai.Client(api_key = json.load(open('config.json', 'r'))["api_key"])

# Define the chatGPT function
def chatGPT(user_query, conversation, headers, seed=None, systemPrompt=None,
            model="gpt-3.5-turbo", helicone=True) -> str:
    if conversation is None and systemPrompt is not None:
        conversation = [{"role": "system", "content": systemPrompt}]
    if conversation is None:
        conversation = []
    if user_query is None:
        user_query = ""
    if (
        systemPrompt is not None
        and len(conversation) > 0
        and conversation[0]["role"] != "system"
    ):
        conversation.insert(0, {"role": "system", "content": systemPrompt})
    conversation.append({"role": "user", "content": user_query})
    heliconeId = str(uuid.uuid4())
    # switch the model if it is beyond the token limit
    tokenizer = tiktoken.get_encoding("cl100k_base")
    if len(list(tokenizer.encode(str(conversation)))) > 000:
        model = "gpt-3-turbo"
    response = client.chat.completions.create(
        model=model,
        messages=conversation,
        temperature=0,
        seed=seed,
    )
    conversation.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )
    return response.choices[0].message.content

# Define the video ID
video_id = "9IQ_ldV9z_A"

class VideoGen:
    def __init__(self, video_id, out_folder) -> None:
        self.video_id = video_id
        self.out_folder = out_folder
        self.video_path = None
        self.transcript = None

    def download_video(self):
        print("Downloading video...")
        yt = YouTube(f"https://www.youtube.com/watch?v={self.video_id}")
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
        if stream is None:
            print("Error: No video stream found.")
            return

        try:
            self.video_path = stream.download(output_path=f"{self.out_folder}/original/")
        except Exception as e:
            print(f"Error downloading video: {e}")
            return

        if self.video_path is None:
            print("Error: Video download failed.")
        else:
            print("Video downloaded.")

    def get_transcript(self):
        print("Retrieving transcript...")
        self.transcript = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(self.video_id)
        print("Transcript retrieved.")
        print(self.transcript)

    def generate_clips(self):
        example_json = """
        [
            {
                "start_timestamp": 132.4,
                "end_timestamp": 168.9,
                "reason": "This is funny because..."
            },
            {
                "start_timestamp": 132.4,
                "end_timestamp": 168.9,
                "reason": "this is fascinating because..."
            }
        ]
        """

        video = VideoFileClip(self.video_path)
        video_duration = video.duration

        # Split the video into chunks for processing
        chunk_size = 60*15  # Split the video into 5-minute chunks
        num_chunks = int(video_duration // chunk_size) + (video_duration % chunk_size > 0)

        all_clips = []
        for i in range(num_chunks):
            start_time = i * chunk_size
            end_time = min((i + 1) * chunk_size, video_duration)

            print(f"Processing chunk {i + 1} of {num_chunks} ({start_time:.2f} - {end_time:.2f} seconds)")

            chunk_transcript = [
                caption
                for caption in self.transcript
                if start_time <= caption["start"] <= end_time
            ]

            find_clips_prompt = f"""
            From the transcript I give you, find 30 second to 60 second interesting/funny clips and output the timestamps like this JSON format:

            {example_json}

            Here is the transcript:

            {str(chunk_transcript)}


            BE SURE THEY ARE FUNNY ENTERTAINING RELATABLE AND GIVE THE FULL CONTEXT
            BE SURE THEY DO NOT CUT MID SENTENCE OR THOUGHT OR DRAG ON FOR TOO LONG

            15 seconds MINIMUM aim for 30

            make it perfect json with NO EXPLANATION and NO MD FORMATTING
            """

            clips_json = chatGPT(find_clips_prompt, None, None)

            print(clips_json)
            clips = json.loads(clips_json)

            all_clips.extend(clips)

        return all_clips

    def cut_videos(self, clips):
        print("Cutting video clips...")
        video = VideoFileClip(self.video_path)
        audio = AudioFileClip(self.video_path)
        for i, clip in enumerate(clips, start=1):
            start_time = clip["start_timestamp"]
            end_time = clip["end_timestamp"]

            if end_time - start_time < 15 or end_time - start_time > 45:
                continue

            clip_video = video.subclip(start_time, end_time)
            clip_audio = audio.subclip(start_time, end_time)
            final_clip = clip_video.set_audio(clip_audio)
            clip_path = os.path.join(self.out_folder, f"clip_{i}_{start_time}_{end_time}.mp4")
            final_clip.write_videofile(clip_path,
                                        codec='libx264',
                                        audio_codec='aac',
                                        temp_audiofile='temp-audio.m4a',
                                        remove_temp=True)
            print(f"Clip {i} saved: {clip_path}")
        print("Video clips cut and saved.")


    def crop_to_iphone_aspect_ratio(self):
        iphone_aspect_ratio = 9.0/16

        for input_path in os.listdir("./output/"):
            if os.path.isdir("./output/" + input_path) or input_path.startswith("."):
                continue

            input_path = "./output/" + input_path
            output_path = input_path + "_cropped.mp4"

            print(f"Cropping video {input_path} to iPhone aspect ratio...")

            # Get the input video resolution
            cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {input_path}"
            resolution = subprocess.check_output(cmd, shell=True).decode().strip()
            width, height = [int(x) for x in resolution.split('x')]

            # Calculate the crop dimensions
            original_aspect_ratio = width / float(height)
            if original_aspect_ratio > iphone_aspect_ratio:
                new_height = height
                new_width = int(new_height * iphone_aspect_ratio)
            else:
                new_width = width
                new_height = int(new_width / iphone_aspect_ratio)

            crop_dimensions = f"{new_width}:{new_height}"

            command = f"ffmpeg -i {input_path} -filter:v \"crop={crop_dimensions}\" -c:a copy {output_path}"
            subprocess.call(command, shell=True)

            print(f"Cropped video saved: {output_path}")

if __name__ == "__main__":
    out_folder = "output"
    os.makedirs(out_folder, exist_ok=True)

    video_gen = VideoGen(video_id, out_folder)
    video_gen.download_video()
    video_gen.get_transcript()
    clips = video_gen.generate_clips()
    print("Found clips:")
    print(json.dumps(clips, indent=2))
    video_gen.cut_videos(clips)
    video_gen.crop_to_iphone_aspect_ratio()

    for file in os.listdir("./output/"):
        if "cropped" in file:
            os.system(f"python3 caption.py output/{file} --model base --output_dir output_final")

    print("Process completed.")
