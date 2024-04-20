import shutil
import openai
import uuid
import tiktoken
import json
import youtube_transcript_api
from pytube import YouTube
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import cv2
import numpy as np
import subprocess
import os
import re
import whisper
import tempfile
import time

# Set up OpenAI API client
client = openai.Client(api_key=json.load(open('config.json', 'r'))["api_key"])

# Define the chatGPT function
def chatGPT(user_query, conversation, headers, seed=None, systemPrompt=None,
            model="gpt-3.5-turbo-0125", helicone=True) -> str:
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
    # switch the model if it is beyond the token limit
    tokenizer = tiktoken.get_encoding("cl100k_base")
    if len(list(tokenizer.encode(str(conversation)))) > 000:
        model = "gpt-3.5-turbo"
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
                "length": 41.6,
                "reason": "This is funny because..."
            },
            {
                "start_timestamp": 132.4,
                "length": 31.7,
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

            find_clips_prompt = f"""This is a transcript of a video.
            Please identify the most viral sections from the whole,
            make sure they are more than 15 seconds in duration,

            Make Sure you provide extremely accurate timestamps respond only in this format
            {example_json}

            Here is the Transcription:
            {chunk_transcript}

            Look based on these things:

            1. Does it set immediate expectation? Does it say something that catches the user attention and sets an expectation for the video?
            2. Is that expectation hinted at enough throughout the video where users will stay engaged?
            3. Is it generally interesting as a topic?


            Ensure they are at least 15 seconds in ` AND NO MORE THAN 35 SECONDS. You can add them together, do not worry about the sections themselves"""

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
            end_time = start_time + clip["length"]

            if end_time - start_time < 15 or end_time - start_time > 45:
                continue

            clip_video = video.subclip(start_time, end_time)
            clip_audio = audio.subclip(start_time, end_time)
            final_clip = clip_video.set_audio(clip_audio)
            clip_name = f"{self.video_id}_{i:03d}_{start_time:.2f}_{end_time:.2f}.mp4"
            clip_path = os.path.join(self.out_folder, clip_name)
            final_clip.write_videofile(clip_path,
                                        codec='libx264',
                                        audio_codec='aac',
                                        temp_audiofile='temp-audio.m4a',
                                        remove_temp=True)
            print(f"Clip {i} saved: {clip_path}")
        print("Video clips cut and saved.")


    def crop_around_face_with_tiktok_ratio(self, input_path, output_path):
        print(f"Cropping video {input_path} around the talking person with TikTok aspect ratio...")

        cropscale = json.load(open("config.json"))["crop_scale"]
        # Calculate the desired width and height of the region
        region_width = 9 * cropscale
        region_height = 16 * cropscale

        # Load the video
        cap = cv2.VideoCapture(input_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        cv2.namedWindow("debug")

        # Load the pre-trained deep learning model for face detection
        prototxt_path = "./deploy.prototxt"
        model_path = "./model.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


        wo_audio = output_path + "no_audio.mp4"

        # Create VideoWriter object with the TikTok aspect ratio
        out = cv2.VideoWriter(wo_audio, fourcc, fps, (region_width, region_height))



        prev_largest_box = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Prepare the frame for face detection
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # Pass the blob through the network to perform face detection
            net.setInput(blob)
            detections = net.forward()

            # Initialize variables to keep track of the largest face detected
            largest_area = 0
            largest_box = None

            # Process the detections and find the bounding box of the largest face
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Adjust confidence threshold as needed
                    # Extract bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Calculate the area of the bounding box
                    area = (endX - startX) * (endY - startY)

                    # Update largest_area and largest_box if current face is larger
                    if area > largest_area:
                        largest_area = area
                        largest_box = (startX, startY, endX, endY)

            if largest_box is not None:
                # Extract coordinates of the largest face
                (startX, startY, endX, endY) = largest_box

                # Calculate the center point of the face
                center_x = (endX + startX) // 2
                center_y = (endY + startY) // 2

                # Calculate the top-left and bottom-right coordinates of the region
                left_pad = max(0, center_x - region_width // 2)
                right_pad = min(frame.shape[1], center_x + region_width // 2)
                top_pad = max(0, center_y - region_height // 2)
                bottom_pad = min(frame.shape[0], center_y + region_height // 2)

                # Extract the region from the frame
                region = frame[top_pad:bottom_pad, left_pad:right_pad]

                # Resize the region to maintain the aspect ratio
                region = cv2.resize(region, (region_width, region_height), interpolation=cv2.INTER_LINEAR)

                # Switch focus only if a different speaker's face is detected as the largest for 15 consecutive frames
                if largest_box != prev_largest_box:
                    frame_count += 1
                else:
                    frame_count = 0

                if frame_count >= 15:
                    prev_largest_box = largest_box
                    frame_count = 0

                    # Display and write the region
                    # cv2.imshow("debug", region)
                    # cv2.waitKey(1)
                out.write(region)
            else:
                # If no face is detected, crop the center 9x16 region
                center_x = frame.shape[1] // 2
                center_y = frame.shape[0] // 2
                left_pad = max(0, center_x - region_width // 2)
                right_pad = min(frame.shape[1], center_x + region_width // 2)
                top_pad = max(0, center_y - region_height // 2)
                bottom_pad = min(frame.shape[0], center_y + region_height // 2)

                # Extract the center region from the frame
                region = frame[top_pad:bottom_pad, left_pad:right_pad]

                # Resize the region to maintain the aspect ratio
                region = cv2.resize(region, (region_width, region_height), interpolation=cv2.INTER_LINEAR)

                # Write the region to the output video
                out.write(region)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        video_out = VideoFileClip(wo_audio)
        vid_uncropped = VideoFileClip(input_path)
        audio = vid_uncropped.audio

        video_out = video_out.set_audio(audio)

        video_out.write_videofile(output_path,
                                    codec='libx264',
                                    audio_codec='aac',
                                    temp_audiofile='temp-audio.m4a',
                                    remove_temp=True)

    def remove_silences(self, video_path, silence_time_threshold):
        """
        Remove silent portions from a video based on the specified threshold.

        Args:
            video_path (str): Path to the input video file.
            silence_time_threshold (float): Threshold for silence duration in seconds.

        Returns:
            A new video file with silent portions removed.
        """
        # Load the video
        video = VideoFileClip(video_path)

        # Extract audio from the video
        audio = video.audio

        # Detect non-silent chunks
        non_silent_times = audio.to_soundarray(
            lambda audio_chunk: np.any(np.abs(audio_chunk) > 0.01)
        )

        # Create a clip with only non-silent portions
        non_silent_clips = [
            clip
            for is_silent, clip in zip(non_silent_times, audio.iter_chunks())
            if not is_silent
        ]

        # Concatenate the non-silent clips
        non_silent_audio = CompositeAudioClip(non_silent_clips)

        # Create a new video with the non-silent audio
        new_video = video.set_audio(non_silent_audio)

        # Write the new video to a file
        output_path = f"output_{os.path.basename(video_path)}"
        new_video.write_videofile(output_path)

        # Close the original video and audio clips
        video.close()
        audio.close()
        new_video.close()

        return output_path

    def final_llm_filter(self, video_path):
        model = whisper.load_model("base")
        temp_dir = tempfile.gettempdir()

        output_path = os.path.join(temp_dir, f"{(video_path)}.wav")

        video = VideoFileClip(self.out_folder + "/" + video_path)
        video.audio.write_audiofile(output_path)
        video.close()


        filter_prompt = f"""
            You are a final filter for a video channel.
            You Write out an explaination given a transcript
            and decide if it will go viral or not based on 3 factors

            1. Does it set immediate expectation OR Does it say something that catches the user attention and then sets an expectation for the video?
            2. Is that expectation hinted at enough throughout the video where users will stay engaged?
            3. Is it generally interesting as a topic?


            Here is the transcript:

            {model.transcribe(output_path)['text']}


            When you complete your response type the final answer: [Yes] or [No]. THIS IS REQUIRED


        """

        out = chatGPT(filter_prompt, None, None)
        indexStart = out.find("[")
        indexEnd = out.find("]")

        answer = out[indexStart+1:indexEnd]

        print(out)
        print(answer)
        if answer == "No":
            os.remove(self.out_folder + "/" + video_path)



if __name__ == "__main__":
    out_folder = "./output"
    os.makedirs(out_folder, exist_ok=True)

    video_gen = VideoGen("0duRCl9GQNw", out_folder)
    video_gen.download_video()
    video_gen.get_transcript()
    clips = video_gen.generate_clips()
    print("Found clips:")
    print(json.dumps(clips, indent=2))
    video_gen.cut_videos(clips)

    for file in os.listdir("./output/"):
        if "ready_for_final" not in file and not file.startswith(".") and not os.path.isdir("./output/" + file) and "no_audio" not in file:
            print(file)
            try:
                video_gen.crop_around_face_with_tiktok_ratio("./output/" + file, "./output/" + file + "_ready_for_final.mp4")
                # Remove silences from the cropped video
                # video_gen.remove_silences("./output/" + file + "_cropped.mp4", silence_time_threshold=0.3)
            except:
                print("cropping failed...")

    # for file in os.listdir("./output/"):
    #     if "cropped" in file and "no_audio" not in file and not file.startswith("."):
    #         video_gen.final_llm_filter(file)
    #         pass

    for file in os.listdir("./output/"):
        if "ready_for_final" in file and not file.startswith("."):
            captioned_file = os.path.splitext(file)[0] + "_captioned.mp4"
            os.system(f"python3.11 caption.py output/{file} --model base --output_dir output_final")

    print("Process completed.")
