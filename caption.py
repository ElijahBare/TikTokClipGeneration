import os
import moviepy.editor as mp
import whisper
import argparse
import warnings
import tempfile
from utils import filename, str2bool, write_srt
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.editor import ImageClip
import imgkit
import time
import shutil

#thanks to https://github.com/meshbound/Shovel/tree/63b132f11238f720fc9270964fc43e7894358186 for the caption generation :)))

def text_to_image(
       text: str,
       font_size: int = 50,
       font_name: str = "Arial",
       font_color: str = "white",
       outline_ratio: float = 0.2,
       outline_color: str = "black"
       ) -> ImageClip:

   body = f"""
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <style>
           h2 {{
               font: 800 {font_size}px {font_name};
               color: {outline_color};
               text-align: center;
               padding: 70px 0;
               inline-size: 1080px;
               overflow-wrap: break-word;
               -webkit-text-fill-color: {font_color};
               -webkit-text-stroke: {font_size * outline_ratio}px {outline_color};
           }}
       </style>
   </head>
   <body>
       <h2>{text}</h2>
   </body>
   </html>"""
   options = {
       "transparent": "",
   }
   file_path = f"output/frames/{int(time.time() * 1000)}.png"
   imgkit.from_string(string=body, output_path=file_path, options=options)
   return ImageClip(file_path)

def main():
   parser = argparse.ArgumentParser(
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument("video", nargs="+", type=str,
                       help="paths to video files to transcribe")
   parser.add_argument("--model", default="small",
                       choices=whisper.available_models(), help="name of the Whisper model to use")
   parser.add_argument("--output_dir", "-o", type=str,
                       default=".", help="directory to save the outputs")
   parser.add_argument("--output_srt", type=str2bool, default=False,
                       help="whether to output the .srt file along with the video files")
   parser.add_argument("--srt_only", type=str2bool, default=False,
                       help="only generate the .srt file and not create overlayed video")
   parser.add_argument("--verbose", type=str2bool, default=False,
                       help="whether to print out the progress and debug messages")

   parser.add_argument("--task", type=str, default="transcribe", choices=[
                       "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
   parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"],
   help="What is the origin language of the video? If unset, it is detected automatically.")

   args = parser.parse_args().__dict__
   model_name: str = args.pop("model")
   output_dir: str = args.pop("output_dir")
   output_srt: bool = args.pop("output_srt")
   srt_only: bool = args.pop("srt_only")
   language: str = args.pop("language")

   os.makedirs(output_dir, exist_ok=True)

   if model_name.endswith(".en"):
       warnings.warn(
           f"{model_name} is an English-only model, forcing English detection.")
       args["language"] = "en"
   # if translate task used and language argument is set, then use it
   elif language != "auto":
       args["language"] = language

   model = whisper.load_model(model_name)
   audios = get_audio(args.pop("video"))
   subtitles = get_subtitles(
       audios, output_srt or srt_only, output_dir, lambda audio_path: model.transcribe(audio_path, **args)
   )

   if srt_only:
       return

   for path, srt_path in subtitles.items():
       out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

       print(f"Adding subtitles to {filename(path)}...")

       video = mp.VideoFileClip(path)
       audio = mp.AudioClip
       subtitles_clip = SubtitlesClip(srt_path, lambda txt: text_to_image(txt, font_size=45, font_color='pink'))
       final = mp.CompositeVideoClip([video, subtitles_clip.set_position(("center", video.size[1]*3/5))])
       final.write_videofile(out_path,
                                   codec='libx264',
                                   audio_codec='aac',
                                   temp_audiofile='temp-audio.m4a',
                                   remove_temp=True)
       video.close()
       final.close()

def get_audio(paths):
   temp_dir = tempfile.gettempdir()

   audio_paths = {}

   for path in paths:
       print(f"Extracting audio from {filename(path)}...")
       output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

       video = mp.VideoFileClip(path)
       video.audio.write_audiofile(output_path)
       video.close()

       audio_paths[path] = output_path

   return audio_paths

def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, transcribe: callable):
   subtitles_path = {}

   for path, audio_path in audio_paths.items():
       srt_path = output_dir if output_srt else tempfile.gettempdir()
       srt_path = os.path.join(srt_path, f"{filename(path)}.srt")

       print(
           f"Generating subtitles for {filename(path)}... This might take a while."
       )

       warnings.filterwarnings("ignore")
       result = transcribe(audio_path)
       warnings.filterwarnings("default")

       with open(srt_path, "w", encoding="utf-8") as srt:
           write_srt(result["segments"], file=srt)

       subtitles_path[path] = srt_path

   return subtitles_path

if __name__ == '__main__':
   os.makedirs("./output/frames", exist_ok=True)
   main()
   shutil.rmtree("./output/frames")
