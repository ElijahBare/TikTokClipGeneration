import os
from typing import Iterator, TextIO


def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())}, got {string}")


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"



def write_srt(segments, file):
    segment_num = 1
    for segment in segments:
        text = segment["text"].strip()
        start = segment["start"]
        end = segment["end"]

        words = text.split()
        word_start = start
        for word in words:
            word_end = word_start + len(word) / len(text) * (end - start)
            file.write(f"{segment_num}\n")
            file.write(f"{format_time(word_start)} --> {format_time(word_end)}\n")
            file.write(f"{word}\n\n")
            word_start = word_end
            segment_num += 1

def format_time(time):
    hours = int(time / 3600)
    minutes = int((time % 3600) / 60)
    seconds = int(time % 60)
    milliseconds = int((time % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]
