import ffmpeg

video_format = "flv"
server_url = "http://192.168.200.110:8556"


process = (
    ffmpeg
    .input("videos/Westphalen-OneMinute-10fps.mp4")
    .output(
        server_url, 
        codec = "copy", # use same codecs of the original video
        listen=1, # enables HTTP server
        f=video_format)
    .global_args("-re") # argument to act as a live stream
    .run()
)