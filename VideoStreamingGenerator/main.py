# from kafka import KafkaProducer
# import os, cv2, json, base64

# def serializeImg(img):
#     # scale_percent = 50 # percent of original size
#     # width = int(img.shape[1] * scale_percent / 100)
#     # height = int(img.shape[0] * scale_percent / 100)
#     # dim = (width, height)
#     # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

#     img_bytes = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
#     return img_bytes


# def publishFrame(producer, video_path):
#     video = cv2.VideoCapture(video_path)
#     video_name = os.path.basename(video_path).split(".")[0]
#     frame_no = 1
#     success = True
#     while success:
#         success, frame = video.read()        
#         # pushing every 3rd frame
#         if frame_no % 3 == 0 and success:
#             frame_bytes = serializeImg(frame)

#             data = {
#                 "frame": frame_bytes, 
#                 "frameNumber": frame_no,
#                 "videoName": video_name
#             }

#             future = producer.send('video-stream', data)
#             result = future.get(timeout=60)

#         frame_no += 1

#     video.release()
#     print("Numero de frames: ", frame_no)
#     return
    

# producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers='192.168.200.110:9092')
# videoDir = "/home/users/datasets/filmagens-05-08-22/Westphalen-AW.mp4"

# publishFrame(producer, videoDir)


# import gi

# gi.require_version('Gst', '1.0')
# gi.require_version('GstRtspServer', '1.0')
# from gi.repository import Gst, GstRtspServer, GLib

# loop = GLib.MainLoop()
# Gst.init(None)

# class TestRtspMediaFactory(GstRtspServer.RTSPMediaFactory):
#     def __init__(self):
#         GstRtspServer.RTSPMediaFactory.__init__(self)

#     def do_create_element(self, url):
#         #set mp4 file path to filesrc's location property
#         src_demux = "filesrc location=/home/users/datasets/filmagens-05-08-22/Westphalen-AW.mp4 ! qtdemux name=demux"
#         h264_transcode = "demux.video_0"
#         #uncomment following line if video transcoding is necessary
#         #h264_transcode = "demux.video_0 ! decodebin ! queue ! x264enc"
#         pipeline = "{0} {1} ! queue ! rtph264pay name=pay0 config-interval=1 pt=96".format(src_demux, h264_transcode)
#         print ("Element created: " + pipeline)
#         return Gst.parse_launch(pipeline)

# class GstreamerRtspServer():
#     def __init__(self):
#         self.rtspServer = GstRtspServer.RTSPServer()
#         factory = TestRtspMediaFactory()
#         factory.set_shared(True)
#         mountPoints = self.rtspServer.get_mount_points()
#         mountPoints.add_factory("/stream1", factory)
#         self.rtspServer.attach(None)

# if __name__ == '__main__':
#     s = GstreamerRtspServer()
#     loop.run()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  20 02:07:13 2019
@author: prabhakar
"""
# import necessary argumnets 
import gi
import cv2
import argparse

# import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.cap = cv2.VideoCapture(opt.device_id)
        self.number_frames = 0
        self.fps = opt.fps
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(opt.image_width, opt.image_height, self.fps)
    # method to capture the video feed from the camera and push it to the
    # streaming buffer.
    def on_need_data(self, src, length):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # It is better to change the resolution of the camera 
                # instead of changing the image shape as it affects the image quality.
                frame = cv2.resize(frame, (opt.image_width, opt.image_height), \
                    interpolation = cv2.INTER_LINEAR)
                data = frame.tostring()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                                                                                       self.duration,
                                                                                       self.duration / Gst.SECOND))
                if retval != Gst.FlowReturn.OK:
                    print(retval)
    # attach the launch string to the override method
    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    
    # attaching the source element to the rtsp media
    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

# Rtsp server implementation where we attach the factory sensor with the stream uri
class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.set_service(str(opt.port))
        self.get_mount_points().add_factory(opt.stream_uri, self.factory)
        self.attach(None)

# getting the required information from the user 
parser = argparse.ArgumentParser()
parser.add_argument("--device_id", required=True, help="device id for the \
                video device or video file location")
parser.add_argument("--fps", required=True, help="fps of the camera", type = int)
parser.add_argument("--image_width", required=True, help="video frame width", type = int)
parser.add_argument("--image_height", required=True, help="video frame height", type = int)
parser.add_argument("--port", default=8554, help="port to stream video", type = int)
parser.add_argument("--stream_uri", default = "/video_stream", help="rtsp video stream uri")
opt = parser.parse_args()

try:
    opt.device_id = int(opt.device_id)
except ValueError:
    pass

# initializing the threads and running the stream on loop.
GObject.threads_init()
Gst.init(None)
server = GstServer()
loop = GObject.MainLoop()
loop.run()