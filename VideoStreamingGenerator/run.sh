docker build --no-cache --tag video-streaming-generator .
docker tag video-streaming-generator wagnerrua/vehicle:video-streaming-generator
docker push wagnerrua/vehicle:video-streaming-generator