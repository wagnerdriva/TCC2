docker build --tag plate-detection .
docker tag plate-detection wagnerrua/vehicle:plate-detection
docker push wagnerrua/vehicle:plate-detection