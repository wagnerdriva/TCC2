docker build --no-cache --tag tracking .
docker tag tracking wagnerrua/vehicle:tracking
docker push wagnerrua/vehicle:tracking