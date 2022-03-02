docker build --no-cache --tag events-persistence .
docker tag events-persistence wagnerrua/vehicle:events-persistence
docker push wagnerrua/vehicle:events-persistence