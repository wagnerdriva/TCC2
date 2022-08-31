docker build --tag vemos-backend .
docker tag vemos-backend wagnerrua/vehicle:vemos-backend
docker push wagnerrua/vehicle:vemos-backend