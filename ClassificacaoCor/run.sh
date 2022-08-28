docker build --tag color-classification .
docker tag color-classification wagnerrua/vehicle:color-classification
docker push wagnerrua/vehicle:color-classification