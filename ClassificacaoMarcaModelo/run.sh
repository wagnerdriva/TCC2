docker build --no-cache --tag brand-classification .
docker tag brand-classification wagnerrua/vehicle:brand-classification
docker push wagnerrua/vehicle:brand-classification