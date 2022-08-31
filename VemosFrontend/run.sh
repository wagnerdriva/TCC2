docker build --tag vemos-frontend .
docker tag vemos-frontend wagnerrua/vehicle:vemos-frontend
docker push wagnerrua/vehicle:vemos-frontend
kubectl replace --force -f deployment.yaml