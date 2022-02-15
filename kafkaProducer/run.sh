docker build --no-cache --tag kafka-producer .
docker tag kafka-producer wagnerrua/kafka:producer
docker push wagnerrua/kafka:producer