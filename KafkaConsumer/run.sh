docker build --no-cache --tag kafka-consumer .
docker tag kafka-consumer wagnerrua/kafka:consumer
docker push wagnerrua/kafka:consumer