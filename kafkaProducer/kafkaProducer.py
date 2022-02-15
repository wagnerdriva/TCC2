from kafka import KafkaProducer
import json, time


print("Producer iniciado com sucesso...")
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers='10.0.10.11:9092')

for i in range(10):
    future = producer.send('topic_test', {'foo': i})
    result = future.get(timeout=60)
    print("Producer enviou: ", {'foo': i})
    time.sleep(10)