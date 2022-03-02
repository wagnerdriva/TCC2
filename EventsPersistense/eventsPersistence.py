from kafka import KafkaConsumer, KafkaProducer
from pymongo import MongoClient
import sys, json, random

print("Persistencia de eventos iniciada com sucesso...")

producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers='10.0.10.11:9092')
consumer = KafkaConsumer('event_bus', group_id='events-persistence', bootstrap_servers='10.0.10.11:9092')


url = f'mongodb://admin:admin@10.0.10.11:27017/'
client = MongoClient(url)

collection = client['vehicles']['vehicles']
# cursor = collection.find({})
# for document in cursor:
#     print(document)

for msg in consumer:
    info = json.loads(msg.value)
    print(f"Novo evento recebido recebido: {info}")

    collection.insert_one(info)
