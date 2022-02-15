from kafka import KafkaConsumer
import sys

print("Consumer iniciado com sucesso...")
consumer = KafkaConsumer('topic_test', group_id='favorite_group', bootstrap_servers='10.0.10.11:9092')

for msg in consumer:
    print(msg)