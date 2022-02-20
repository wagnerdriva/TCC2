from kafka import KafkaConsumer, KafkaProducer
import sys, json, random

print("Detector de placas iniciado com sucesso...")

producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers='10.0.10.11:9092')
consumer = KafkaConsumer('imagem_veiculo', group_id='deteccao_placas', bootstrap_servers='10.0.10.11:9092')


for msg in consumer:
    info = json.loads(msg.value)
    print(f"Novo veiculo recebido: {info}")

    event = {
        "vehicleID": info["id"],
        "plate": random.randint(1, 100),
        "type": "PlateDetection" 
    }

    future = producer.send('event_bus', event)
    result = future.get(timeout=60)
    print("Plate Detection enviou: ", event)
