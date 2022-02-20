from kafka import KafkaProducer
import json, time, sys
import uuid, random, base64

print("Script de tracking iniciado com sucesso...")

producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers='10.0.10.11:9092')

# Por enquanto estamos fazendo um esqueleto que publica informações falsas nos topicos

while True:
    uniqueID = str(uuid.uuid4())
    event = {
        "id": uniqueID,
        "image": base64.b64encode(uniqueID.encode("ascii")).decode("ascii"),
        "camera": random.randint(1, 5),
        "type": "NewVehicle"
    }

    future = producer.send('imagem_veiculo', event)
    result = future.get(timeout=60)
    print("Tracking enviou: ", event)

    time.sleep(10)
