import json, time, sys
import uuid, random, base64

print("Script de tracking iniciado com sucesso...")

producer = None
if len(sys.argv) > 1 and sys.argv[1] != "dev":
    from kafka import KafkaProducer
    producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers='10.0.10.11:9092')

# Por enquanto estamos fazendo um esqueleto que publica informações falsas nos topicos

while True:
    uniqueID = str(uuid.uuid4())
    vehicle = {
        "id": uniqueID,
        "image": base64.b64encode(uniqueID.encode("ascii")).decode("ascii"),
        "camera": random.randint(1, 5)
    }

    if not producer:
        print(vehicle)
    else:
        future = producer.send('imagem_veiculo', vehicle)
        result = future.get(timeout=60)

    time.sleep(10)
