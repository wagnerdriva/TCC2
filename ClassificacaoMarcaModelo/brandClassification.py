from kafka import KafkaConsumer, KafkaProducer
import sys, json, random

print("Classificador de marcas iniciado com sucesso...")

producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers='10.0.10.11:9092')
consumer = KafkaConsumer('imagem_veiculo', group_id='brand_classification', bootstrap_servers='10.0.10.11:9092')


for msg in consumer:
    info = json.loads(msg.value)
    print(f"Novo veiculo recebido: {info}")

    brands = ["Citroen", "Ford", "Renault", "Fiat", "BMW", "Ferrari", "Volkswagen", "Toyota"]

    event = {
        "vehicleID": info["id"],
        "brand": brands[random.randint(0, 7)],
        "type": "CarClassification" 
    }

    future = producer.send('event_bus', event)
    result = future.get(timeout=60)
    print("Brand Classification enviou: ", event)
