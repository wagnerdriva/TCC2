from kafka import KafkaConsumer, KafkaProducer
import sys, json, random

print("Classificador de cores iniciado com sucesso...")

producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers='10.0.10.11:9092')
consumer = KafkaConsumer('imagem_veiculo', group_id='color_classification', bootstrap_servers='10.0.10.11:9092')


for msg in consumer:
    info = json.loads(msg.value)
    print(f"Novo veiculo recebido: {info}")

    colors = ["Azul", "Vemelho", "Laranja", "Vermelho", "Verde", "Branco", "Preto", "Cinza"]

    event = {
        "vehicleID": info["id"],
        "color": colors[random.randint(0, 7)],
        "type": "ColorClassification" 
    }

    future = producer.send('event_bus', event)
    result = future.get(timeout=60)
    print("Color Classification enviou: ", event)
