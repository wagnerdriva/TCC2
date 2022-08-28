from kafka import KafkaConsumer, KafkaProducer
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import json, base64, cv2, time, os

load_dotenv()

print("Conectando ao broker: ", os.getenv("KAFKA_BROKER"))
print("Producer topic: ", os.getenv("KAFKA_PRODUCER_TOPIC"))
print("Consumer topic: ", os.getenv("KAFKA_CONSUMER_TOPIC"))
print("Consumer group: ", os.getenv("KAFKA_CONSUMER_GROUP"))
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers=os.getenv("KAFKA_BROKER"))
consumer = KafkaConsumer(os.getenv("KAFKA_CONSUMER_TOPIC"), group_id=os.getenv("KAFKA_CONSUMER_GROUP"), bootstrap_servers=os.getenv("KAFKA_BROKER"))

print("Carregando modelo..")
# load the model we saved
model = load_model('./model/EfficientNetB1-ColorsV2-Final.hdf5')
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',metrics=['categorical_accuracy'])

cores = ['Preto', 'Azul', 'Marrom', 'Amarelo', 'Verde', 'Vermelho', 'Prata', 'Branco']

print("Classificador de cores iniciado com sucesso...")

for msg in consumer:
    info = json.loads(msg.value)

    #print('Iniciando predicao...')

    start = time.time()
    jpg_original = base64.b64decode(info["frame"])
    nparr = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (299,299))
    img=np.expand_dims(img, axis=0)


    pred = model.predict(img)

    index=np.argmax(pred[0])
    klass=cores[index]
    probability=pred[0][index]*100
    end = time.time()
    print(f'Cor: {klass} / Probabilidade: {probability} / Tempo: {end - start}')

    event = {
        "vehicleID": info.get("vehicleID"),
        "color": klass,
        "probability": probability,
        "type": "ColorClassification" 
    }

    future = producer.send(os.getenv("KAFKA_PRODUCER_TOPIC"), event)
    result = future.get(timeout=60)