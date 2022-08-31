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

print("Carregando modelos..")
# load the model we saved

print("Carregando modelo de categorias: ", os.getenv("MODEL_CATEGORIAS"))
modelCategorias = load_model(os.getenv("MODEL_CATEGORIAS"))
print("Carregando modelo de marcas: ", os.getenv("MODEL_MARCAS"))
modelMarcas = load_model(os.getenv("MODEL_MARCAS"))
print("Carregando modelo de modelos: ", os.getenv("MODEL_MODELOS"))
modelModelos = load_model(os.getenv("MODEL_MODELOS"))


modelCategorias.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',metrics=['categorical_accuracy'])
modelMarcas.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',metrics=['categorical_accuracy'])
modelModelos.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',metrics=['categorical_accuracy'])


modelos = ['106', '147-Spazio', '2008', '206', '206-SW', '207', '207-SW', '208', '3008', '307', '308', '408', '500', 'ASX', 'Accent', 'Accord', 'Aircross', 'Amarok', 'Argo', 'Atos', 'Azera', 'BR800', 'Belina', 'Berlingo', 'Brasilia', 'Bravo', 'C3', 'C3-antigo', 'C4-Cactus', 'C4-Lounge', 'C4-Picasso', 'CR-V', 'Captur', 'Cerato', 'Cherokee', 'Chevrolet Agile', 'Chevrolet Astra', 'Chevrolet Camaro', 'Chevrolet Captiva', 'Chevrolet Caravan', 'Chevrolet Celta', 'Chevrolet Chevette', 'Chevrolet Classic', 'Chevrolet Cobalt', 'Chevrolet Corsa', 'Chevrolet Cruze', 'Chevrolet Equinox', 'Chevrolet Ipanema', 'Chevrolet Kadett', 'Chevrolet Marajo', 'Chevrolet Meriva', 'Chevrolet Montana', 'Chevrolet Monza', 'Chevrolet Opala', 'Chevrolet S10', 'Chevrolet Silverado', 'Chevrolet Sonic', 'Chevrolet Spin', 'Chevrolet Suprema', 'Chevrolet Tracker', 'Chevrolet Vectra', 'Chevrolet Zafira', 'Cielo', 'City', 'Civic', 'Civic -antigo', 'Clio', 'Compass', 'Corolla', 'Corolla-antigo', 'Courier', 'Creta', 'Cronos', 'Del-Rey', 'Discovery', 'Doblo', 'Duster', 'Ecosport', 'Edge', 'Elantra', 'Etios', 'Evoque', 'Fiat_128', 'Fiat_500e', 'Fiat_grandsiena', 'Fiesta', 'Fiorino', 'Fit', 'Fit-antigo', 'Fluence', 'Focus', 'Ford_Contour', 'Ford_Escort', 'Ford_Taurus', 'Ford_Territory', 'Fortwo', 'Fox', 'Freemont', 'Frontier', 'Fusca', 'Fusion', 'Golf', 'HR-V', 'Hatch', 'Hilux', 'Hoggar', 'Honda_Civic_antigo', 'Hyundai_Accent', 'Hyundai_GrandSantaFe', 'Hyundai_HB20', 'Hyundai_HB20s', 'I30', 'I30-CW', 'IX35', 'Idea', 'J2', 'J3', 'J5', 'J6', 'JAC_T8', 'Jetta', 'Ka', 'Ka-antigo', 'Kia_Carens', 'Kia_Clarus', 'Kicks', 'Kwid', 'L200', 'Lancer', 'Linea', 'Livina', 'Logan', 'Logus', 'March', 'Marea', 'Megane', 'Mitsubishi_EclipseCross', 'Mitsubishi_L200Outdoor', 'Mitsubishi_L200Triton', 'Mitsubishi_OutlanderSport', 'Mitsubishi_PajeroSport', 'Mobi', 'Mondeo', 'Mustang', 'New-beetle', 'Nissan_Leaf', 'Nissan_Tiida', 'Nivus', 'Outlander', 'Pajero-Full', 'Palio', 'Pampa', 'Parati', 'Passat', 'Passat-antigo', 'Passat_Variant', 'Peugeot_306', 'Peugeot_406', 'Peugeot_RCZ', 'Picanto', 'Polo', 'Porsche_718', 'Porsche_PANAMERA', 'Premio', 'Prius', 'Punto', 'QQ', 'RAV4', 'Ranger', 'Renault_19', 'Renault_DusterOroch', 'Renegade', 'Royale', 'SUBARU_IMPREZA', 'SW4', 'Sandero', 'Santa-Fe', 'Santana', 'Saveiro', 'Scenic', 'Sentra', 'Sentra-antigo', 'Sephia', 'Siena', 'Sonata', 'Sorento', 'Soul', 'Spacefox', 'Sportage', 'Stilo', 'Strada', 'Symbol', 'T-cross', 'Tempra', 'Tiggo', 'Tiggo2', 'Tiggo5X', 'Tiguan', 'Tipo', 'Tipo_antigo', 'Toro', 'Toyota_Canry', 'Toyota_CorollaCross', 'Toyota_EtiosCross', 'Toyota_Fielder', 'Toyota_Tacoma', 'Toyota_Tundra', 'Tucson', 'Uno', 'Uno_Antigo', 'Up', 'V40', 'Veloster', 'Verona', 'Versa', 'Versailles', 'Virtus', 'Volkswagen_Apollo', 'Volkswagen_Bora', 'Volkswagen_CrossUp', 'Volkswagen_Gol', 'Volkswagen_PoloSedan', 'Volkswagen_Taos', 'Volvo_S60', 'Volvo_XC40', 'Volvo_XC90', 'Voyage', 'WR-V', 'Weekend', 'XC60', 'Xsara', 'Yaris', 'audi_a3', 'audi_a3_sedan', 'audi_a4_avant', 'audi_a4_sedan', 'audi_a5_sportback', 'audi_a6_avant', 'audi_a6_sedan', 'audi_a7_sportback', 'audi_q3', 'audi_q5', 'audi_q7', 'audi_q8', 'audi_rs5_sportback', 'bmw_116i', 'bmw_118i', 'bmw_120i', 'bmw_125i', 'bmw_130i', 'bmw_135i', 'bmw_135i_coupe', 'bmw_320', 'bmw_320i', 'bmw_X2', 'bmw_X3', 'bmw_X4', 'bmw_X6', 'bmw_m2', 'bmw_m5', 'bmw_m6_coupe', 'bmw_x1', 'bmw_x5', 'cherry_arrizo5', 'cherry_arrizo6', 'cherry_celer', 'cherry_celer_sedan', 'cherry_tiggo6', 'cherry_tiggo7', 'cherry_tiggo8', 'chevrolet_onix', 'chevrolet_prisma', 'citroen_c3picasso', 'citroen_c4grandpicasso', 'citroen_ds3', 'citroen_ds4', 'citroen_ds5', 'jeep_GrandCherokee', 'landrover_discoverysport', 'landrover_rangerover vogue', 'landrover_rangeroversport', 'landrover_rangerovervelar', 'mercedes_A200', 'mercedes_B200', 'mercedes_C180', 'mercedes_C43AMG', 'mercedes_C63AMG', 'mercedes_CL63AMG', 'mercedes_CLA180', 'mercedes_CLA200', 'mercedes_CLA250', 'mercedes_CLA45AMG', 'mercedes_CLS350', 'mercedes_CLS55AMG', 'mercedes_CLS63AMG', 'mercedes_GLA200', 'mercedes_GLA250', 'mercedes_SL55AMG', 'mercedes_SL63AMG', 'mercedes_SLK200', 'mercedes_SLK55AMG', 'mini_Countryman', 'mini_Roadster', 'mini_cooper']
marcas = ['Audi', 'BMW', 'Chery', 'Chevrolet', 'Citroen', 'Fiat', 'Ford', 'Gurgel', 'Honda', 'Hyundai', 'Jac', 'Jeep', 'Kia', 'LandRover', 'Mercedes', 'Mini', 'Mitsubishi', 'Nissan', 'Peugeot', 'Porsche', 'Renault', 'Smart', 'Subaru', 'Toyota', 'Volkswagen', 'Volvo']
categorias = ['HATCH', 'MINIVAN', 'PICAPE', 'SEDAN', 'SPORT', 'SUV']

print("Classificador de marcas iniciado com sucesso...")

def predictModelos(img):
    img = cv2.resize(img, (240, 240))
    img=np.expand_dims(img, axis=0)
    predictions = modelModelos.predict(img)

    index=np.argmax(predictions[0])
    klass=modelos[index]
    probability=predictions[0][index]*100
    print(f'Modelo: {klass} / Probabilidade: {probability}')

    return klass, probability

def predictCategorias(img):
    img = cv2.resize(img, (224, 224))
    img=np.expand_dims(img, axis=0)
    predictions = modelCategorias.predict(img)

    index=np.argmax(predictions[0])
    klass=categorias[index]
    probability=predictions[0][index]*100
    print(f'Categoria: {klass} / Probabilidade: {probability}')

    return klass, probability

def predictMarcas(img):
    img = cv2.resize(img, (240, 240))
    img=np.expand_dims(img, axis=0)
    predictions = modelMarcas.predict(img)

    index=np.argmax(predictions[0])
    klass=marcas[index]
    probability=predictions[0][index]*100
    print(f'Marca: {klass} / Probabilidade: {probability}')

    return klass, probability

for msg in consumer:
    info = json.loads(msg.value)

    jpg_original = base64.b64decode(info["frame"])
    nparr = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
    start = time.time()
    categoria, prob_categoria = predictCategorias(img)
    marca, prob_marca = predictMarcas(img)
    modelo, prob_modelo = predictModelos(img)
    end = time.time()

    print("Tempo total: ", (end - start))

    event = {
        "vehicleID": info.get("vehicleID"),
        "marca": { "marca": marca, "prob_marca": prob_marca},
        "modelo": { "modelo": modelo, "prob_modelo": prob_modelo},
        "categoria": { "categoria": categoria, "prob_categoria": prob_categoria},
        "type": "CarClassification" 
    }

    future = producer.send(os.getenv("KAFKA_PRODUCER_TOPIC"), event)
    future.get(timeout=60)