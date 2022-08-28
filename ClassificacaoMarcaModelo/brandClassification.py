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
modelCategorias = load_model('./models/EfficientNetB0-Categorias-Final.hdf5')
modelMarcas = load_model('./models/EfficientNetB1-Marcas-Final.hdf5')
modelModelos = load_model('./models/EfficientNetB1-Modelos-Final.hdf5')


modelCategorias.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',metrics=['categorical_accuracy'])
modelMarcas.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',metrics=['categorical_accuracy'])
modelModelos.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',metrics=['categorical_accuracy'])


modelos = ['106', '2008', '3008', '940', 'ASX - antigo', 'Accent', 'Bandeirante', 'Bandeirante-camionete', 'Berlingo-antigo', 'Boxer', 'C3-antigo', 'Chevrolet Space-van', 'Chevrolet Trafic', 'Civic', 'Civic -antigo', 'Corolla', 'Corolla-antigo', 'Duster', 'F250', 'Fiat_128', 'Fiat_500e', 'Fiat_coupe', 'Fiat_grandsiena', 'Fit-antigo', 'Ford_Bronco', 'Ford_Consul', 'Ford_Contour', 'Ford_Escort', 'Ford_Expedition', 'Ford_Explorer', 'Ford_F350', 'Ford_F4000', 'Ford_Furglaine', 'Ford_Maverick', 'Ford_Rural', 'Ford_Taurus', 'Ford_Territory', 'Ford_Torino', 'Ford_f100', 'H100', 'Honda_Civic_antigo', 'Hyundai_Accent', 'Hyundai_Equus', 'Hyundai_Galloper', 'Hyundai_Genesis', 'Hyundai_GrandSantaFe', 'Hyundai_H1Starex', 'Hyundai_HB20', 'Hyundai_HB20X', 'Hyundai_HB20s', 'Hyundai_Terracan', 'Hyundai_Tiburon', 'Hyundai_Veracruz', 'INFINITI_EX35', 'INFINITI_FX35', 'INFINITI_FX45', 'INFINITI_FX50', 'INFINITI_G37', 'INFINITI_J30', 'JAC_IEV40', 'JAC_T5', 'JAC_T50', 'JAC_T60', 'JAC_T8', 'JAC_T80', 'JAC_V260', 'JipeUniversal', 'Jumper', 'Jumpy', 'Kia_Bongo', 'Kia_Cadenza', 'Kia_Carens', 'Kia_Carnival', 'Kia_Clarus', 'Kia_Magentis', 'Kia_Mohave', 'Kia_Opirus', 'Kia_Optima', 'Kia_Quoris', 'Kia_Rio', 'Kia_Stinger', 'Lexus_RX350FSPORT', 'Mitsubishi_3000GT', 'Mitsubishi_EclipseCross', 'Mitsubishi_Galant', 'Mitsubishi_L200Outdoor', 'Mitsubishi_L200Savana', 'Mitsubishi_L200Triton', 'Mitsubishi_OutlanderSport', 'Mitsubishi_PajeroDakar', 'Mitsubishi_PajeroIO', 'Mitsubishi_PajeroSport', 'Mitsubishi_SpaceWagon', 'Mitsubishi_eclipse', 'Nissan_ Xtrail', 'Nissan_Altima', 'Nissan_Armada', 'Nissan_GTR', 'Nissan_Grand Livina', 'Nissan_Juke', 'Nissan_Leaf', 'Nissan_Maxima', 'Nissan_Murano', 'Nissan_Pathfinder', 'Nissan_Pickup', 'Nissan_Quest', 'Nissan_Tiida', 'Nissan_Xterra', 'Nivus', 'Oggi', 'Peugeot_306', 'Peugeot_406', 'Peugeot_5008', 'Peugeot_RCZ', 'Porsche_718', 'Porsche_911', 'Porsche_914', 'Porsche_928', 'Porsche_944', 'Porsche_BOXSTER', 'Porsche_CAYENNE', 'Porsche_CAYMAN', 'Porsche_CAYMANS', 'Porsche_MACAN', 'Porsche_PANAMERA', 'Porsche_SPYDER550', 'Porsche_TAYCAN', 'RAM_1500', 'RAM_2500', 'Renault_19', 'Renault_Dauphine', 'Renault_DusterOroch', 'Renault_GrandScenic', 'Renault_Laguna', 'Renault_Trafic', 'Renault_Zoe', 'SUBARU_IMPREZA', 'SUBARU_LEGACY', 'SUBARU_SVX', 'SUBARU_VIVIO', 'SUBARU_XV', 'Santana-quantum', 'Sentra-antigo', 'Suzuki_GrandVitara', 'Suzuki_SX4', 'Suzuki_Samurai', 'Suzuki_Scross', 'Suzuki_Swift', 'T-cross', 'Tipo', 'Toyota_Canry', 'Toyota_CorollaCross', 'Toyota_Corona', 'Toyota_EtiosCross', 'Toyota_FJCruiser', 'Toyota_Fielder', 'Toyota_Highlander', 'Toyota_LandCruiserPrado', 'Toyota_Paseo', 'Toyota_Sienna', 'Toyota_Tacoma', 'Toyota_Tundra', 'Toyota_Venza', 'Troller_T4Expedition', 'Troller_TX4', 'Twingo', 'Variant', 'Variant_Novo', 'Volkswagen_Apollo', 'Volkswagen_Bora', 'Volkswagen_CrossUp', 'Volkswagen_EOS', 'Volkswagen_Eurovan', 'Volkswagen_Gol', 'Volkswagen_PassatVariant', 'Volkswagen_Pointer', 'Volkswagen_PoloSedan', 'Volkswagen_Quantum', 'Volkswagen_Taos', 'Volkswagen_Touareg', 'Volvo_850', 'Volvo_940', 'Volvo_960', 'Volvo_C30', 'Volvo_C70', 'Volvo_S40', 'Volvo_S60', 'Volvo_S70', 'Volvo_S80', 'Volvo_S90', 'Volvo_V50', 'Volvo_V60', 'Volvo_V70', 'Volvo_XC40', 'Volvo_XC90', 'Weekend', 'Wrangler', 'Xsara-antigo', 'asia_towner', 'audi_a3', 'audi_a3_sedan', 'audi_a4_avant', 'audi_a4_sedan', 'audi_a5_sportback', 'audi_a6_avant', 'audi_a6_sedan', 'audi_a7_sportback', 'audi_q3', 'audi_q5', 'audi_q7', 'audi_q8', 'audi_r8', 'audi_rs5_sportback', 'audi_tt', 'bmw_116i', 'bmw_118i', 'bmw_120i', 'bmw_125i', 'bmw_130i', 'bmw_135i', 'bmw_135i_coupe', 'bmw_320', 'bmw_320i', 'bmw_X2', 'bmw_X3', 'bmw_X4', 'bmw_X6', 'bmw_Z3', 'bmw_m2', 'bmw_m4', 'bmw_m5', 'bmw_m6_cabriolet', 'bmw_m6_coupe', 'bmw_x1', 'bmw_x5', 'cadillac_ctsv', 'cadillac_escalade', 'cadillac_limousine', 'cadillac_srx', 'cherry_arrizo5', 'cherry_arrizo6', 'cherry_celer', 'cherry_celer_sedan', 'cherry_tiggo6', 'cherry_tiggo7', 'cherry_tiggo8', 'chevrolet_c10', 'chevrolet_c20', 'chevrolet_corvette', 'chevrolet_malibu', 'chevrolet_onix', 'chevrolet_prisma', 'chrysler_300M', 'chrysler_300c', 'chrysler_crossfire', 'chrysler_stratus', 'citroen_c3picasso', 'citroen_c4grandpicasso', 'citroen_ds3', 'citroen_ds4', 'citroen_ds5', 'citroen_xsarapicasso', 'dodge_durango', 'dodge_grandcaravan', 'dodge_journey', 'dodge_ram', 'dodge_stealth', 'effa_k01', 'effa_v21', 'effa_v22', 'effa_v25', 'effa_van', 'honda_legend', 'honda_prelude', 'jaguar_fpace', 'jaguar_ftype', 'jeep_Commander', 'jeep_GrandCherokee', 'jeep_Liberty', 'jeep_cj5', 'jeep_cj6', 'landrover_discovery2', 'landrover_discovery3', 'landrover_discovery4', 'landrover_discoverysport', 'landrover_freelander', 'landrover_freelander2', 'landrover_rangerover', 'landrover_rangerover vogue', 'landrover_rangeroversport', 'landrover_rangerovervelar', 'lexus_ES', 'lexus_ES300H', 'lexus_ES330', 'lexus_ES350', 'lexus_IS250', 'lexus_IS250FSPORT', 'lexus_LS', 'lexus_LS430', 'lexus_LS460L', 'lexus_NX200T', 'lexus_NX300', 'lexus_NX300h', 'lexus_RX350', 'lexus_RX450h', 'lexus_UX250h', 'marcopolo_volare', 'mercedes_A160', 'mercedes_A200', 'mercedes_B180', 'mercedes_B200', 'mercedes_C180', 'mercedes_C180K', 'mercedes_C43AMG', 'mercedes_C63AMG', 'mercedes_CL63AMG', 'mercedes_CLA180', 'mercedes_CLA200', 'mercedes_CLA250', 'mercedes_CLA45AMG', 'mercedes_CLK320', 'mercedes_CLK430', 'mercedes_CLS350', 'mercedes_CLS55AMG', 'mercedes_CLS63AMG', 'mercedes_GL500', 'mercedes_GLA200', 'mercedes_GLA250', 'mercedes_GLA45AMG', 'mercedes_GLC300', 'mercedes_GLS450', 'mercedes_GLS500', 'mercedes_SL320', 'mercedes_SL350', 'mercedes_SL400', 'mercedes_SL500', 'mercedes_SL55AMG', 'mercedes_SL63AMG', 'mercedes_SLC300', 'mercedes_SLC43AMG', 'mercedes_SLK200', 'mercedes_SLK230', 'mercedes_SLK320', 'mercedes_SLK55AMG', 'mercedes_SPRINTER', 'mercedes_VITO', 'mini_Countryman', 'mini_JohnCooperWorks', 'mini_One', 'mini_Paceman', 'mini_Roadster', 'mini_cooper']
marcas = ['Audi', 'BMW', 'Cadillac', 'Cherry', 'Chevrolet', 'Chrysler', 'Citroen', 'Dodge', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'Infiniti', 'JAC', 'Jaguar', 'Jeep', 'Kia', 'LandRover', 'Lexus', 'Mercedes', 'Mini', 'Mitsubishi', 'Nissan', 'Peugeot', 'Porsche', 'RAM', 'Renault', 'Subaru', 'Susuki', 'Toyota', 'Troller', 'Volkswagen', 'Volvo', 'Willys']
categorias = ['HATCH', 'MINIBUS', 'MINIVAN', 'PICAPE', 'SEDAN', 'SPORT', 'SUV']

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