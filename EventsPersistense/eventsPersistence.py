from collections import Counter
from kafka import KafkaConsumer
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import json, cv2, base64, os
import numpy as np

load_dotenv()

print("Conectando ao broker: ", os.getenv("KAFKA_BROKER"))
print("Consumer topic: ", os.getenv("KAFKA_CONSUMER_TOPIC"))
print("Consumer group: ", os.getenv("KAFKA_CONSUMER_GROUP"))
consumer = KafkaConsumer(os.getenv("KAFKA_CONSUMER_TOPIC"), group_id=os.getenv("KAFKA_CONSUMER_GROUP"), bootstrap_servers=os.getenv("KAFKA_BROKER"))

print("Conectando ao mongodb: ", os.getenv("MONGODB_URL"))
url = os.getenv("MONGODB_URL")
client = MongoClient(url)
print("Versao mongodb: ", client.server_info()["version"])

collection = client['vehicles']['vehicles']

import requests

def sendtoserver(img, id, name):
    imencoded = cv2.imencode(".jpg", img)[1]
    file = {'image': (name, imencoded.tostring(), 'image/jpeg', {'Expires': '0'})}
    response = requests.post(f"http://192.168.200.110:3002/vemos/data/upload/{id}", files=file)
    return response

def choiceAttributes(possibleField, field):
    countColors = Counter()
    for item in vehicle.get(possibleField, []):
        countColors[item[field]] += 1

    maisComuns = countColors.most_common()
    
    finalAttribute = ""
    if len(maisComuns) < 2:
        finalAttribute = maisComuns[0][0]
    elif maisComuns[0][1] > maisComuns[1][1]:
        finalAttribute = maisComuns[0][0]
    else:
        probEscolhida = 0
        for counted in maisComuns:
            mean = 0
            listAttributes = list(filter(lambda d: d[field] == counted[0], vehicle.get(possibleField, [])))
            for model in listAttributes:
                mean += model.get("probability")
            if probEscolhida < mean/len(listAttributes):
                probEscolhida = mean/len(listAttributes)
                finalAttribute = counted[0]

    print(finalAttribute)
    return finalAttribute

print("Persistencia de eventos iniciada com sucesso...")

for msg in consumer:
    info = json.loads(msg.value)

    id = info["vehicleID"]
    dt = datetime.now()
    ts = datetime.timestamp(dt)

    x = collection.find_one({"id": id})
    if x == None:
        collection.insert_one({ "id": id, "createdAt": dt})


    if info["type"] == "ColorClassification":
        collection.update_one({ "id": info["vehicleID"] }, {"$push":{"possibleColors": { "probability": info["probability"], "color": info["color"]}}})

    elif info["type"] == "CarClassification":
        collection.update_one(
            { "id": info["vehicleID"] }, 
            {"$push":{ "possibleModels": { "probability": info["modelo"]["prob_modelo"], "model": info["modelo"]["modelo"]},
                       "possibleBrands": { "probability": info["marca"]["prob_marca"], "brand": info["marca"]["marca"]},
                       "possibleCategories": { "probability": info["categoria"]["prob_categoria"], "category": info["categoria"]["categoria"]}}})

    elif info["type"] == "PlateDetection":
        collection.update_one({ "id": info["vehicleID"] }, {"$push":{"possiblePlates": info["plate"]}})

        jpg_original = base64.b64decode(info["frame"])
        nparr = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image_name = f"vehicle{id}_{dt}.jpg"
        sendtoserver(img, id, image_name)
        collection.update_one({ "id": info["vehicleID"] }, {"$push":{"possibleImages": image_name}})

    vehicle = collection.find_one({"id": id})

    if info["type"] == "ColorClassification":
        color = choiceAttributes("possibleColors", "color")
        collection.update_one({ "id": id }, {"$set":{"color": color}})

    elif info["type"] == "CarClassification":
        model = choiceAttributes("possibleModels", "model")
        model = model.split("_")
        if len(model) > 1:
            model = model[1]
        else:
            model = model[0]
        collection.update_one({ "id": id }, {"$set":{"model": model}})

        brand = choiceAttributes("possibleBrands", "brand")
        collection.update_one({ "id": id }, {"$set":{"brand": brand}})

        category = choiceAttributes("possibleCategories", "category")
        collection.update_one({ "id": id }, {"$set":{"category": category}})

    elif info["type"] == "PlateDetection":
        countPlates = Counter(vehicle.get("possiblePlates", []))
        collection.update_one({ "id": id }, {"$set":{"plate": countPlates.most_common()[0][0]}})

    


    

# event = {
#         "id": uniqueID,
#         "image": base64.b64encode(uniqueID.encode("ascii")).decode("ascii"),
#         "camera": random.randint(1, 5),
#         "timestamp": 
#         "type": "NewVehicle"
#     }

# {
#     id: string
#     images: string[]
#     placa: string
#     marca: string
#     modelo: string
#     cor: string
#     locations: [
#         {
#             timestamp: string
#             position: string
#         }
#     ]
# }