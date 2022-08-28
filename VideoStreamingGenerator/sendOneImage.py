from kafka import KafkaProducer
from dotenv import load_dotenv
import cv2, json, base64, os

load_dotenv()

def serializeImg(img):
    # scale_percent = 50 # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    img_bytes = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    return img_bytes


def publishFrame(producer, imagePath):
    image = cv2.imread(imagePath) 

    imageBytes = serializeImg(image)

    data = {
        "vehicleID": 96,
        "frame": imageBytes,
        "conf": 0.92,
        "type": "NewVehicle"
    }

    future = producer.send('imagem_veiculo', data)
    future.get(timeout=60)

    return
    

print("Conectando ao broker: ", os.getenv("KAFKA_BROKER"))
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers=os.getenv("KAFKA_BROKER"))
imageDir = "./images/0.92-output111.jpg"

publishFrame(producer, imageDir)