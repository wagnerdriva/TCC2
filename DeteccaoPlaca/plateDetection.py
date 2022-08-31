# required library
import json, base64, cv2, time, os

from sklearn.preprocessing import LabelEncoder
from kafka import KafkaConsumer, KafkaProducer
from keras.models import model_from_json
from local_utils import detect_lp
from dotenv import load_dotenv
from os.path import splitext
from PIL import Image
import numpy as np

load_dotenv()

print("Conectando ao broker: ", os.getenv("KAFKA_BROKER"))
print("Producer topic: ", os.getenv("KAFKA_PRODUCER_TOPIC1"), os.getenv("KAFKA_PRODUCER_TOPIC2"))
print("Consumer topic: ", os.getenv("KAFKA_CONSUMER_TOPIC"))
print("Consumer group: ", os.getenv("KAFKA_CONSUMER_GROUP"))
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), bootstrap_servers=os.getenv("KAFKA_BROKER"))
consumer = KafkaConsumer(os.getenv("KAFKA_CONSUMER_TOPIC"), group_id=os.getenv("KAFKA_CONSUMER_GROUP"), bootstrap_servers=os.getenv("KAFKA_BROKER"))

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(img, resize=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Load model architecture, weight and labels
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    
    return prediction

print("Detector de placas iniciado com sucesso...")

def detecta_placa(img):

    vehicle, LpImg, cor = get_plate(img)

    pixvals = LpImg[0]
    minval = np.percentile(pixvals, 30)
    maxval = np.percentile(pixvals, 60)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 255
    Image.fromarray(pixvals.astype(np.uint8))
    pixvals = pixvals.astype(np.uint8)

    if (len(LpImg)): #check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(7,7),0)
        
        # Applied inversed thresh_binary 
        binary = cv2.threshold(blur, 180, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)



    # ALTERAÇÃO MOISES
    # esse método abaixo detecta muito melhor
    cont, _  = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(cont))

    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()
    test_roi = plate_image.copy()

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        # ALTERAÇÃO MOISES
        #if 1<=ratio<=3.5: # Only select contour with defined ratio
        #    if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
        if 0.35 < (w/h) < 1.2: # se a divisão da largura pela altura está no intervalo (0.35, 1.2)
            if 0.03 < ((w*h)/(plate_image.shape[0] * plate_image.shape[1])) < 0.1: # se a área do identificado/area da imagem está entre (0.03, 0.1)
        
        
                #print((w*h)/(plate_image.shape[0] * plate_image.shape[1]))
                # print(w/h)
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    print("Detect {} letters...".format(len(crop_characters)))
    if len(crop_characters) < 4:
        return ""

    cols = len(crop_characters)

    final_string = ''
    for i,character in enumerate(crop_characters):
        title = np.array2string(predict_from_model(character,model,labels))
        final_string+=title.strip("'[]")

    return final_string

for msg in consumer:
    info = json.loads(msg.value)

    jpg_original = base64.b64decode(info["frame"])
    nparr = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # print('Iniciando predicao...')
    placa = ""
    try:
        start = time.time()
        placa = detecta_placa(img)
        end = time.time()
        print(f'Placa: {placa} / Tempo: {end - start}')
    except:
        placa = ""

    if len(placa) >= 4:
        event = {
            "vehicleID": info.get("vehicleID"),
            "frame": info["frame"],
            "plate": placa,
            "type": "PlateDetection" 
        }

        future = producer.send(os.getenv("KAFKA_PRODUCER_TOPIC1"), event)
        future.get(timeout=60)
        future = producer.send(os.getenv("KAFKA_PRODUCER_TOPIC2"), event)
        future.get(timeout=60)