import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image

def load_species_list(file_path='labels.txt'):
    """Load species list from labels.txt file"""
    with open(file_path, 'r') as f:
        species_list = [line.strip() for line in f.readlines() if line.strip()]
    return species_list

# Load species list from file
SPECIES_LIST = load_species_list()

# Mushroom species database
MUSHROOM_DB = {
    'Amanita phalloides': {
        'kategori': 'Beracun',
        'ciri_ciri': [
            'Bentuk tudung seperti topi',
            'Tudung berwarna hijau muda hampir putih',
            'Tangkai berwarna putih'
        ],
        'gambar': 'assets/amanita.jpg'
    },
    'Boletus edulis': {
        'kategori': 'Dapat Dikonsumsi',
        'ciri_ciri': [
            'Bentuk tudung tebal, berdaging, dan cembung',
            'Tudung berwarna cokelat kekuningan hingga cokelat kemerahan tua',
            'Bagian bawah tudung memiliki pori-pori berwarna kuning'
        ],
        'gambar': 'assets/boletus.jpg'
    },
    'Chlorophyllum molybdites': {
        'kategori': 'Beracun',
        'ciri_ciri': [
            'Bentuk tudung memiliki sisik dan tonjolan di tengah',
            'Tudung berwarna putih hingga krem',
            'Bawah tudung berwarna krem kehijauan hingga hijau gelap'
        ],
        'gambar': 'assets/chlorophyllum.jpg'
    },
    'Flammulina velutipes': {
        'kategori': 'Dapat Dikonsumsi',
        'ciri_ciri': [
            'Bentuk tudung berukuran 2 sampai 10 cm',
            'Tudung berwarna oranye cerah',
            'Batang ditutupi bulu halus'
        ],
        'gambar': 'assets/flammulina.jpg'
    },
    'Galerina marginata': {
        'kategori': 'Beracun',
        'ciri_ciri': [
            'Bentuk tudung seperti parabola',
            'Tudung berwarna oranye kecokelatan',
            'Tangkai berwarna kuning kecokelatan'
        ],
        'gambar': 'assets/galerina.jpg'
    },
    'Pleurotus ostreatus': {
        'kategori': 'Dapat Dikonsumsi',
        'ciri_ciri': [
            'Bentuk tudung setengah lingkaran seperti kipas',
            'Tudung berwarna putih hingga kekuningan',
            'Ukuran tangkai sangat pendek'
        ],
        'gambar': 'assets/pleurotus.jpg'
    }
}

def load_keras_model(model_path):
    """Load and cache the model"""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model: {str(e)}")

def preprocess_image(img_file, target_size=(384, 384)):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(img_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        return img_array, img
    except Exception as e:
        raise RuntimeError(f"Gagal memproses gambar: {str(e)}")


def predict_mushroom(img_file, model, threshold=0.5):
    """Make prediction on mushroom image"""
    try:
        img_array, img = preprocess_image(img_file)
        
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        confidence = float(preds[0][pred_idx] * 100)
        
        if confidence < (threshold * 100):
            return {
                'species': 'Tidak Dikenal',
                'category': 'Tidak Dikenal',
                'confidence': 0,
                'recognized': False,
                'image': img
            }
        else:
            species = SPECIES_LIST[pred_idx]
            return {
                'species': species,
                'category': MUSHROOM_DB[species]['kategori'],
                'confidence': confidence,
                'recognized': True,
                'image': img
            }
    except Exception as e:
        raise RuntimeError(f"Gagal membuat prediksi: {str(e)}")

def get_species_details(species_name):
    """Get complete details for a species"""
    return MUSHROOM_DB.get(species_name, None)