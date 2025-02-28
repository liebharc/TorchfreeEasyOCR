import os

os.environ["LRU_CACHE_CAPACITY"] = "1"

BASE_PATH = os.path.dirname(__file__)

imgH = 64

recognition_models = {
    'gen2' : {
        'english_g2':{
            'filename': 'english_g2.pth',
            'model_script': 'english',
            'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip',
            'md5sum': '5864788e1821be9e454ec108d61b887d',
            'symbols': "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €",
            'characters': "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        }
    }
}
