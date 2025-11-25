DATA_DIR = "../../data/"
TRAIN_DIR = "C:/Users/haric/Downloads/aptos2019/train_images/"
CSV_PATH = "C:/Users/haric/Downloads/aptos2019/train.csv"
TEST_DIR = "C:/Users/haric/Downloads/aptos2019/test_images/"
TEST_CSV = "C:/Users/haric/Downloads/aptos2019/test.csv"
MODEL_PATH = "../../data/models/"

TRAIN_SPLIT = 0.8
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
NUM_WORKERS = 0
EPOCHS = 14
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASSES = 5
MODEL_NAME = "vit_base_patch16_224"
USE_MEDT = True
MEDT_VARIANT = "medt"
MEDT_CKPT = ""
FREEZE_MEDT = True
MODEL_SAVE = MODEL_PATH + MODEL_NAME
USE_AMP = True

