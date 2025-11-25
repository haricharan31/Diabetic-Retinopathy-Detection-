TRAIN_DIR = "C:/Users/haric/Downloads/aptos2019/train_images/"
TEST_DIR = "C:/Users/haric/Downloads/aptos2019/test_images/"
CSV_PATH = "C:/Users/haric/Downloads/aptos2019/train.csv"
TEST_CSV = "C:/Users/haric/Downloads/aptos2019/test.csv"
MODEL_PATH = "../../data/models/"

TRAIN_SPLIT = 0.8
LEARNING_RATE = 3e-5
HEAD_LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 14
FREEZE_AFTER_EPOCHS = 10
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASSES = 5
MODEL_NAME = "vit_base_patch16_224"
USE_AMP = True
EXPECTED_ACCURACY_GAIN = 0.25
CHECKPOINT_DIR = "results/"


