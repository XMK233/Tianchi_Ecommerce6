import os

class Config:
    # Base paths based on config.py location
    CODE_PLAN_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PLAN_DIR = CODE_PLAN_DIR.replace("forCoding_code", "forCoding_data")
    
    # Project Data Root (parent of plan_1 in data dir)
    DATA_ROOT = os.path.dirname(DATA_PLAN_DIR)
    
    TRAINED_MODEL_PATH = os.path.join(DATA_PLAN_DIR, "trained_models")
    if not os.path.exists(TRAINED_MODEL_PATH):
        os.makedirs(TRAINED_MODEL_PATH, exist_ok=True)
        
    TEST_OUTPUT = os.path.join(DATA_PLAN_DIR, "preprocessedData/Result.csv")
    if not os.path.exists(os.path.dirname(TEST_OUTPUT)):
        os.makedirs(os.path.dirname(TEST_OUTPUT), exist_ok=True)
    
    TRAIN_REVIEWS = os.path.join(DATA_ROOT, 'TRAIN', 'Train_reviews.csv')
    TRAIN_LABELS = os.path.join(DATA_ROOT, 'TRAIN', 'Train_labels.csv')
    TEST_REVIEWS = os.path.join(DATA_ROOT, 'TEST', 'Test_reviews.csv')
    
    BERT_PATH = '/mnt/d/ModelScopeModels/google-bert/bert-base-chinese/'

    MAX_LEN = 512
    BATCH_SIZE = 16
    EPOCHS_EXTRACT = 25
    EPOCHS_CLASSIFY = 25
    LR = 2e-5
    
    CATEGORIES = [
        'None', '价格', '使用体验', '其他', '功效', '包装', '尺寸', 
        '成分', '整体', '新鲜度', '服务', '气味', '物流', '真伪'
    ]
    
    POLARITIES = ['正面', '负面', '中性']
    
    # Tagging labels
    TAGS = ['O', 'B-ASP', 'I-ASP', 'B-OP', 'I-OP']
    TAG2IDX = {t: i for i, t in enumerate(TAGS)}
    IDX2TAG = {i: t for i, t in enumerate(TAGS)}
    
    # Classification labels (Category + Polarity combined? Or separate?)
    # Let's do separate.
    CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
    IDX2CAT = {i: c for i, c in enumerate(CATEGORIES)}
    
    POL2IDX = {p: i for i, p in enumerate(POLARITIES)}
    IDX2POL = {i: p for i, p in enumerate(POLARITIES)}
