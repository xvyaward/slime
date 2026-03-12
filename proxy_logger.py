import logging
from flask import Flask, request, jsonify
import requests
from transformers import AutoTokenizer
import json
import sys

app = Flask(__name__)
TARGET_URL = "http://127.0.0.1:13142"
MODEL_PATH = "/root/changhun.lee/models/EXAONE-4.0-32B"

# --- 로깅 설정 ---
LOG_FILE = "teacher_inspection.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 토크나이저 로드 (EXAONE 토큰을 읽기 위해 필요함)
logger.info(f"Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

@app.route('/generate', methods=['POST'])
@app.route('/v1/completions', methods=['POST'])
def proxy():
    data = request.get_json()
    
    if data and 'input_ids' in data:
        # 1. 데이터 구조에 따라 안전하게 ids를 가져온다냥
        raw_ids = data['input_ids']
        
        # 만약 리스트의 리스트 형태 [[...]] 라면 첫 번째 리스트를 가져오고,
        # 그냥 리스트 [...] 라면 그대로 사용한다냥.
        if isinstance(raw_ids[0], list):
            original_ids = raw_ids[0]
        else:
            original_ids = raw_ids
        
        # 2. 토큰을 텍스트로 변환
        try:
            decoded_text = tokenizer.decode(original_ids, skip_special_tokens=False)
            logger.info("="*60)
            logger.info(f" [Inspection] Student sent {len(original_ids)} tokens.")
            logger.info(f" [Decoded Text]:\n{decoded_text}")
            logger.info("="*60)
        except Exception as decode_err:
            logger.error(f"Decoding failed: {decode_err}")

if __name__ == '__main__':
    logger.info("Proxy Inspector started on port 13141...")
    app.run(host='0.0.0.0', port=13141, debug=False)