import pandas as pd
import glob
import os

# 1. 파일 경로 설정 (다운로드 받은 폴더 경로)
input_dir = '/root/changhun.lee/models/OpenThoughts3-1.2M-70k-opd/data'
output_file = '/root/changhun.lee/models/OpenThoughts3_converted.parquet'

# 2. 파일 목록 확보
all_files = glob.glob(os.path.join(input_dir, "*.parquet"))
all_files.sort()

def transform_with_instruction(row):
    """
    앞뒤에 고정 지시문을 추가하고 리스트 형식으로 반환합니다.
    """
    convs = row['conversations']
    original_problem = ""
    
    for msg in convs:
        if msg['from'] == 'human':
            original_problem = msg['value']
            break
    
    # 지시문 조립 (앞부분 지시문 + 문제 + 뒷부분 지시문)
    prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n"
    suffix = "\n\nRemember to put your answer on its own line after \"Answer:\"."
    
    full_content = f"{prefix}{original_problem}{suffix}"
    
    return [
        {
            "content": full_content,
            "role": "user"
        }
    ]

# 3. 로드 및 변환 작업
dfs = []
for f in all_files:
    df = pd.read_parquet(f)
    
    # 컬럼명 변경: answer -> label
    if 'answer' in df.columns:
        df = df.rename(columns={'answer': 'label'})
    
    # prompt 컬럼 생성
    df['prompt'] = df.apply(transform_with_instruction, axis=1)
    
    dfs.append(df[['prompt', 'label']])

# 4. 통합 및 저장
final_df = pd.concat(dfs, ignore_index=True)
final_df.to_parquet(output_file, index=False)

# 5. 첫 번째 데이터의 Prompt Content와 Label 상세 출력
print("="*80)
print(" [데이터 변환 완료 및 샘플 확인] ")
print("="*80)
print(f"총 통합 데이터 수: {len(final_df)}")
print(f"저장 경로: {output_file}")
print("-" * 80)

# 첫 번째 행 데이터 추출
first_row = final_df.iloc[0]
first_prompt_content = first_row['prompt'][0]['content']
first_label = first_row['label']

print("▶ [첫 번째 데이터: Prompt Content]")
print("-" * 80)
print(first_prompt_content)
print("-" * 80)
print(f"▶ [첫 번째 데이터: Label]\n{first_label}")
print("="*80)