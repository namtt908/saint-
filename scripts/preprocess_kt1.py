import pandas as pd
import numpy as np
from pathlib import Path
import json
import concurrent.futures
import os
from tqdm import tqdm
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import shutil
from datetime import datetime

def process_chunk(chunk_files: list, output_dir: Path, chunk_id: int):
    """Process a chunk of files and save intermediate results."""
    try:
        # Đọc và xử lý chunk
        ddf = dd.read_csv(chunk_files, engine='pyarrow')
        ddf['user_id'] = ddf.map_partitions(lambda df: df.index.map(lambda x: Path(chunk_files[x]).stem))
        
        # Chuyển về pandas và xử lý
        df = ddf.compute()
        
        # Xử lý timestamp và elapsed_time
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['elapsed_time'] = df['elapsed_time'] / 1000
        
        # Lưu kết quả tạm thời
        temp_file = output_dir / f'temp_chunk_{chunk_id}.parquet'
        table = pa.Table.from_pandas(df)
        pq.write_table(table, temp_file)
        
        return True
    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {e}")
        return False

def merge_chunks(output_dir: Path, num_chunks: int):
    """Merge all processed chunks and perform final processing."""
    print("Merging chunks...")
    
    # Đọc tất cả chunks
    chunks = []
    for i in range(num_chunks):
        temp_file = output_dir / f'temp_chunk_{i}.parquet'
        if temp_file.exists():
            df = pd.read_parquet(temp_file)
            chunks.append(df)
    
    # Gộp chunks
    df = pd.concat(chunks, ignore_index=True)
    
    # Sắp xếp và tính toán lag_time
    print("Sorting and calculating lag times...")
    df = df.sort_values(['user_id', 'timestamp'])
    df['lag_time'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    df['lag_time'] = df['lag_time'].fillna(0)
    
    # Tạo concept_id và các trường khác
    print("Creating additional features...")
    df['concept_id'] = df['question_id'].astype('category').cat.codes
    df['part'] = 1
    df['correct'] = 1  # Placeholder
    df['had_explanation'] = 0
    
    # Chọn và sắp xếp cột
    columns = [
        'user_id', 'question_id', 'concept_id', 'part',
        'correct', 'elapsed_time', 'lag_time', 'had_explanation'
    ]
    df = df[columns]
    
    # Lưu kết quả cuối cùng
    print("Saving final results...")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_dir / 'processed_kt1.parquet')
    
    # Lưu metadata
    metadata = {
        'num_users': df['user_id'].nunique(),
        'num_questions': df['question_id'].nunique(),
        'num_concepts': df['concept_id'].nunique(),
        'num_interactions': len(df),
        'avg_sequence_length': df.groupby('user_id').size().mean()
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Xóa các file tạm
    print("Cleaning up temporary files...")
    for i in range(num_chunks):
        temp_file = output_dir / f'temp_chunk_{i}.parquet'
        if temp_file.exists():
            temp_file.unlink()

def main():
    # Paths
    data_dir = Path('data/raw/KT1')
    output_dir = Path('data/processed/kt1')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo thư mục tạm
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    # Lấy danh sách file
    print("Getting list of files...")
    user_files = list(data_dir.glob('u*.csv'))
    total_files = len(user_files)
    print(f"Found {total_files} user files")
    
    # Chia thành các chunks nhỏ hơn
    chunk_size = 10000  # Xử lý 10k file mỗi lần
    num_chunks = (total_files + chunk_size - 1) // chunk_size
    
    # Xử lý từng chunk
    print(f"Processing {num_chunks} chunks...")
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_files)
        chunk_files = user_files[start_idx:end_idx]
        
        print(f"\nProcessing chunk {i+1}/{num_chunks} ({len(chunk_files)} files)...")
        process_chunk(chunk_files, temp_dir, i)
    
    # Gộp và xử lý cuối cùng
    merge_chunks(output_dir, num_chunks)
    
    # Xóa thư mục tạm
    shutil.rmtree(temp_dir)
    
    print("Done!")

if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Total processing time: {end_time - start_time}") 