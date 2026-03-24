import os
import requests
import mercantile
import time
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import multiprocessing
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 1. 配置参数 ---
CSV_PATH = "/data/xulj/dinov3-salad/sateliteimage/cities.csv"
SAVE_ROOT = "/data/xulj/dinov3-salad/sateliteimage/Global_Satellites"
LOG_PATH = "/data/xulj/dinov3-salad/sateliteimage/processed_ids.txt"
TARGET_ZOOM = 19
TARGET_SIDE_PX = 4096
NUM_PROCESSES = 4          
MAX_THREADS = 5            
TOTAL_GOAL = 14000         # 🎯 你的硬指标：总共只存 14000 张

proxies = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def check_proxy_health():
    try:
        requests.get("https://www.google.com", proxies=proxies, timeout=3, verify=False)
        return True
    except:
        return False

def get_session():
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.verify = False
    return session

def download_single_tile(tile_info):
    url, i, j, session = tile_info
    try:
        time.sleep(random.uniform(0.02, 0.08))
        r = session.get(url, headers=headers, proxies=proxies, timeout=8) 
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content))
            if np.array(img.convert('L')).std() < 3.0: return None
            return (i, j, img)
    except:
        pass
    return None

def worker_process(df_chunk, worker_id, shared_counter, exit_flag):
    if len(df_chunk) == 0: return
    session = get_session()
    num_tiles_edge = TARGET_SIDE_PX // 256
    expected_total = num_tiles_edge ** 2
    
    pbar = tqdm(df_chunk.iterrows(), total=len(df_chunk), 
                desc=f"Worker {worker_id}", position=worker_id, leave=True)
    
    with open(LOG_PATH, "a") as log_file:
        for _, row in pbar:
            # 1. 检查全局停止标志
            if exit_flag.value: break
            
            # 2. 网络健康检查
            while not check_proxy_health():
                if exit_flag.value: break # 等待期间也要查标志
                if worker_id == 0:
                    print(f"\n🚨 [NETWORK ERROR] 代理断开，正在重连...")
                time.sleep(10)

            target_id = str(row['id'])
            anchor_t = mercantile.tile(float(row['longitude']), float(row['latitude']), TARGET_ZOOM)
            anchor_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{TARGET_ZOOM}/{anchor_t.y}/{anchor_t.x}"
            
            is_valid = False
            try:
                resp = session.get(anchor_url, headers=headers, proxies=proxies, timeout=5)
                if resp.status_code == 200:
                    anchor_img = Image.open(BytesIO(resp.content))
                    if np.array(anchor_img.convert('L')).std() >= 5.5:
                        is_valid = True
            except:
                pass

            if not is_valid:
                log_file.write(f"{target_id}\n")
                log_file.flush()
                continue

            # 3. 下载逻辑
            min_x, min_y = anchor_t.x - 8, anchor_t.y - 8
            tile_tasks = []
            for i in range(num_tiles_edge):
                for j in range(num_tiles_edge):
                    if i == 8 and j == 8: continue
                    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{TARGET_ZOOM}/{min_y+j}/{min_x+i}"
                    tile_tasks.append((url, i, j, session))

            canvas = Image.new('RGB', (TARGET_SIDE_PX, TARGET_SIDE_PX))
            canvas.paste(anchor_img, (8 * 256, 8 * 256))
            tiles_map = {(8, 8): True}

            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                futures = [executor.submit(download_single_tile, task) for task in tile_tasks]
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        tiles_map[(res[0], res[1])] = True
                        canvas.paste(res[2], (res[0] * 256, res[1] * 256))

            # 4. 保存与计数
            if len(tiles_map) == expected_total:
                save_path = os.path.join(SAVE_ROOT, f"{target_id}_z{TARGET_ZOOM}.png")
                canvas.save(save_path, "PNG")
                
                # 更新全局计数器
                with shared_counter.get_lock():
                    shared_counter.value += 1
                    current_total = shared_counter.value
                
                log_file.write(f"{target_id}\n")
                log_file.flush()
                pbar.set_postfix({"Total_Saved": current_total})
                
                # 检查是否达到最终目标
                if current_total >= TOTAL_GOAL:
                    with exit_flag.get_lock():
                        exit_flag.value = True
                    if worker_id == 0:
                        print(f"\n✨ 目标达成！已收集满 {TOTAL_GOAL} 张图片。")
                    break

def main():
    if not os.path.exists(SAVE_ROOT): os.makedirs(SAVE_ROOT, exist_ok=True)
    if not check_proxy_health():
        print("❌ 代理不可用，请检查 SSH 隧道！"); sys.exit(1)

    # 加载历史数据
    processed_ids = set()
    existing_img_count = 0
    if os.path.exists(SAVE_ROOT):
        for f in os.listdir(SAVE_ROOT):
            if f.endswith('.png'):
                processed_ids.add(int(f.split('_')[0]))
                existing_img_count += 1
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            for line in f:
                if line.strip(): processed_ids.add(int(line.strip()))

    # 初始化共享变量
    shared_counter = multiprocessing.Value('i', existing_img_count)
    exit_flag = multiprocessing.Value('b', False)

    if existing_img_count >= TOTAL_GOAL:
        print(f"🎉 任务早已完成，已有 {existing_img_count} 张图！"); return

    df = pd.read_csv(CSV_PATH)
    df = df[~df['id'].isin(processed_ids)]
    print(f"✅ 已处理: {len(processed_ids)} | 已保存: {existing_img_count} | 目标: {TOTAL_GOAL}")

    df = df.sample(frac=1).reset_index(drop=True)
    df_chunks = np.array_split(df, NUM_PROCESSES)
    
    processes = []
    for i in range(NUM_PROCESSES):
        p = multiprocessing.Process(target=worker_process, args=(df_chunks[i], i, shared_counter, exit_flag))
        p.start()
        processes.append(p)

    for p in processes: p.join()
    print("🏁 程序运行结束。")

if __name__ == "__main__":
    main()