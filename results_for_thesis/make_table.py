import csv

# 各列を正規化するときの基準値（赤字セル）
COL_NORM = [51.55, 33.0, 9.6]  # [gsm8k基準, mbpp基準, ja_mgsm基準]

# 入出力ファイル名
INPUT_FILE = "results.csv"    # 元のCSVファイル名を指定
OUTPUT_FILE = "table.csv"   # 出力CSVファイル名

with open(INPUT_FILE, "r", newline="", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:
    
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    # ヘッダーの読み込みと書き込み
    header = next(reader)  
    # 新しいヘッダーを書き込む
    writer.writerow(['name', 'gsm8k', 'mbpp', 'ja_mgsm', 'avg', 'nor.avg'])
    
    for row in reader:
        # CSVの想定カラム: name, gsm8k, mbpp, ja_mgsm
        name = row[0]
        gsm8k = float(row[1])
        mbpp = float(row[2])
        ja_mgsm = float(row[3])
        
        # 1) 平均 (average)
        avg = round((gsm8k + mbpp + ja_mgsm) / 3.0, 2)  # 小数点2桁
        
        # 2) 正規化後の平均 (normalized_average)
        #    指定の赤字セル値で割った後に3列分を足して3で割る
        gsm8k_norm = gsm8k / COL_NORM[0]
        mbpp_norm  = mbpp  / COL_NORM[1]
        ja_mgsm_norm = ja_mgsm / COL_NORM[2]
        
        norm_avg = round((gsm8k_norm + mbpp_norm + ja_mgsm_norm) / 3.0, 3)  # 小数点3桁
        
        # 出力用に行をまとめる
        new_row = [
            name,
            gsm8k,
            mbpp,
            ja_mgsm,
            avg,
            norm_avg
        ]
        writer.writerow(new_row)

print(f"'{OUTPUT_FILE}' に平均および正規化後の平均を追記して出力しました。")
