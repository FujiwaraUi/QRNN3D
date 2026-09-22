import os

paths = (
    "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_complex.txt", 
    "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/ICVL_test_gauss.txt",
    "/mnt/data/User/Code/research/Methods/06_QRNN3D/QRNN3D/src/ICVL_train.txt"
)

# ファイルごとに個別でデータを格納する辞書（キー: ファイル名, 値: セット）
file_contents = {}

# 1. 各ファイルを個別のセットとして読み込む
for path in paths:
    # パスからファイル名（例: ICVL_train.txt）を取得してキーにする
    file_name = os.path.basename(path)
    
    file_contents[file_name] = set()
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.strip()
            if cleaned_line:
                file_contents[file_name].add(cleaned_line)

# 扱いやすいように各セットを変数に代入
train_set = file_contents["ICVL_train.txt"]
complex_set = file_contents["ICVL_test_complex.txt"]
gauss_set = file_contents["ICVL_test_gauss.txt"]

# 2. 重複（共通要素）を判定する

# ① train と test_complex の比較
dup_complex = train_set & complex_set  # & で共通要素を抽出
print(f"【ICVL_train と ICVL_test_complex の重複】")
if dup_complex:
    print(f"❌ 重複が {len(dup_complex)} 件あります。")
    print("重複している要素:", dup_complex)
else:
    print("✅ 重複はありません。")

print("-" * 40)

# ② train と test_gauss の比較
dup_gauss = train_set & gauss_set
print(f"【ICVL_train と ICVL_test_gauss.txt の重複】")
if dup_gauss:
    print(f"❌ 重複が {len(dup_gauss)} 件あります。")
    print("重複している要素:", dup_gauss)
else:
    print("✅ 重複はありません。")

