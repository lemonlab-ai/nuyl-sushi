# Skill Level 應用端實驗設計（2D / 3D / MANO-compatible）

本文件對應論文「後續階段」：在主模型完成動作分類後，評估手部關鍵點參數是否可用來區分 `skill_level`（`beginner` / `intermediate` / `expert`）。

## 1. 研究問題

1. 使用手部關鍵點序列，是否能穩定預測 skill level？
2. `2D`、`3D image`、`3D world`、`3D MANO-compatible` 哪一種表示法最好？
3. 模式差異是否在 subject-wise 驗證下仍成立（避免同一人洩漏）？

## 2. 需要的資料欄位

請在 `data/meta/video_meta.json` 的每支影片加入：

- `skill_level`: `beginner` / `intermediate` / `expert`

完成 `convert_via_to_master.py` 後，`master_annotations.json` 的每個 video record 會帶有 `skill_level`。

## 3. 內圈可直接執行流程

1. 轉 master（含 skill_level）：

```bash
python scripts/convert_via_to_master.py
```

2. 匯出 skill-level 清單（可盤點資料覆蓋）：

```bash
python scripts/export_skill_level_manifest.py \
  --master-json artifacts/master/master_annotations.json \
  --output-csv artifacts/skill_level/skill_level_manifest.csv
```

3. 批次抽 keypoints（MediaPipe）：

```bash
python scripts/extract_skill_level_keypoints.py \
  --master-json artifacts/master/master_annotations.json \
  --output-dir artifacts/skill_level/keypoints \
  --subset all \
  --drop-unknown-skill
```

4. 跑 skill-level 分類基準（2D/3D/MANO 對比）：

```bash
python scripts/eval_skill_level_from_keypoints.py \
  --index-path artifacts/skill_level/keypoints/skill_level_keypoint_index.csv \
  --coord-modes 2d,3d_image,3d_world,3d_mano \
  --hand-mode both \
  --num-folds 5 \
  --seed 42 \
  --allow-mano-proxy \
  --output-dir outputs/skill_level
```

輸出：

- `outputs/skill_level/skill_level_keypoint_benchmark.json`
- `outputs/skill_level/skill_level_keypoint_benchmark.csv`

## 4. 評估指標與判讀

主指標：

- `Macro F1`（主指標，處理類別不平衡）
- `Balanced Accuracy`
- `Accuracy`

判讀重點：

1. 先看 `macro_f1_mean` 最高的座標模式（2D/3D/MANO）。
2. 再看 `macro_f1_std` 是否過大（穩定性）。
3. 若 `3d_mano` 顯著優於 `2d`，可支持「3D 結構對 skill-level 有辨識價值」。

## 5. 建議論文章節安排

1. 主實驗（既有）：Task A/B 模型效能。
2. 延伸應用：Gesture-based skill-level classification（本文件流程）。
3. Ablation：2D vs 3D image vs 3D world vs 3D MANO-compatible。
4. 討論：
   - 哪些類別最易混淆（例如 beginner vs intermediate）
   - 哪些視角/資料品質會影響判別
   - 是否可作為教學回饋系統前置模組

## 6. 與外圈整合方式

把以下檔案交給外圈即可做下一輪研究規劃：

1. `outputs/skill_level/skill_level_keypoint_benchmark.json`
2. `outputs/skill_level/skill_level_keypoint_benchmark.csv`
3. `artifacts/skill_level/keypoints/skill_level_keypoint_index.csv`

建議外圈優先做：

1. 超參數搜尋（`max_frames`, `hand_mode`, subject fold seed）
2. 更強分類器替換（例如線性層 -> MLP / sequence encoder）
3. 與主模型結果做關聯分析（分類錯誤樣本是否同時為低 skill-level）
