# Gesture Coach 網頁 Demo 使用指南（中文）

這份文件是給你快速操作 `apps/streamlit_app.py` 裡的 `Gesture Coach (MVP)` 分頁。

## 1. 功能總覽

Gesture Coach 支援：

1. 兩段手部動作序列比對（DTW）
2. 比對座標模式：
   - `2D (x,y)`
   - `3D image (x,y,z)`
   - `3D world (x,y,z)`
   - `3D MANO-compatible joints (x,y,z)`
3. 兩種輸入來源：
   - 直接上傳 keypoint JSON
   - 上傳兩段影片，用 MediaPipe 先抽 keypoints 再比對

---

## 2. 啟動前準備

建議使用專案 conda 環境（Windows）：

```powershell
C:\Users\Public\conda-envs\ntnu-sushi\python.exe -m pip install -r requirements.txt
```

> `requirements.txt` 已包含 `mediapipe`。  
> 若只用 JSON 比對，不抽影片 keypoint，理論上不一定需要 MediaPipe。

---

## 3. 啟動網頁

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_streamlit.ps1 -EnvName ntnu-sushi
```

打開瀏覽器：

```text
http://127.0.0.1:8501
```

進入第三個分頁：`Gesture Coach (MVP)`。

---

## 4. 操作流程

## A. 直接上傳 JSON 比對

1. `Input mode` 選 `Upload keypoint JSON`
2. 上傳 `Reference keypoint JSON`
3. 上傳 `Test keypoint JSON`
4. 設定 `Hand mode`（left/right/both）
5. 設定 `Coordinate mode`
6. 點 `Run Gesture Compare`

結果會顯示：

1. `Gesture similarity (0-100)`
2. `DTW distance`
3. `mean/p90 pair distance`
4. 對齊距離折線圖

## B. 上傳影片後抽 keypoint 再比對

1. `Input mode` 選 `Extract from videos (MediaPipe)`
2. 上傳參考影片（Reference video）
3. 上傳測試影片（Test video）
4. 設定抽取參數（`extract_max_frames`、`min_detection_confidence`、`min_tracking_confidence`）
5. 設定 `Coordinate mode`
6. 點 `Extract + Compare`

系統會：

1. 先抽取 keypoints
2. 存成 JSON（可下載）
3. 自動比對並輸出分數

---

## 5. MANO-compatible 模式怎麼用

選 `Coordinate mode = 3D MANO-compatible joints (x,y,z)` 時：

1. 若 JSON 已有 `left_mano` / `right_mano`，直接比對
2. 若沒有，可勾選 `Allow MANO proxy from world_3d if MANO joints missing`
3. 勾選後會用 `left_world_3d/right_world_3d`（或 `left_3d/right_3d`）暫代 MANO joints

注意：

1. 這是 MANO 資料介面/流程整合，不是完整 MANO mesh 回歸訓練
2. 若你要真正 MANO 參數（pose/shape），需接外部 MANO 回歸模型

---

## 6. JSON 格式建議

最外層：

```json
{
  "fps": 30.0,
  "frames": [
    {
      "frame_idx": 0,
      "t_sec": 0.0,
      "left": [[x, y, conf], ...],
      "right": [[x, y, conf], ...],
      "left_3d": [[x, y, z, conf], ...],
      "right_3d": [[x, y, z, conf], ...],
      "left_world_3d": [[x, y, z, conf], ...],
      "right_world_3d": [[x, y, z, conf], ...],
      "left_mano": [[x, y, z, conf], ...],
      "right_mano": [[x, y, z, conf], ...]
    }
  ]
}
```

備註：

1. 2D 舊格式 `[x, y, conf]` 仍相容
2. 3D 建議用 `[x, y, z, conf]`
3. 每隻手預設 21 個 keypoints

---

## 7. 常見問題

## Q1: 出現 `mediapipe` import error

安裝套件：

```powershell
C:\Users\Public\conda-envs\ntnu-sushi\python.exe -m pip install mediapipe
```

## Q2: MANO-compatible 模式報缺 `left_mano/right_mano`

1. 先勾 `Allow MANO proxy from world_3d...`
2. 或改用 `3D world (x,y,z)` 模式

## Q3: 分數波動大

1. 先固定 `Hand mode=both`
2. 降低抽取畫面變異（拍攝角度/距離）
3. 先用同一人同動作做 baseline 校正
