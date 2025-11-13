# Git 版本管理指南 - 腦動脈瘤檢測訓練系統

## 目錄
- [專案概述](#專案概述)
- [前置準備](#前置準備)
- [初始化設定](#初始化設定)
- [日常工作流程](#日常工作流程)
- [Git 基本操作](#git-基本操作)
- [常見問題處理](#常見問題處理)
- [.gitignore 設定](#gitignore-設定)
- [快速參考卡](#快速參考卡)

---

## 專案概述

### 目標
建立 GitHub 版本管理系統，追蹤腦動脈瘤檢測與分割系統的訓練程式碼開發歷程。

### 專案資訊
- **專案名稱**：腦動脈瘤檢測與分割系統
- **GitHub 倉庫**：https://github.com/yioyy/aneurysm_training_code.git
- **分支策略**：單一 `main` 分支（簡化管理）
- **專案位置**：`C:\Users\user\Desktop\nnUNet\nnResUNet-github\`

### 專案結構
```
nnResUNet-github/
├── nnResUNet/                  # nnU-Net 核心程式碼
├── model_weights/              # 模型權重（不上傳 Git）
├── nnUNet_results/             # 訓練結果（不上傳 Git）
├── nnUNet_preprocessed/        # 前處理資料
├── nnUNet_raw/                 # 原始資料集
├── pipeline_aneurysm_torch.py  # PyTorch Pipeline
├── gpu_aneurysm.py            # GPU 推理腳本
└── README.md                   # 專案說明文件
```

---

## 前置準備

### 1. 確認 GitHub 倉庫已建立

- **倉庫 URL**：https://github.com/yioyy/aneurysm_training_code.git
- 倉庫目前為空，準備推送本地程式碼

### 2. 安裝必要工具

- **Git for Windows**：從 [git-scm.com](https://git-scm.com/download/win) 下載安裝
- **PowerShell** 或 **Git Bash**：Windows 內建或隨 Git 安裝
- **文字編輯器**：VS Code / Notepad++（建議 VS Code）

### 3. 設定 Git 使用者資訊

首次使用 Git 需要設定您的身分：

```powershell
# 設定使用者名稱
git config --global user.name "您的名字"

# 設定電子郵件（建議使用 GitHub 註冊的 email）
git config --global user.email "your.email@example.com"

# 驗證設定
git config --list
```

---

## 初始化設定

### 步驟 1：建立 .gitignore 檔案

在上傳程式碼前，先建立 `.gitignore` 避免上傳不必要的大型檔案。

```powershell
# 進入專案資料夾
cd "C:\Users\user\Desktop\nnUNet\nnResUNet-github"

# 建立 .gitignore
notepad .gitignore
```

複製以下內容到 `.gitignore`：

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
*.eggs/
dist/
build/

# 日誌檔案
log/
logs/
*.log

# 大型模型檔案（太大不適合 Git）
model_weights/
nnUNet_results/*/fold_*/

# 處理結果和暫存檔
process/
process_*/
mySEG*/
*.tmp
*.bak
*.orig

# Jupyter Notebook
.ipynb_checkpoints/

# IDE 設定檔
.vscode/
.idea/
*.swp
*.swo
*~

# 作業系統產生的檔案
.DS_Store
Thumbs.db
desktop.ini

# 測試資料
test_data/
temp/
example_output/*/

# 大型資料集（建議使用符號連結或外部儲存）
# nnUNet_raw/
# nnUNet_preprocessed/
```

**重要提醒：**
- 模型權重檔案通常很大（數百 MB 到數 GB），不適合放在 Git
- 建議將模型權重存放在雲端硬碟或伺服器，在 README 說明下載方式

### 步驟 2：初始化 Git 倉庫

```powershell
# 確認在專案資料夾
cd "C:\Users\user\Desktop\nnUNet\nnResUNet-github"

# 初始化 Git 倉庫
git init

# 查看當前狀態
git status
```

### 步驟 3：新增檔案到 Git

```powershell
# 將所有檔案加入暫存區（會自動排除 .gitignore 中的檔案）
git add .

# 查看將要提交的檔案
git status

# 提交到本地倉庫
git commit -m "Initial commit: 腦動脈瘤檢測訓練系統初始版本"
```

### 步驟 4：連接到 GitHub 並推送

```powershell
# 將分支重命名為 main（如果目前是 master）
git branch -M main

# 連接到 GitHub 遠端倉庫
git remote add origin https://github.com/yioyy/aneurysm_training_code.git

# 推送到 GitHub
git push -u origin main
```

**首次推送可能需要登入：**
- 會彈出 GitHub 登入視窗
- 輸入您的 GitHub 帳號密碼或使用 Token

✅ **完成！現在程式碼已經在 GitHub 上了！**

可以到 https://github.com/yioyy/aneurysm_training_code 查看。

---

## 日常工作流程

### 情境 1：修改程式碼後提交

```powershell
# 1. 查看有哪些檔案被修改
git status

# 2. 查看具體修改內容
git diff

# 3. 將修改的檔案加入暫存區
git add pipeline_aneurysm_torch.py gpu_aneurysm.py

# 或者一次加入所有修改
git add .

# 4. 提交到本地倉庫（附上清楚的說明）
git commit -m "修改動脈瘤檢測閾值，提升靈敏度"

# 5. 推送到 GitHub
git push origin main
```

### 情境 2：每日工作開始前更新程式碼

如果有多人協作或在不同電腦工作：

```powershell
# 從 GitHub 拉取最新程式碼
git pull origin main
```

### 情境 3：查看修改歷史

```powershell
# 查看提交歷史（圖形化顯示）
git log --oneline --graph

# 查看最近 10 筆提交
git log -10

# 查看某個檔案的修改歷史
git log --follow pipeline_aneurysm_torch.py

# 查看某次提交的詳細內容
git show <commit_id>
```

### 情境 4：比較不同版本

```powershell
# 比較工作目錄與上次提交的差異
git diff

# 比較暫存區與上次提交的差異
git diff --staged

# 比較兩個 commit 之間的差異
git diff <commit_id_1> <commit_id_2>

# 比較特定檔案在兩個版本間的差異
git diff <commit_id_1> <commit_id_2> -- pipeline_aneurysm_torch.py
```

### 情境 5：撤銷修改

```powershell
# 放棄工作目錄中某個檔案的修改（危險！無法復原）
git checkout -- pipeline_aneurysm_torch.py

# 放棄所有未提交的修改（危險！）
git checkout .

# 取消已加入暫存區的檔案（但保留修改）
git reset HEAD pipeline_aneurysm_torch.py

# 回退到上一個 commit（保留修改在工作目錄）
git reset --soft HEAD~1

# 回退到上一個 commit（刪除所有修改，危險！）
git reset --hard HEAD~1
```

### 情境 6：回到舊版本

```powershell
# 查看提交歷史，找到要回退的 commit ID
git log --oneline

# 方法 1：建立新分支指向舊版本（安全，推薦）
git checkout <commit_id> -b backup-version

# 方法 2：直接回退（危險！會刪除之後的所有 commit）
git reset --hard <commit_id>
git push origin main --force
```

### 情境 7：提交訊息範例

好的提交訊息應該清楚說明「做了什麼」和「為什麼」：

```bash
# ✅ 好的範例
git commit -m "修正 Dataset127 前處理時的記憶體溢出問題"
git commit -m "新增早停機制，避免過度訓練"
git commit -m "更新 README：補充 TWCC 環境設定說明"
git commit -m "優化血管分割後處理，移除小於 100 voxel 的雜訊"

# ❌ 不好的範例
git commit -m "update"
git commit -m "fix bug"
git commit -m "修改程式"
git commit -m "test"
```

---

## Git 基本操作

### 常用指令總覽

| 操作 | 指令 | 說明 |
|------|------|------|
| 查看狀態 | `git status` | 查看檔案修改狀態 |
| 查看差異 | `git diff` | 查看具體修改內容 |
| 加入暫存 | `git add <file>` | 將檔案加入暫存區 |
| 加入全部 | `git add .` | 加入所有修改的檔案 |
| 提交 | `git commit -m "訊息"` | 提交到本地倉庫 |
| 推送 | `git push origin main` | 推送到 GitHub |
| 拉取 | `git pull origin main` | 從 GitHub 更新 |
| 查看歷史 | `git log --oneline` | 查看提交歷史 |
| 放棄修改 | `git checkout -- <file>` | 放棄工作目錄的修改 |
| 回退版本 | `git reset --hard <commit>` | 回退到指定版本 |

### Git 工作流程圖

```
工作目錄           暫存區              本地倉庫           遠端倉庫
(Working)  →  (Staging)  →  (Local Repo)  →  (GitHub)
             git add         git commit      git push

                                           ← 
                                         git pull
```

### 檔案狀態說明

```powershell
git status
```

可能的輸出：

```
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   pipeline_aneurysm_torch.py    # 已暫存，準備提交

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   gpu_aneurysm.py               # 已修改，未暫存

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        new_script.py                             # 新檔案，未追蹤
```

---

## 常見問題處理

### Q1：如何忽略已經追蹤的檔案？

如果某個檔案已經被 Git 追蹤，後來才加入 `.gitignore`：

```powershell
# 從 Git 追蹤中移除（但保留檔案）
git rm --cached model_weights/large_model.h5

# 提交變更
git commit -m "移除大型模型檔案的 Git 追蹤"
git push origin main
```

### Q2：推送時出現 "rejected" 錯誤？

```
! [rejected]        main -> main (fetch first)
error: failed to push some refs
```

**原因**：遠端有新的提交，本地版本落後。

**解決方案**：

```powershell
# 方法 1：拉取後合併（推薦）
git pull origin main
# 如果有衝突，手動解決後：
git add .
git commit -m "合併遠端更新"
git push origin main

# 方法 2：強制推送（危險！會覆蓋遠端版本）
git push origin main --force
```

### Q3：如何處理合併衝突？

當 `git pull` 出現衝突時：

```powershell
# 1. Git 會告訴您哪些檔案有衝突
git status

# 2. 編輯衝突檔案
notepad pipeline_aneurysm_torch.py
```

衝突檔案會顯示：

```python
<<<<<<< HEAD (本地版本)
threshold = 0.5
=======
threshold = 0.6
>>>>>>> origin/main (遠端版本)
```

手動選擇保留哪個版本，或合併兩者：

```python
threshold = 0.55  # 折衷方案
```

```powershell
# 3. 標記為已解決
git add pipeline_aneurysm_torch.py

# 4. 完成合併
git commit -m "解決 threshold 衝突"
git push origin main
```

### Q4：誤刪檔案如何復原？

```powershell
# 復原單一檔案（從上次 commit）
git checkout HEAD -- pipeline_aneurysm_torch.py

# 復原所有刪除的檔案
git checkout HEAD -- .
```

### Q5：如何查看某個檔案的舊版本內容？

```powershell
# 查看特定 commit 的檔案內容
git show <commit_id>:pipeline_aneurysm_torch.py

# 將舊版本的檔案還原到工作目錄
git checkout <commit_id> -- pipeline_aneurysm_torch.py
```

### Q6：如何刪除最後一次 commit？

```powershell
# 保留修改，只刪除 commit
git reset --soft HEAD~1

# 完全刪除 commit 和修改（危險！）
git reset --hard HEAD~1

# 如果已經推送到 GitHub，需要強制推送
git push origin main --force
```

### Q7：大型檔案上傳失敗？

GitHub 單一檔案限制為 100 MB。

**解決方案 1：使用 Git LFS（大型檔案儲存）**

```powershell
# 安裝 Git LFS
git lfs install

# 追蹤大型檔案類型
git lfs track "*.h5"
git lfs track "*.pth"

# 提交 .gitattributes
git add .gitattributes
git commit -m "設定 Git LFS 追蹤模型檔案"

# 正常提交大型檔案
git add model_weights/
git commit -m "新增模型權重"
git push origin main
```

**解決方案 2：不上傳模型檔案（推薦）**

在 `.gitignore` 排除模型檔案，改用以下方式分享：
- Google Drive / OneDrive 連結
- 機構內部伺服器
- Hugging Face Model Hub

在 README 說明下載方式。

### Q8：如何複製專案到其他電腦？

```powershell
# 在新電腦上執行
git clone https://github.com/yioyy/aneurysm_training_code.git

# 進入專案資料夾
cd aneurysm_training_code

# 查看所有檔案
ls
```

---

## .gitignore 設定

完整的 `.gitignore` 範本（針對本專案優化）：

```gitignore
# ====================================
# Python 相關
# ====================================
__pycache__/
*.pyc
*.pyo
*.pyd
*.so
*.egg
*.egg-info/
dist/
build/
*.whl

# ====================================
# 深度學習模型與資料
# ====================================
# 大型模型檔案
model_weights/**/*.h5
model_weights/**/*.pth
model_weights/**/*.pb
model_weights/**/*.ckpt
*.weights
*.caffemodel

# 訓練結果（保留 plans.json 等設定檔）
nnUNet_results/*/fold_*/
nnUNet_results/**/*.pth
nnUNet_results/**/*.pkl
nnUNet_results/**/validation_raw/
nnUNet_results/**/validation_raw_postprocessed/

# 前處理資料（太大）
nnUNet_preprocessed/**/nnUNetData_plans_*/
*.npz
*.npy

# 原始資料集（通常由外部管理）
# 取消註解以排除：
# nnUNet_raw/

# ====================================
# 日誌與暫存檔
# ====================================
log/
logs/
*.log
*.out
*.err
tensorboard_logs/

# 處理結果
process/
process_*/
process_example/
mySEG*/
temp/
tmp/

# ====================================
# Jupyter Notebook
# ====================================
.ipynb_checkpoints/
*-checkpoint.ipynb

# ====================================
# IDE 與編輯器
# ====================================
# VS Code
.vscode/
*.code-workspace

# PyCharm
.idea/
*.iml

# Vim
*.swp
*.swo
*~

# Sublime Text
*.sublime-project
*.sublime-workspace

# ====================================
# 作業系統
# ====================================
# macOS
.DS_Store
.AppleDouble
.LSOverride

# Windows
Thumbs.db
desktop.ini
$RECYCLE.BIN/

# Linux
*~
.directory

# ====================================
# 測試與範例輸出
# ====================================
test_data/
example_output/*/
output_*/
predictions/
inference_results/

# ====================================
# 其他
# ====================================
*.tmp
*.bak
*.orig
*.rej
.cache/
```

### 如何更新 .gitignore

```powershell
# 1. 編輯 .gitignore
notepad .gitignore

# 2. 如果有已追蹤的檔案需要排除
git rm -r --cached model_weights/
git rm -r --cached nnUNet_results/

# 3. 提交變更
git add .gitignore
git commit -m "更新 .gitignore：排除大型模型檔案"
git push origin main
```

---

## 快速參考卡

### 每日工作流程

```powershell
# 1. 開始工作前：更新程式碼
cd "C:\Users\user\Desktop\nnUNet\nnResUNet-github"
git pull origin main

# 2. 進行開發工作
# ... 修改程式碼 ...

# 3. 查看修改
git status
git diff

# 4. 提交變更
git add .
git commit -m "清楚的提交訊息"

# 5. 推送到 GitHub
git push origin main
```

### 緊急救援指令

```powershell
# 放棄所有未提交的修改（危險！）
git checkout .

# 查看最近 10 次提交
git log -10 --oneline

# 回到上一個版本
git reset --hard HEAD~1

# 強制與遠端同步（放棄本地所有修改）
git fetch origin
git reset --hard origin/main
```

### 實用別名設定（選用）

讓常用指令更簡短：

```powershell
# 設定別名
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.ci commit
git config --global alias.br branch
git config --global alias.lg "log --oneline --graph --all"

# 使用別名
git st      # 等同於 git status
git lg      # 等同於 git log --oneline --graph --all
```

### Git 命令速查表

| 需求 | 指令 |
|------|------|
| 查看狀態 | `git status` |
| 查看差異 | `git diff` |
| 暫存檔案 | `git add <file>` |
| 暫存全部 | `git add .` |
| 提交 | `git commit -m "message"` |
| 推送 | `git push origin main` |
| 拉取 | `git pull origin main` |
| 查看歷史 | `git log --oneline --graph` |
| 查看遠端 | `git remote -v` |
| 放棄單一檔案修改 | `git checkout -- <file>` |
| 放棄所有修改 | `git checkout .` |
| 取消暫存 | `git reset HEAD <file>` |
| 回退一個 commit | `git reset --soft HEAD~1` |
| 查看分支 | `git branch` |
| 切換分支 | `git checkout <branch>` |
| 建立分支 | `git checkout -b <branch>` |

---

## 進階技巧

### 使用 Git Tags 標記重要版本

```powershell
# 建立標籤（例如：發表論文時的版本）
git tag -a v1.0 -m "RSNA 2025 競賽版本"

# 推送標籤到 GitHub
git push origin v1.0

# 查看所有標籤
git tag

# 檢出特定標籤
git checkout v1.0
```

### 使用 Git Stash 暫存工作

當需要臨時切換任務但不想提交未完成的工作：

```powershell
# 暫存當前修改
git stash

# 查看暫存列表
git stash list

# 恢復最近的暫存
git stash pop

# 恢復特定暫存
git stash apply stash@{0}

# 刪除暫存
git stash drop stash@{0}
```

### 查看誰修改了某行程式碼

```powershell
# 顯示每一行的最後修改者和時間
git blame pipeline_aneurysm_torch.py

# 查看特定行範圍
git blame -L 100,150 pipeline_aneurysm_torch.py
```

---

## 協作開發建議

### 提交規範

建議使用統一的提交訊息格式：

```
[類型] 簡短描述

詳細說明（選用）

相關 Issue/PR：#123
```

**類型**：
- `[新增]`：新增功能
- `[修正]`：修復 bug
- `[優化]`：效能或程式碼優化
- `[文檔]`：文件更新
- `[重構]`：程式碼重構
- `[測試]`：新增或修改測試

**範例**：

```bash
git commit -m "[新增] 實作動態學習率調整機制

新增 CosineAnnealingLR 調度器，在訓練後期降低學習率
以提升模型收斂穩定性。

相關 Issue: #45"
```

### 定期備份

雖然 GitHub 已經是備份，但建議：

```powershell
# 定期將整個倉庫打包備份
cd C:\Users\user\Desktop\nnUNet
tar -czf nnResUNet-backup-$(Get-Date -Format "yyyyMMdd").tar.gz nnResUNet-github/
```

---

## 結語

### 核心概念提醒

✅ **正確理解**：
- Git 會追蹤所有程式碼的修改歷史
- 每次 commit 都是一個可回溯的版本
- `.gitignore` 可避免上傳不必要的大型檔案
- 頻繁提交小變更比一次提交大變更更好
- 清楚的提交訊息是未來的您會感謝的禮物

❌ **常見誤解**：
- ❌ Git 會自動備份所有檔案（需要手動 commit 和 push）
- ❌ push 後無法修改（可以，但要謹慎使用 `--force`）
- ❌ .gitignore 可以忽略已追蹤的檔案（需要先 `git rm --cached`）
- ❌ 可以隨意 `git reset --hard`（會永久遺失修改）

### 最佳實踐

1. **每日至少一次 commit**：養成提交習慣
2. **寫清楚的提交訊息**：說明「做了什麼」和「為什麼」
3. **push 前先 pull**：避免衝突
4. **不要上傳大型檔案**：使用 `.gitignore` 和 Git LFS
5. **定期檢視歷史**：`git log` 查看開發軌跡
6. **善用分支（進階）**：實驗新功能時可建立分支

### 學習資源

- [Git 官方文件](https://git-scm.com/doc)（中文）
- [GitHub 指南](https://docs.github.com/zh)
- [Pro Git 電子書](https://git-scm.com/book/zh/v2)（免費）
- [Learn Git Branching](https://learngitbranching.js.org/?locale=zh_TW)（互動式教學）

---

## 專案資訊

- **專案名稱**：腦動脈瘤檢測與分割訓練系統
- **GitHub 倉庫**：https://github.com/yioyy/aneurysm_training_code.git
- **專案位置**：`C:\Users\user\Desktop\nnUNet\nnResUNet-github\`
- **分支策略**：單一 `main` 分支

**記住：Git 是您的開發夥伴，能幫助您追蹤每一次進步！**

---

最後更新日期：2025-11-13
