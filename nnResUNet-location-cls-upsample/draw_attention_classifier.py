"""
繪製 AttentionClassifier (Cross-Attention 分類頭) 架構圖
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['font.monospace'] = ['Microsoft JhengHei', 'Consolas', 'DejaVu Sans Mono']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis('off')
ax.set_aspect('equal')

# --- 顏色定義 ---
C_INPUT   = '#4A90D9'   # 輸入
C_OP      = '#F5A623'   # 操作
C_ATTN    = '#E74C3C'   # Attention 核心
C_NORM    = '#2ECC71'   # Norm / Dropout
C_LINEAR  = '#9B59B6'   # Linear
C_PARAM   = '#1ABC9C'   # 可學習參數
C_ARROW   = '#333333'

def draw_box(cx, cy, w, h, text, color, fontsize=11, text_color='white'):
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor='#333333', linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, wrap=True)

def draw_arrow(x1, y1, x2, y2, text='', color=C_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, text, fontsize=9, color='#555555', va='center')

def draw_dashed_arrow(x1, y1, x2, y2, text='', color=C_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, linestyle='dashed'))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, text, fontsize=9, color='#555555', va='center')

# ============================================================
# 標題
# ============================================================
ax.text(7, 21.3, 'AttentionClassifier 架構圖', ha='center', va='center',
        fontsize=18, fontweight='bold', color='#222222')
ax.text(7, 20.8, '(RSNA 風格 Cross-Attention 分類頭)', ha='center', va='center',
        fontsize=12, color='#666666')

# ============================================================
# 1. Encoder Bottleneck 輸入
# ============================================================
y = 19.8
draw_box(7, y, 5.5, 0.7, 'Encoder Bottleneck (skips[-1])\n[B, 512, D, H, W]', C_INPUT, fontsize=10)

# ============================================================
# 2. Flatten 空間維度
# ============================================================
y_flatten = 18.5
draw_arrow(7, 19.8 - 0.35, 7, y_flatten + 0.35)
draw_box(7, y_flatten, 4.5, 0.7, 'Flatten 空間維度\n[B, 512, D*H*W]', C_OP, fontsize=10)

# ============================================================
# 3. Permute
# ============================================================
y_perm = 17.2
draw_arrow(7, y_flatten - 0.35, 7, y_perm + 0.35)
draw_box(7, y_perm, 4.5, 0.7, 'Permute → [S, B, D]\nS=D*H*W, D=embed_dim', C_OP, fontsize=10)

# ============================================================
# 4. 可學習查詢 + Cross-Attention (核心)
# ============================================================
y_query = 15.3
y_attn = 15.3

# 查詢向量 (左側)
draw_box(3, y_query, 3.5, 1.0,
         'Class Query\n(可學習參數)\n[Q, embed_dim]', C_PARAM, fontsize=10)

# Cross-Attention (中間偏右)
draw_box(8.5, y_attn, 4.5, 1.2,
         'MultiheadAttention\nQ=class_query\nK=V=image features', C_ATTN, fontsize=10)

# 箭頭：Permute → Attention (K, V)
draw_arrow(7, y_perm - 0.35, 8.5, y_attn + 0.6, 'K, V')

# 箭頭：Query → Attention (Q)
draw_arrow(3 + 1.75, y_query, 8.5 - 2.25, y_attn, 'Q')

# ============================================================
# 5. Attention 輸出
# ============================================================
y_attn_out = 13.5
draw_arrow(8.5, y_attn - 0.6, 7, y_attn_out + 0.35)
draw_box(7, y_attn_out, 4.5, 0.7,
         'Attended Output\n[Q, B, embed_dim]', C_INPUT, fontsize=10)

# ============================================================
# 6. LayerNorm
# ============================================================
y_norm = 12.2
draw_arrow(7, y_attn_out - 0.35, 7, y_norm + 0.35)
draw_box(7, y_norm, 3.5, 0.65, 'LayerNorm', C_NORM, fontsize=11)

# ============================================================
# 7. Dropout
# ============================================================
y_drop = 11.1
draw_arrow(7, y_norm - 0.325, 7, y_drop + 0.325)
draw_box(7, y_drop, 3.5, 0.65, 'Dropout', C_NORM, fontsize=11)

# ============================================================
# 8. Permute + Flatten
# ============================================================
y_flat2 = 9.8
draw_arrow(7, y_drop - 0.325, 7, y_flat2 + 0.35)
draw_box(7, y_flat2, 4.5, 0.7,
         'Permute + Flatten\n[B, Q * embed_dim]', C_OP, fontsize=10)

# ============================================================
# 9. Linear 分類器
# ============================================================
y_linear = 8.5
draw_arrow(7, y_flat2 - 0.35, 7, y_linear + 0.35)
draw_box(7, y_linear, 4.5, 0.7,
         'Linear\n[B, Q*embed_dim] → [B, num_classes]', C_LINEAR, fontsize=10)

# ============================================================
# 10. 輸出
# ============================================================
y_out = 7.2
draw_arrow(7, y_linear - 0.35, 7, y_out + 0.35)
draw_box(7, y_out, 4.0, 0.7,
         'cls_output\n[B, num_classes]', C_INPUT, fontsize=11)

# ============================================================
# 右側說明區域：數值範例
# ============================================================
info_x = 2.5
info_y = 5.5
ax.text(7, info_y + 0.6, '數值範例 (預設參數)', ha='center', fontsize=13,
        fontweight='bold', color='#333333')

info_lines = [
    'embed_dim = 512  (encoder 最後一層 channel 數)',
    'query_num = 4    (可學習查詢向量數量)',
    'num_heads = 4    (多頭注意力頭數)',
    'num_classes = 5  (0=無動脈瘤, 1-4=四個 location)',
    '',
    '資料流範例：',
    '  輸入:  [B, 512, 4, 4, 4]',
    '  Flatten:  [B, 512, 64]      (S=64 個空間 token)',
    '  Permute:  [64, B, 512]      (給 MHA 用)',
    '  Query:    [4, B, 512]       (4 個可學習查詢)',
    '  Attended: [4, B, 512]       (注意力輸出)',
    '  Flatten:  [B, 2048]         (4 × 512)',
    '  Linear:   [B, 5]            (最終分類)',
]

for i, line in enumerate(info_lines):
    ax.text(info_x, info_y - 0.05 - i * 0.38, line, fontsize=9.5,
            fontfamily='monospace', color='#333333', va='top')

# ============================================================
# 圖例
# ============================================================
legend_items = [
    (C_INPUT, '輸入 / 輸出'),
    (C_OP,    '張量操作'),
    (C_PARAM, '可學習參數'),
    (C_ATTN,  'Cross-Attention'),
    (C_NORM,  'Norm / Dropout'),
    (C_LINEAR,'Linear 分類器'),
]
lx, ly = 0.5, 0.8
for i, (color, label) in enumerate(legend_items):
    rect = mpatches.FancyBboxPatch(
        (lx + i * 2.2, ly - 0.2), 0.4, 0.4,
        boxstyle="round,pad=0.05", facecolor=color, edgecolor='#333333', linewidth=1
    )
    ax.add_patch(rect)
    ax.text(lx + i * 2.2 + 0.55, ly, label, fontsize=9, va='center', color='#333333')

plt.tight_layout()
save_path = r'C:\Users\user\Desktop\nnUNet\nnResUNet-github\nnResUNet-location-cls-upsample\AttentionClassifier_architecture.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'圖片已儲存至: {save_path}')
