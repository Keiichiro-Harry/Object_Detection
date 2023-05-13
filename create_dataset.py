import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

# パラメータの設定
N = 432 # 生成する画像数(画像1枚あたり)(=dict_x・dict_y・2(左右)にすると偏りがない)
bbox_color = (0, 0, 255) # 長方形のバウンディングボックスの色
M = 64 # 画像の一辺の長さ
dist_x = 18 #歪み値x
dist_y = 12 #歪み値y
blank = 5 #もとの画像より5pxずつ余白をとる
outside = 3 #バウンディングボックスは対象より3px離れて作図

# M×Mの正方形の画像を読み込み
img_H = cv2.imread('H.png')
img_S = cv2.imread('S.png')
img_U = cv2.imread('U.png')
imgs=[img_H, img_S, img_U]

# N枚の画像を生成し、フォルダに保存
os.makedirs('dataset', exist_ok=True)
for i in range(N):
    # 台形の形状を決定
    if i < N/2:
        # 右上と右下の座標を決定。
        right_top = (M-1-blank, blank)
        right_bottom = (M-1-blank, M-1-blank)
        # 左上と左下の座標を決定
        move_x = i % dist_x
        move_y = i % dist_y
        left_top = (move_x + blank, move_y + blank)
        left_bottom = (move_x + blank, M-1-blank-move_y)
    else:
        # 左上と左下の座標を決定
        left_top = (blank, blank)
        left_bottom = (blank, M-1-blank)
        # 右上と右下の座標を決定
        move_x = i % dist_x
        move_y = i % dist_y

        right_top = (M-1-blank-move_x, move_y + blank)
        right_bottom = (M-1-blank-move_x, M-1-blank-move_y)
    
    # 台形の形状に基づいて変換行列を求める
    dst_points = np.float32([left_top, left_bottom, right_bottom, right_top])
    src_points = np.float32([(0, 0), (0, M-1), (M-1, M-1), (M-1, M-1)])
    MMM = cv2.getPerspectiveTransform(src_points, dst_points)

    # 元の画像を台形に変換
    for j in range(len(imgs)):
        warped_img = cv2.warpPerspective(imgs[j], MMM, (M, M))

        # 周りを水色や灰色で塗りつぶす
        mask = np.zeros((M, M, 3), dtype=np.uint8)
        if i % 2 == 0 :
            if i < N/2 :
                cv2.fillPoly(mask, [np.int32([[0, 0], [M, 0], [M, M], [0, M]])], (255, 191, 0))
            else :
                cv2.fillPoly(mask, [np.int32([[0, 0], [M, 0], [M, M], [0, M]])], (30, 30, 30))
        else :
            if i < N/2 :
                cv2.fillPoly(mask, [np.int32([[0, 0], [M, 0], [M, M], [0, M]])], (30, 30, 30))
            else :
                cv2.fillPoly(mask, [np.int32([[0, 0], [M, 0], [M, M], [0, M]])], (255, 191, 0))

        cv2.fillPoly(mask, [np.int32(dst_points)], (0, 0, 0))
        masked_warped_img = cv2.addWeighted(warped_img, 1, mask, 0.5, 0)

        # 台形の周りにバウンディングボックスを描画
        bounding_box = Image.new('RGBA', (M, M))
        draw = ImageDraw.Draw(bounding_box)
        #draw.polygon([left_top, left_bottom, right_bottom, right_top],
        #             outline=bbox_color)
        draw.rectangle([min(left_top[0],right_top[0])-outside,min(left_top[1],right_top[1])-outside,max(left_bottom[0],right_bottom[0])+outside,max(left_bottom[1],right_bottom[1])+outside])

        del draw
        bounding_box_array = np.array(bounding_box)
        masked_warped_img[bounding_box_array[:, :, 3] != 0] = bbox_color

        # アノテーション作成
        filename = os.path.join('dataset', f'{i*len(imgs)+j}.txt')
        open(filename, 'w').close()

        with open(filename, 'a') as file:
            #file.write(f"image_{i}.png,{left_top}, {left_bottom}, {right_bottom}, {right_top}, 1"+'\n')
            file.write(f"{j} {(left_top[0]+right_top[0])/2/M} 0.5 {((right_bottom[0]-left_bottom[0])+outside*2)/M} {1-(blank-outside)*2/M}"+'\n')
        # 変形後の画像を保存
        filename = os.path.join('dataset', f'{i*len(imgs)+j}.png')
        cv2.imwrite(filename, masked_warped_img)
