import cv2
import numpy as np
import glob
import os

# ==================== 参数配置 ====================
# 棋盘格内角点数 (列数, 行数) —— 10×7 方格的棋盘格对应内角点为 9×6
CHESSBOARD_SIZE = (9, 6)

# 棋盘格方格的边长（单位：毫米）—— 必须用尺子测量屏幕上方格的实际边长
SQUARE_SIZE = 25.0   # 例如 25.0 mm

# 图片文件夹
IMAGE_FOLDER = 'calibration_images'
OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================================================

# 准备三维世界坐标点 (z=0平面)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 存储所有图像的目标点和图像点
obj_points = []   # 3D 点
img_points = []   # 2D 点

# 读取图片
images = glob.glob(os.path.join(IMAGE_FOLDER, '*.jpg')) + \
         glob.glob(os.path.join(IMAGE_FOLDER, '*.png')) + \
         glob.glob(os.path.join(IMAGE_FOLDER, '*.jpeg'))

if len(images) == 0:
    print(f"错误：在 '{IMAGE_FOLDER}' 文件夹中未找到任何图片。")
    exit()

print(f"找到 {len(images)} 张图片，开始检测角点...")

success_count = 0
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"  ⚠️  无法读取 {os.path.basename(fname)}，跳过")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 先尝试使用 cv2.findChessboardCornersSB (更鲁棒)
    ret, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, None)
    if not ret:
        # 回退到普通方法
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        # 亚像素精度优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners)
        success_count += 1

        # 绘制角点并保存
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        out_path = os.path.join(OUTPUT_FOLDER, f'corners_{success_count:02d}.jpg')
        cv2.imwrite(out_path, img)
        print(f"  ✅ 成功：{os.path.basename(fname)}")
    else:
        print(f"  ❌ 失败：{os.path.basename(fname)} - 未检测到角点")

print(f"\n成功检测角点的图片数: {success_count}/{len(images)}")

if success_count < 10:
    print("⚠️  警告：有效图片少于 10 张，标定结果可能不可靠，建议至少 15 张。")
if success_count == 0:
    print("没有成功检测到任何角点，程序终止。请检查：")
    print("1. 棋盘格是否为 10×7 黑白方格 (内角点 9×6)？")
    print("2. 图片是否清晰、完整、无反光？")
    exit()

# 相机标定
print("\n开始相机标定...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("\n========== 标定结果 ==========")
print(f"重投影误差 (RMS): {ret:.4f}")
print("\n内参矩阵 K:")
print(mtx)
print("\n畸变参数 D = [k1, k2, p1, p2, k3]:")
print(dist)

# 计算平均重投影误差（可选）
mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
mean_error /= len(obj_points)
print(f"\n平均重投影误差: {mean_error:.4f} 像素")

# 去畸变对比 (对第一张成功图片处理)
first_img_path = images[0]
sample_img = cv2.imread(first_img_path)
if sample_img is not None:
    h, w = sample_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(sample_img, mtx, dist, None, newcameramtx)

    # 解决尺寸不一致：将去畸变图缩放到与原图相同尺寸（不裁剪）
    dst_resized = cv2.resize(dst, (w, h))
    comparison = np.hstack((sample_img, dst_resized))

    out_compare = os.path.join(OUTPUT_FOLDER, 'undistort_comparison.jpg')
    cv2.imwrite(out_compare, comparison)
    print(f"\n去畸变对比图已保存至：{out_compare}")
else:
    print("无法读取第一张图片进行去畸变对比。")

print("\n所有步骤完成！结果图片保存在 'output' 文件夹。")