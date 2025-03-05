from PIL import Image

# 创建一个字典，用于保存已经打开的大图片，避免重复加载
loaded_images = {}

# 标注文件路径
annotation_file = "/data/qiaoq/Project/salad_tz/RESULT_9_mine.txt"

with open(annotation_file, "r", encoding="utf-8") as f:
    for line in f:
        # 忽略空行
        if not line.strip():
            continue
        
        # 解析每一行的内容，假设每行按照空格分隔
        parts = line.strip().split()
        if len(parts) != 6:
            print(f"跳过格式不正确的一行：{line.strip()}")
            continue

        # 第一列：保存截取图片的文件名
        output_filename = parts[0]
        # 第二列：大图片的文件名
        big_image_filename = parts[1]

        # 后四列为矩形区域的左上和右下角的坐标
        try:
            left = int(parts[2])
            top = int(parts[3])
            right = int(parts[4])
            bottom = int(parts[5])
        except ValueError:
            print(f"坐标转换错误，跳过这一行：{line.strip()}")
            continue

        # 如果还没有加载大图片，则加载一次并保存
        if big_image_filename not in loaded_images:
            try:
                img = Image.open(big_image_filename)
                loaded_images[big_image_filename] = img
            except Exception as e:
                print(f"加载图片 {big_image_filename} 失败: {e}")
                continue
        else:
            img = loaded_images[big_image_filename]
        
        # 裁剪区域 (left, upper, right, lower)
        cropped_img = img.crop((left, top, right, bottom))
        
        try:
            cropped_img.save(output_filename)
            print(f"保存裁剪图像为 {output_filename}")
        except Exception as e:
            print(f"保存图片 {output_filename} 失败: {e}")