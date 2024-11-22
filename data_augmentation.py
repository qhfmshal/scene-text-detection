# 데이터를 회전, 반전시키고 그에 따른 gt파일 처리. gt 파일로부터 YOLO 데이터셋 생성
import os
import json
import os.path as osp
from glob import glob
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

IMAGE_EXTENSIONS = {'.gif', '.jpg', '.png'}

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)

def parse_label_file(label_path):
    def rearrange_points(points):
        start_idx = np.argmin([np.linalg.norm(p, ord=1) for p in points])
        if start_idx != 0:
            points = np.roll(points, -start_idx, axis=0).tolist()
        return points

    with open(label_path, encoding='utf-8') as f:
        lines = f.readlines()

    words_info, languages = dict(), set()
    for word_idx, line in enumerate(lines):
        items = line.strip().split(',', 9)
        language, transcription = items[8], items[9]
        points = np.array(items[:8], dtype=np.float32).reshape(4, 2).tolist()
        points = rearrange_points(points)

        illegibility = transcription == '###'
        orientation = 'Horizontal'
        language = get_language_token(language)
        words_info[word_idx] = dict(
            points=points, transcription=transcription, language=[language],
            illegibility=illegibility, orientation=orientation, word_tags=None
        )
        languages.add(language)

    return words_info, dict(languages=languages)


# polygon 90도 회전. dirtection = right은 시계 방향으로 회전. left는 반시계방향 회전 
def rotate_polygon_90(polygon, image_height, image_width, dirtection):
    rotated_polygon = []
    if dirtection == 'right':
      for i in range(0, len(polygon), 2):
          x, y = polygon[i], polygon[i + 1]
          new_x = image_height - y
          new_y = x
          rotated_polygon.extend([new_x, new_y])
    else :
      for i in range(0, len(polygon), 2):
          x, y = polygon[i], polygon[i + 1]
          new_x = y
          new_y = image_width - x
          rotated_polygon.extend([new_x, new_y])

    return rotated_polygon
  
# 이미지와 polygon을 회전시키고, 저장
def img_rotate_save(image_path,label_path,output_image_dir,output_label_dir,direction) :
  image = cv2.imread(image_path)
  if image is None:
      print(f"이미지를 불러올 수 없습니다: {image_path}")
  else :
    image_height, image_width = image.shape[:2]

    if direction == 'right' :
      rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else :
      rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rotated_lines = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue

        polygon = list(map(int, parts[:8]))
        text = ",".join(parts[8:])
        rotated_polygon = rotate_polygon_90(polygon, image_height, image_width,direction)

        rotated_line = ",".join(map(str, rotated_polygon)) + "," + text
        rotated_lines.append(rotated_line)

    output_image_path = os.path.join(output_image_dir, f"rotated_{direction}_{image_path.split('/')[-1]}")
    cv2.imwrite(output_image_path, rotated_image)
    output_gt_path = os.path.join(output_label_dir, f"gt_rotated_{direction}_{label_path.split('/')[-1][3:]}")
    with open(output_gt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rotated_lines))

# polygon 좌우 반전
def flip_polygon_horizontally(polygon, image_width):

    flipped_polygon = []
    for i in range(0, len(polygon), 2):
        x, y = polygon[i], polygon[i+1]
        flipped_x = image_width - x
        flipped_polygon.extend([flipped_x, y])
    return flipped_polygon


# 이미지와 polygon을 좌우 반전시키고, 저장
def img_flip_save(image_path,label_path,output_image_dir,output_label_dir) :
  image = cv2.imread(image_path)
  if image is None:
    print(f"이미지를 불러올 수 없습니다: {image_path}")
  else :
    fliped_image = cv2.flip(image, 1)
    image_height, image_width = image.shape[:2]

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fliped_lines = []

    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue

        polygon = list(map(int, parts[:8]))
        text = ",".join(parts[8:])
        fliped_polygon = flip_polygon_horizontally(polygon, image_width)

        fliped_line = ",".join(map(str, fliped_polygon)) + "," + text
        fliped_lines.append(fliped_line)

    output_image_path = os.path.join(output_image_dir, f"fliped_{image_path.split('/')[-1]}")
    cv2.imwrite(output_image_path, fliped_image)
    output_gt_path = os.path.join(output_label_dir, f"gt_fliped_{label_path.split('/')[-1][3:]}")
    with open(output_gt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fliped_lines))

# ICDAR2017-MLT의 GT를 yolo 형식(xywh)로 변환
def icdar_to_yolo(icdar_image_dir,icdar_annotation_dir,yolo_output_dir):
  
  os.makedirs(yolo_output_dir, exist_ok=True)
  # 단일 클래스(text)이므로 클래스 개수는 1
  class_id = 0

  for image_file in tqdm(os.listdir(icdar_image_dir)):
      if not image_file.endswith(('.jpg', '.png','gif')):
          continue

      # 이미지 크기
      image_path = os.path.join(icdar_image_dir, image_file)
      with Image.open(image_path) as img:
          img_width, img_height = img.size

      # ICDAR GT 파일 읽기
      annotation_file = os.path.join(icdar_annotation_dir, 'gt_'+os.path.splitext(image_file)[0] + '.txt')
      if not os.path.exists(annotation_file):
          continue

      yolo_annotation = []
      with open(annotation_file, "r") as f:
          for line in f:
              parts = line.strip().split(",")
              coords = list(map(int, parts[:8]))

              # Bounding Box 좌표
              x_coords = coords[::2]
              y_coords = coords[1::2]
              x_min, y_min = min(x_coords), min(y_coords)
              x_max, y_max = max(x_coords), max(y_coords)

              # YOLO 형식으로 변환
              x_center = ((x_min + x_max) / 2) / img_width
              y_center = ((y_min + y_max) / 2) / img_height
              width = (x_max - x_min) / img_width
              height = (y_max - y_min) / img_height

              # YOLO 형식으로 저장
              yolo_annotation.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

      # YOLO Annotation 파일 저장
      yolo_annotation_file = os.path.join(yolo_output_dir, os.path.splitext(image_file)[0] + ".txt")
      with open(yolo_annotation_file, "w") as f:
          f.write("\n".join(yolo_annotation))

def main():
  
  # 이미지 회전, 좌우반전 및 저장
  image_dir = '/content/YOLO_real/Train/images'
  label_dir = '/content/YOLO_real/Train/gt'
  output_image_dir = '/content/drive/MyDrive/aug_img_test'
  output_label_dir = '/content/drive/MyDrive/aug_gt_test'
  
  image_paths = {x for x in glob(osp.join(image_dir, '*')) if osp.splitext(x)[1] in IMAGE_EXTENSIONS}
  label_paths = set(glob(osp.join(label_dir, '*.txt')))
  assert len(image_paths) == len(label_paths)
  
  sample_ids, samples_info = list(), dict()
  
  for image_path in tqdm(image_paths):
      sample_id = osp.splitext(osp.basename(image_path))[0]
  
      label_path = osp.join(label_dir, 'gt_{}.txt'.format(sample_id))
      assert label_path in label_paths
  
      words_info, extra_info = parse_label_file(label_path)
      if 'ko' not in extra_info['languages'] or extra_info['languages'].difference({'ko', 'en'}):
          continue
      else :
        img_rotate_save(image_path,label_path,output_image_dir,output_label_dir,'right')
        img_rotate_save(image_path,label_path,output_image_dir,output_label_dir,'left')
        img_flip_save(image_path,label_path,output_image_dir,output_label_dir)

  # ICDAR2017-MLT의 GT를 yolo 형식(xywh)로 변환
  icdar_image_dir = "/content/drive/MyDrive/aug_img"
  icdar_annotation_dir = "/content/drive/MyDrive/aug_gt"
  yolo_output_dir = "/content/drive/MyDrive/aug_labels"
  icdar_to_yolo(icdar_image_dir,icdar_annotation_dir,yolo_output_dir)

if __name__ == '__main__':
    main()
