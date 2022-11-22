# ./local

---

## 설명

- 이 폴더는 local 작업 및 파일들을 저장하기 위한 공간입니다.

- 이 폴더의 README.md, demo.jpg를 제외한 모든 파일은 git 제어에서 무시됩니다.

## Demo 사용 해보기

- 작업에 앞서, 터미널의 현재 디렉토리를 ./local로 변경해주세요.

- yolo 데모 다운로드

```bash
pipenv run mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

- 가상환경으로 python 실행

```bash
pipenv run python
```

- 데모 구동

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'yolov3_mobilenetv2_320_300e_coco.py'
checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/cat.jpg')
```
