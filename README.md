# 재활용 품목 분류를 위한 Object Detection

---

## install

- 환경 구성에 필요한 package 설치
- mmcv 설치(pipenv로 직접설치 안됨)

```bash
pipenv install
pipenv run mim install mmcv-full
```

- mmcv와 mmdetection이 제대로 설치되었는지 확인
- [.local/README.md](.local/README.md) 참조



[train]
python tools/train.py configs/~~

[inference]
python tools/test.py configs/~~ work_dirs/~ /latest.pth 