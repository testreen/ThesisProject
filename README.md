# ThesisProject
MSc thesis project at KTH

### To install
Using Conda environment, use the requirements.txt file. For EfficientNet, additional steps required described below. 

#### EfficientNet
Original code from: https://github.com/lukemelas/EfficientNet-PyTorch.

To install:
```
cd EfficientNet-Pytorch
pip install -e .
```

To train example:
```
python efficientNet.py --epochs 100 --pretrained --batch-size 64 --wd 1e-4 --lr 1.25e-2 --image_size 32 --momentum 0.9 --advprop -val
```

Arguments and dataset root folder provided in [efficientNet.py](./efficientNet.py) and file paths defined in [parseData.py](./parseData.py).

#### EfficientDet
Original code from: https://github.com/toandaominh1997/EfficientDet.Pytorch.

No extra installation required.

Available arguments and file paths provided in each file.

To test that everything is working (EfficientNet pre-trained backbone and EfficientDet):
```
cd EfficientDet.Pytorch
python test.py
```

To train example:
```
cd EfficientDet.Pytorch
python train.py
```

To evaluate a trained model:
```
cd EfficientDet.Pytorch
python eval.py
```

To evaluate and visualize one 2000x2000 WSI:
```
cd EfficientDet.Pytorch
python evalSlide.py
```

