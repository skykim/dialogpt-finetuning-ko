## DialoGPT Fine-tuning Korean

- Fine-tuning DialoGPT model with the Korean emotional dialogue corpus
- DialoGPT 모델에 감성대화 말뭉치 데이터셋을 파인튜닝한 한국어 감성대화 예제

## Demo

```
Q:요새 집에만 있고 친구를 못 만나니까 외로워.
A:친구를 만나지 못해 외로움을 느끼시는군요.
```

```
Q:요즘 근심이 많아.
A:어떤 걱정인지 자세히 말씀해주실 수 있으신가요?
```

```
Q:가족 중 아무도 나에게 기대를 하지 않아.
A:가족들이 기대를 하지 않아서 속상하시겠어요.
```

```
Q:가족 병원비로 나가는 돈이 많아서 감당하기가 너무 힘들어. 속상해.
A:병원비 때문에 감당하기 힘드시군요.
```

## Requirements
- python 3.8
- pytorch 1.8.0, torchvision 0.9.0, torchaudio 0.8.0, tensorboardX
- cuda 11.1
- huggingface transformers
- pandas, scikit-learn 
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 transformers pandas scikit-learn tensorboardX tensorboard -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset
- Download [감성 대화 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86) from AI Hub
- Extract a json file in the following format.

```
[
	[
		"아들이 학교에서 무슨 일이 있었던 거 같은데 물어보지 못하겠어.",
		"아드님이 걱정되시나 봐요. 어떤 이유에서 물어보지 못하시나요?"
	],
	[
		"딸이 최근에 남자 친구랑 헤어지고 우울해 보여서 말 걸기가 좀 조심스러워.",
		"따님께서 남자 친구와 헤어진 후에 조심스러우시군요."
	],
  ...
]
```


## Model

- [Huggingface DialoGPT-small model](https://huggingface.co/microsoft/DialoGPT-small)
- Pretrained Model: 30 epochs with 23306 datasets
  - Extract the [zip file](https://drive.google.com/file/d/1xy7eYIoIDYoXDeZ_rBRIyPjmohbxaE46/view?usp=share_link) of the pre-trained model (30 epochs) to 'output-small-save' folder

## Training and Inference

- python train.py
- python inference.py

## References

- [Covid Doctor chatbot using DialoGPT](https://github.com/rushic24/DialoGPT-Finetune)
