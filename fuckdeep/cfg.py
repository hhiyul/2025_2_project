class Cfg:
    IMAGE_SIZE = 224  # (H,W) 전처리에서 Resize((224,224)) 가정
    PATCH_DOWN = 16   # CNN으로 1/16 해상도까지 다운샘플 (224->14)
    D_MODEL = 384     # Transformer 임베딩 차원
    N_HEAD = 6
    DEPTH  = 6
    MLP_RATIO = 4
    DROPOUT = 0.1
    ATTN_DROPOUT = 0.1

    # 학습 관련
    LR = 3e-4
    EPOCHS = 10
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    #쥬피터 노트북에선 2이상 안됨
    MIXED_PRECISION = True

    # 태스크 토글
    MULTITASK = False  # True: (과일종, 신선도) 두 개의 헤드
    NUM_CLASSES = 14   # 단일 과제일 때 클래스 수

    # 멀티태스크일 때 각 헤드 클래스 수
    NUM_FRUIT = 7
    NUM_FRESH = 2