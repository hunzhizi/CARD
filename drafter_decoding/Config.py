class Config:
    REJECTION_FLAG = -1
    TARGET_MODEL_RANK = 1
    DRAFTER_RANK = 0
    _heng_yuan_yun = "/hy-tmp"
    _3090ti = "/mnt/data/zhouShaoRepo/model"
    MODEL_DIR = _heng_yuan_yun
    PREDICTION_NUM = 1

if __name__ == '__main__':
    import torch
    tree_state = torch.zeros(50, 1)
    tree_state[0][0] = 1
    tree_state[1][0] = 2
    tree_state[2][0] = 3
    print(tree_state)