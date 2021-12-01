import torch
from torch.utils.data import Dataset


class GestureDataset(Dataset):

    def __init__(self, x_skeleton, y_gesture):
        super(GestureDataset, self).__init__()
        self.gesture = {"rock": 0, "paper": 1, "scissors": 2}
        self.x_skeleton = x_skeleton
        self.y_gesture = y_gesture

    def __len__(self):
        return len(self.y_gesture)

    def __getitem__(self, index):
        x_skeleton = self.x_skeleton[index]
        y_gesture = self.y_gesture[index]

        # print(torch.tensor(self.gesture[y_gesture], dtype=torch.int32))
        return torch.tensor(x_skeleton, dtype=torch.float32), \
               torch.tensor(self.gesture[y_gesture], dtype=torch.int32)