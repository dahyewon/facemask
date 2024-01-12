import torch
from torchvision import transforms
from example.train_model.models.resnet34 import MasksModelFromResNet

class ModelHandler:
    def load_model(self):
        model = MasksModelFromResNet(len(self.class_names), pretrained=True, train_all_layers=True)
        # model = torch.load(f'{self.model_path}')
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'))['model_state_dict']) # gpu에서 학습시키고 cpu로 옮길 때에는 map location 수정해줘야 오류발생하지 않음
        return model

    def get_transform(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

class DataHandler:
    def check_type(self, check_class, data):
        data = check_class(**data)
        
        return data
