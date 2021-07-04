from tl.resnet import ResNet

resnet = ResNet("devel_data/train")
resnet.train(10)
result = resnet.predict("devel_data/test", "devel_data/out")
result.to_csv("devel_data/out.csv")