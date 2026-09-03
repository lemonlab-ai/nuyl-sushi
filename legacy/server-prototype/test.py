from mmaction.apis import inference_recognizer, init_recognizer


config_path = './configs/bmn_2xb8-400x100-9e_activitynet-feature.py'
checkpoint_path = 'bmn_2xb8-400x100-9e_activitynet-feature_20220908-79f92857.pth'
image_path = '/videos/test.mp4'

model = init_recognizer(config_path, checkpoint_path, device="cuda:0")

result = inference_recognizer(model, image_path)
