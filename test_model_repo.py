from mayo_spindles.model_repo.collection import ModelRepository

repo = ModelRepository()
for model_name in repo.get_model_names():
    print(model_name, end=' ')
    try:
        model = repo.load(model_name)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_params)
    except Exception as e:
        print(e)
