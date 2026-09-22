def freeze_module_layers(sub_model, num_layers_to_freeze):
    """
    Generic freezer for WavLM, RoBERTa, or MiniLM.
    """
    # 1. Freeze the feature extractor / embeddings first
    # WavLM has feature_extractor; RoBERTa has embeddings
    for name, module in sub_model.named_children():
        if any(x in name for x in ["embeddings", "feature_extractor", "feature_projection"]):
            for param in module.parameters():
                param.requires_grad = False

    # 2. Access the layers
    # Hugging Face models usually store layers in a ModuleList called 'layer' or 'layers'
    encoder = None
    if hasattr(sub_model, "encoder"):
        encoder = sub_model.encoder
    elif hasattr(sub_model, "transformer"): # Some variants
        encoder = sub_model.transformer

    if encoder:
        # Check for 'layer' (RoBERTa/MiniLM) or 'layers' (WavLM/Hubert)
        layers = getattr(encoder, "layer", getattr(encoder, "layers", None))
        
        if layers:
            for i, layer in enumerate(layers):
                if i < num_layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
    
    trainable = sum(p.numel() for p in sub_model.parameters() if p.requires_grad)
    print(f"Module Check: {trainable} trainable parameters remaining.")

