from importlib import import_module

def dynamic_import(module_name, class_name):
    module = import_module(module_name)
    return getattr(module, class_name)

def get_dataset_class(config):
    dataset_module = config['DATASET']["MODULE"]
    dataset_class = config['DATASET']["CLASS"]
    return dynamic_import(dataset_module, dataset_class)

def get_tokenizer_class(config):
    tokenizer_module = config['TOKENIZER']["MODULE"]
    tokenizer_class = config['TOKENIZER']["CLASS"]
    return dynamic_import(tokenizer_module, tokenizer_class)

def get_encoder_class(config):
    encoder_module = config['ENCODER']['MODULE']
    encoder_class = config['ENCODER']["CLASS"]
    return dynamic_import(encoder_module, encoder_class)

def get_decoder_class(config):
    decoder_module = config['DECODER']['MODULE']
    decoder_class = config['DECODER']["CLASS"]
    return dynamic_import(decoder_module, decoder_class)

def model_class(config):
    model_module = config['MODEL']["MODULE"]
    model_class = config['MODEL']["CLASS"]
    return dynamic_import(model_module, model_class)

def run_imports(config):
    DatasetClass = get_dataset_class(config)
    TokenizerClass = get_tokenizer_class(config)
    EncoderClass = get_encoder_class(config)
    DecoderClass = get_decoder_class(config)
    ModelClass = model_class(config)
    
    return {
        "DatasetClass": DatasetClass,
        "TokenizerClass": TokenizerClass,
        "EncoderClass": EncoderClass,
        "DecoderClass": DecoderClass,
        "ModelClass": ModelClass
    }