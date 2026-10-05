# Dictionnaire de mapping pour faire correspondre le nom du feature au préfixe des poids
## Attention aux -ft s'il en manque pour les MedicalNet
## Attention aux préfixes 2drnn_ ou 2d_

prefix_map = {
    # "ResNet2D": 'resnet2d',
    # "MedicalNet34": 'medicalnet34-ft',
    # "Siamese_Mashup_Align": "mashup_attn_4br",
    # "Siamese_Synth": "siamois_final_pseudo",
    # "Siamese_Mashup": "mashup_attn",
    # "Siamese_Mashup_DTW_Attn": 'mashup_attention_dwt_attn'
    # "SeResNext26DLSTMSagcorax": 'seresnext26d_lstm_sagcorax',
    "TwoPhase_Mean": "twophase"
}