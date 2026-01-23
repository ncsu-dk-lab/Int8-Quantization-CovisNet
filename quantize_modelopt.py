from modelopt.onnx.quantization import quantize

# Enc

### New
quantize(
    onnx_path="models/0kc5po4ee18_float16_onnx_cuda_enc.onnx",#"models/0kc5po4ee18_float32_onnx_cuda_enc.onnx",
    quantize_mode="int8",                          # fp8, int8, int4, etc.
    calibration_data_path="calib/calib_encoder_inputs.npz",
    calibration_method="entropy",                      # max, entropy, awq_clip, rtn_dq, etc.
    output_path="models/0kc5po4ee18_int8_etp_modelopt_from_onnx_fp16_cuda_enc.onnx",
    dq_only=False,                                  # DQ-only weights; no activation Q nodes
    high_precision_dtype="fp16",
    builderOptimizationLevel=4,
)


# trtexec --onnx=models/0kc5po4ee18_float32_onnx_cuda_enc.onnx --minShapes=image:1x3x224x224 --optShapes=image:8x3x224x224 --maxShapes=image:360x3x224x224 --verbose --dumpLayerInfo --saveEngine=enc.engine

# trtexec --onnx=models/0kc5po4ee18_int8_smoothquant_onnx_cuda_enc.onnx --int8 --fp16 --minShapes=image:1x3x224x224 --optShapes=image:8x3x224x224 --maxShapes=image:360x3x224x224 --verbose --dumpLayerInfo --saveEngine=enc_int8_smoothQ.engine;\

'''
trtexec --onnx=models/0kc5po4ee18_int8_smoothquant_onnx_cuda_enc.onnx --int8 --fp16 --minShapes=image:1x3x224x224 --optShapes=image:8x3x224x224 --maxShapes=image:360x3x224x224 --verbose --dumpLayerInfo --saveEngine=enc_int8_SmoothQ.engine;\
trtexec --onnx=models/0kc5po4ee18_int8_smoothquant_onnx_cuda_bev.onnx --int8 --fp16 --minShapes=features_i:1x128x24,features_j:1x128x24,edge_prediction:1x17 --optShapes=features_i:8x128x24,features_j:8x128x24,edge_prediction:8x17 --maxShapes=features_i:360x128x24,features_j:360x128x24,edge_prediction:360x17 --verbose --dumpLayerInfo --saveEngine=bev_int8_SmoothQ.engine;\
trtexec --onnx=models/0kc5po4ee18_int8_smoothquant_onnx_cuda_msg.onnx --int8 --fp16 --minShapes=features_i:1x128x24,features_j:1x128x24 --optShapes=features_i:8x128x24,features_j:8x128x24 --maxShapes=features_i:360x128x24,features_j:360x128x24 --verbose --dumpLayerInfo --saveEngine=msg_int8_SmoothQ.engine;\
trtexec --onnx=models/0kc5po4ee18_int8_smoothquant_onnx_cuda_bevdec.onnx --int8 --fp16 --minShapes=bev_features:1x384 --optShapes=bev_features:8x384 --maxShapes=bev_features:360x384 --verbose --dumpLayerInfo --saveEngine=bevdec_int8_SmoothQ.engine
'''

# # bev
quantize(
    onnx_path="models/0kc5po4ee18_float32_onnx_cuda_bev.onnx",
    quantize_mode="int8",       # fp8, int8, int4 etc.
    calibration_data_path="calib/calib_bev_inputs.npz",
    calibration_method="entropy",   # max, entropy, awq_clip, rtn_dq etc.
    output_path="models/0kc5po4ee18_int8_etp_modelopt_from_onnx_fp32_cuda_bev.onnx",
    dq_only=False,                                  # DQ-only weights; no activation Q nodes
    high_precision_dtype="fp16",
    builderOptimizationLevel=4,
)

# trtexec --onnx=models_exported/0kc5po4ee18_float16_onnx_cuda_bev.onnx --minShapes=features_i:1x128x24,features_j:1x128x24,edge_prediction:1x17 --optShapes=features_i:8x128x24,features_j:8x128x24,edge_prediction:8x17 --maxShapes=features_i:360x128x24,features_j:360x128x24,edge_prediction:360x17 --fp16 --verbose --dumpLayerInfo --saveEngine=bev.engine

# trtexec --onnx=models/0kc5po4ee18_float32_onnx_cuda_bev.onnx --minShapes=features_i:1x128x24,features_j:1x128x24,edge_prediction:1x17 --optShapes=features_i:8x128x24,features_j:8x128x24,edge_prediction:8x17 --maxShapes=features_i:360x128x24,features_j:360x128x24,edge_prediction:360x17 --verbose --dumpLayerInfo --saveEngine=bev.engine


# msg
quantize(
    onnx_path="models/0kc5po4ee18_float32_onnx_cuda_msg.onnx",
    quantize_mode="int8",       # fp8, int8, int4 etc.
    calibration_data_path="calib/calib_message_inputs.npz",
    calibration_method="entropy",   # max, entropy, awq_clip, rtn_dq etc.
    output_path="models/0kc5po4ee18_int8_etp_modelopt_from_onnx_fp32_cuda_msg.onnx",
    dq_only=False,                                  # DQ-only weights; no activation Q nodes
    high_precision_dtype="fp16",
    builderOptimizationLevel=4,
)

# trtexec --onnx=models_exported/0kc5po4ee18_float16_onnx_cuda_msg.onnx --minShapes=features_i:1x128x24,features_j:1x128x24 --optShapes=features_i:8x128x24,features_j:8x128x24 --maxShapes=features_i:360x128x24,features_j:360x128x24 --fp16 --verbose --dumpLayerInfo --saveEngine=msg.engine

# trtexec --onnx=models/0kc5po4ee18_float32_onnx_cuda_msg.onnx --minShapes=features_i:1x128x24,features_j:1x128x24 --optShapes=features_i:8x128x24,features_j:8x128x24 --maxShapes=features_i:360x128x24,features_j:360x128x24 --verbose --dumpLayerInfo --saveEngine=msg.engine

# bevdec
quantize(
    onnx_path="models/0kc5po4ee18_float32_onnx_cuda_bevdec.onnx",
    quantize_mode="int8",       # fp8, int8, int4 etc.
    calibration_data_path="calib/calib_bevdecoder_inputs.npz",
    calibration_method="entropy",   # max, entropy, awq_clip, rtn_dq etc.
    output_path="models/0kc5po4ee18_int8_etp_modelopt_from_onnx_fp32_cuda_bevdec.onnx",
    dq_only=False,                                  # DQ-only weights; no activation Q nodes
    high_precision_dtype="fp16",
    builderOptimizationLevel=4,
)

# trtexec --onnx=models_exported/0kc5po4ee18_float16_onnx_cuda_bevdec.onnx --minShapes=bev_features:1x384 --optShapes=bev_features:8x384 --maxShapes=bev_features:360x384 --fp16 --verbose --dumpLayerInfo --saveEngine=bevdec.engine

# trtexec --onnx=models/0kc5po4ee18_float32_onnx_cuda_bevdec.onnx --minShapes=bev_features:1x384 --optShapes=bev_features:8x384 --maxShapes=bev_features:360x384 --verbose --dumpLayerInfo --saveEngine=bevdec.engine