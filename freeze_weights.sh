python freeze_graph.py \
    --input_graph=vww_96_grayscale_graph.pb \
    --input_checkpoint="C:\workspace\cs249-assignment2-module-2\vww_96_grayscale\model.ckpt-463859" \
    --input_binary=true \
    --output_graph=person_detection_frozen.pb \
    --output_node_names=MobilenetV1/Predictions/Reshape_1

