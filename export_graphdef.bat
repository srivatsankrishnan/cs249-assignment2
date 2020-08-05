python export_inference_graph.py ^
    --alsologtostderr ^
    --dataset_name=visualwakewords ^
    --model_name=mobilenet_v1_025 ^
    --image_size=96 ^
    --use_grayscale=True ^
    --output_file=vww_96_grayscale_graph.pb

