 python eval_image_classifier.py ^
    --alsologtostderr ^
    --checkpoint_path=<CHECK_POINT_PATH> ^
    --dataset_dir=<DATASET_DIRECTORY>^
    --dataset_name=visualwakewords ^
    --dataset_split_name=val ^
    --model_name=mobilenet_v1_025 ^
    --preprocessing_name=mobilenet_v1 ^
    --use_grayscale=True ^
    --train_image_size=96

