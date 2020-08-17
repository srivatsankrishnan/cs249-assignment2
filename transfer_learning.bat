python train_image_classifier.py ^
    --train_dir=transfer_learning ^
    --dataset_dir=C:\workspace\cs249-assignment2-module-2\data\mask_detection^
    --dataset_name=visualwakewords ^
    --dataset_split_name=train ^
    --preprocessing_name=mobilenet_v1 ^
    --model_name=mobilenet_v1_025 ^
    --train_image_size=96 ^
    --save_summaries_secs=300 ^
    --max_number_of_steps=1005000 ^
    --learning_rate=0.045 ^
    --label_smoothing=0.1 ^
    --learning_rate_decay_factor=0.98 ^
    --num_epochs_per_decay=2.5 ^
    --moving_average_decay=0.9999 ^
    --use_grayscale=True ^
    --checkpoint_path=C:\workspace\cs249-assignment2-module-2\transfer_learning\model.ckpt-1000000^
    --checkpoint_exclude_scopes= MobilenetV1/Logits,MobilenetV1/Predictions ^
    --trainable_scopes= MobilenetV1/Logits,MobilenetV1/Predictions

