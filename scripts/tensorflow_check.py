import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f" - {gpu}")
    # 測試一個簡單的運算
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        print("Compute test result:", tf.matmul(a, b))
else:
    print("No GPU found. Check your CUDA/Driver installation.")