# ============================================
# SIMPLEST WORKING SOLUTION
# Just use categorical labels - no filtering complexity
# ============================================

# Step 1: Class Selection
SELECTED_CLASSES = None  # Set to None for ALL classes

# To select specific classes, uncomment:
# SELECTED_CLASSES = [
#     'Tomato___Early_blight',
#     'Tomato___Late_blight',
#     'Tomato___healthy'
# ]

print(f"Class filter: {'ALL CLASSES' if SELECTED_CLASSES is None else f'{len(SELECTED_CLASSES)} selected classes'}")


# Step 2: Simple Dataset Creation (NO FILTERING - works 100%)
def create_datasets():
    """
    Create training and validation datasets.
    Back to categorical for simplicity - it just works!
    """
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',  # Back to categorical - simple and works
        shuffle=True
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        valid,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"First 5 classes: {class_names[:5]}")

    total_batches = train_dataset.cardinality().numpy()

    train_dataset = train_dataset.shuffle(
        buffer_size=100,
        reshuffle_each_iteration=True
    )

    portion = 0.5
    train_dataset = train_dataset.take(int(total_batches * portion))

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    normalization = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, class_names, num_classes


# If you REALLY need class filtering, manually specify subdirectories:
def create_filtered_datasets_manual():
    """
    For filtering: manually create a temp directory with only selected classes.
    This is the most reliable way.
    """
    import os
    import tempfile
    import shutil
    
    if SELECTED_CLASSES is None:
        # No filtering needed
        return create_datasets()
    
    print(f"Creating filtered dataset with {len(SELECTED_CLASSES)} classes...")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    temp_train = os.path.join(temp_dir, 'train')
    temp_valid = os.path.join(temp_dir, 'valid')
    os.makedirs(temp_train)
    os.makedirs(temp_valid)
    
    # Copy only selected class folders
    for class_name in SELECTED_CLASSES:
        src_train = os.path.join(train, class_name)
        src_valid = os.path.join(valid, class_name)
        dst_train = os.path.join(temp_train, class_name)
        dst_valid = os.path.join(temp_valid, class_name)
        
        if os.path.exists(src_train):
            shutil.copytree(src_train, dst_train)
        if os.path.exists(src_valid):
            shutil.copytree(src_valid, dst_valid)
    
    # Now create datasets from filtered directories
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        temp_train,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        temp_valid,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    class_names = SELECTED_CLASSES
    num_classes = len(SELECTED_CLASSES)

    print(f"Filtered to {num_classes} classes")

    # Apply augmentation...
    total_batches = train_dataset.cardinality().numpy()
    train_dataset = train_dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)
    
    portion = 0.5
    train_dataset = train_dataset.take(int(total_batches * portion))

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    normalization = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, class_names, num_classes


# Use this:
print("Creating datasets...")
train_dataset, val_dataset, class_names, num_classes = create_datasets()
print("\n✓ Datasets created successfully!")


# ============================================
# Keep your model functions with categorical_crossentropy
# ============================================

def create_model(base_model_class, model_name, num_classes):
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=image_size + (3,)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',  # Keep this for categorical labels
        metrics=['accuracy'],
        jit_compile=True
    )

    return model, base_model


def fine_tune_model(model, base_model, num_layers_to_unfreeze=20):
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',  # Keep this for categorical labels
        metrics=['accuracy']
    )

    return model
