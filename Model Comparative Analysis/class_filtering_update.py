# ============================================
# CELL 1: Class Selection (Add after Configuration cell)
# ============================================

# SELECT SPECIFIC CLASSES (OPTIONAL)
# List the class names you want to train on.
# Set to None to use ALL 38 classes, or specify a list like below:

SELECTED_CLASSES = None  # Use all classes

# Example: To train on only specific classes, uncomment and modify:
# SELECTED_CLASSES = [
#     'Tomato___Early_blight',
#     'Tomato___Late_blight',
#     'Tomato___healthy',
#     'Potato___Early_blight',
#     'Potato___Late_blight',
#     'Potato___healthy'
# ]

print(f"Class filter: {'ALL CLASSES' if SELECTED_CLASSES is None else f'{len(SELECTED_CLASSES)} selected classes'}")


# ============================================
# CELL 2: Updated create_datasets() function
# Replace your existing create_datasets() cell with this
# ============================================

def create_datasets():
    """
    Create training and validation datasets WITHOUT caching.
    Uses label_mode='int' for simpler filtering and better memory efficiency.
    Filters to SELECTED_CLASSES if specified.
    """
    # Training dataset with integer labels
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',  # Changed from 'categorical' for simpler filtering
        shuffle=True
    )

    # Validation dataset with integer labels
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        valid,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',  # Changed from 'categorical' for simpler filtering
        shuffle=False
    )

    all_classes = train_dataset.class_names
    
    # Filter to selected classes if specified
    if SELECTED_CLASSES is not None:
        print(f"Filtering to {len(SELECTED_CLASSES)} selected classes...")
        
        # Get indices of selected classes
        selected_indices = [all_classes.index(c) for c in SELECTED_CLASSES]
        
        def filter_classes(image, label):
            # Much simpler with int labels - just check if label is in our list
            return tf.reduce_any(tf.equal(label, selected_indices))
        
        def remap_labels(image, label):
            # Remap old class index to new class index
            for new_idx, old_idx in enumerate(selected_indices):
                label = tf.where(tf.equal(label, old_idx), new_idx, label)
            return image, label
        
        # Apply filtering
        train_dataset = train_dataset.unbatch().filter(filter_classes).map(remap_labels).batch(batch_size)
        val_dataset = val_dataset.unbatch().filter(filter_classes).map(remap_labels).batch(batch_size)
        
        class_names = SELECTED_CLASSES
        num_classes = len(SELECTED_CLASSES)
    else:
        class_names = all_classes
        num_classes = len(all_classes)

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")

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

print("Creating datasets...")
train_dataset, val_dataset, class_names, num_classes = create_datasets()
print("\n✓ Datasets created successfully!")


# ============================================
# CELL 3: Updated create_model() function
# Replace your existing create_model() cell with this
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
        loss='sparse_categorical_crossentropy',  # Changed for int labels
        metrics=['accuracy'],
        jit_compile=True
    )

    return model, base_model

print("Model Created")


# ============================================
# CELL 4: Updated fine_tune_model() function
# Replace your existing fine_tune_model() cell with this
# ============================================

def fine_tune_model(model, base_model, num_layers_to_unfreeze=20):
   
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',  # Changed for int labels
        metrics=['accuracy']
    )

    return model

print('Fine Tune model Created')


# ============================================
# OPTIONAL: View all available classes
# Add this as a new cell to see what classes are available
# ============================================

# See what classes are available in the dataset
temp_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train,
    image_size=image_size,
    batch_size=32
)
print("Available classes in the dataset:")
print("="*60)
for i, name in enumerate(temp_ds.class_names):
    print(f"  {i+1:2d}. {name}")
print("="*60)
print(f"Total: {len(temp_ds.class_names)} classes")
