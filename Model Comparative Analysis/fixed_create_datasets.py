# ============================================
# FIXED create_datasets() function
# This version handles non-image files gracefully
# ============================================

def create_datasets():
    """
    Create training and validation datasets WITHOUT caching.
    Uses label_mode='int' for simpler filtering and better memory efficiency.
    Filters to SELECTED_CLASSES if specified.
    Handles non-image files gracefully.
    """
    # Training dataset with integer labels
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=True
    )

    # Validation dataset with integer labels
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        valid,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=False
    )

    all_classes = train_dataset.class_names
    
    # Filter to selected classes if specified
    if SELECTED_CLASSES is not None:
        print(f"Filtering to {len(SELECTED_CLASSES)} selected classes...")
        
        # Get indices of selected classes
        selected_indices = [all_classes.index(c) for c in SELECTED_CLASSES]
        
        def filter_classes(image, label):
            # Check if label is in our selected list
            return tf.reduce_any(tf.equal(label, selected_indices))
        
        def remap_labels(image, label):
            # Remap old class index to new class index
            for new_idx, old_idx in enumerate(selected_indices):
                label = tf.where(tf.equal(label, old_idx), new_idx, label)
            return image, label
        
        # Apply filtering - keep batched to avoid file reading issues
        # Use filter on batched data instead of unbatched
        def batch_filter_and_remap(images, labels):
            # Filter entire batch
            mask = tf.reduce_any(
                tf.equal(tf.expand_dims(labels, 1), selected_indices), 
                axis=1
            )
            
            # Get filtered images and labels
            filtered_images = tf.boolean_mask(images, mask)
            filtered_labels = tf.boolean_mask(labels, mask)
            
            # Remap labels
            remapped_labels = filtered_labels
            for new_idx, old_idx in enumerate(selected_indices):
                remapped_labels = tf.where(
                    tf.equal(remapped_labels, old_idx), 
                    new_idx, 
                    remapped_labels
                )
            
            return filtered_images, remapped_labels
        
        # Apply filtering on batches, then rebatch
        train_dataset = train_dataset.map(
            batch_filter_and_remap,
            num_parallel_calls=tf.data.AUTOTUNE
        ).unbatch().batch(batch_size)
        
        val_dataset = val_dataset.map(
            batch_filter_and_remap,
            num_parallel_calls=tf.data.AUTOTUNE
        ).unbatch().batch(batch_size)
        
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
