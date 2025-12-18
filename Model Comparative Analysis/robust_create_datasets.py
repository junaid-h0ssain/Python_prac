# ============================================
# SIMPLEST SOLUTION: Pre-filter using subdirectories
# This avoids all the unbatch/rebatch issues
# ============================================

import os
import shutil

# Step 1: Define which classes you want
SELECTED_CLASSES = None  # Set to None for all classes

# Example: Uncomment to select specific classes
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
# Step 2: Simple create_datasets - NO FILTERING
# Just use the dataset as-is
# ============================================

def create_datasets():
    """
    Create training and validation datasets.
    Uses label_mode='int' for better memory efficiency.
    """
    # Determine which directories to use
    if SELECTED_CLASSES is not None:
        # We'll use filtered directories (you need to set these up)
        train_dir = train  # Use your filtered train directory
        valid_dir = valid  # Use your filtered valid directory
        class_names = SELECTED_CLASSES
        num_classes = len(SELECTED_CLASSES)
        print(f"Using filtered dataset with {num_classes} classes")
    else:
        train_dir = train
        valid_dir = valid
        class_names = None  # Will be auto-detected
        num_classes = None
    
    # Training dataset with integer labels
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=True
    )

    # Validation dataset with integer labels
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=False
    )

    # Get class info if not specified
    if class_names is None:
        class_names = train_dataset.class_names
        num_classes = len(class_names)

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


# ============================================
# ALTERNATIVE: Use tf.data.Dataset filtering
# This is more robust than unbatch/filter/batch
# ============================================

def create_datasets_with_filtering():
    """
    Create datasets with class filtering using a more robust approach.
    """
    # Create full datasets first
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=True
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        valid,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=False
    )

    all_classes = train_dataset.class_names
    
    if SELECTED_CLASSES is not None:
        print(f"Filtering to {len(SELECTED_CLASSES)} selected classes...")
        
        # Get indices of selected classes
        selected_indices = tf.constant([all_classes.index(c) for c in SELECTED_CLASSES], dtype=tf.int32)
        num_selected = len(SELECTED_CLASSES)
        
        def filter_and_remap_batch(images, labels):
            """Filter and remap labels in a batch."""
            # Create mask for samples we want to keep
            mask = tf.reduce_any(
                tf.equal(tf.expand_dims(labels, 1), 
                        tf.expand_dims(selected_indices, 0)),
                axis=1
            )
            
            # Filter images and labels
            filtered_images = tf.boolean_mask(images, mask)
            filtered_labels = tf.boolean_mask(labels, mask)
            
            # Remap labels to new indices (0 to num_selected-1)
            # Create a lookup table
            remapped_labels = tf.zeros_like(filtered_labels)
            for new_idx in range(num_selected):
                old_idx = selected_indices[new_idx]
                remapped_labels = tf.where(
                    tf.equal(filtered_labels, old_idx),
                    new_idx,
                    remapped_labels
                )
            
            return filtered_images, remapped_labels
        
        # Apply filtering and remapping
        train_dataset = train_dataset.map(
            filter_and_remap_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        ).unbatch().batch(batch_size)
        
        val_dataset = val_dataset.map(
            filter_and_remap_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
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


# ============================================
# Choose which function to use
# ============================================

print("Creating datasets...")

# Use this if SELECTED_CLASSES is None (all classes)
if SELECTED_CLASSES is None:
    train_dataset, val_dataset, class_names, num_classes = create_datasets()
else:
    # Use the filtering version
    train_dataset, val_dataset, class_names, num_classes = create_datasets_with_filtering()

print("\n✓ Datasets created successfully!")
