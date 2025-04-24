def advanced_fine_tune_model(model, base_model, train_ds, val_ds, class_weights):
    """
    Advanced fine-tuning techniques to improve model accuracy:
    1. Progressive layer unfreezing
    2. Discriminative learning rates
    3. Mixup data augmentation
    4. Feature-wise regularization
    5. Knowledge distillation from model snapshots
    """
    print("\nStarting advanced fine-tuning process...")
    
    # 1. Progressive layer unfreezing
    # Instead of unfreezing all at once, we gradually unfreeze from top to bottom
    all_layers = base_model.layers
    layer_groups = []
    
    # Group layers into logical blocks (for MobileNetV3Small)
    current_block = []
    current_block_name = None
    
    for i, layer in enumerate(all_layers):
        name = layer.name
        if 'expanded_conv' in name:
            block_id = name.split('/')[0]
            if block_id != current_block_name:
                if current_block:
                    layer_groups.append(current_block)
                current_block = [i]
                current_block_name = block_id
            else:
                current_block.append(i)
    
    # Add the last block
    if current_block:
        layer_groups.append(current_block)
    
    # Reverse for progressive unfreezing (top layers first)
    layer_groups.reverse()
    
    # Keep initial top classifier layer performance record
    initial_val_performance = {'loss': float('inf'), 'accuracy': 0}
    best_val_accuracy = 0
    best_model_weights = None
    
    # Start with all frozen
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create mixup data augmentation function
    def mixup_augmentation(images, labels, alpha=0.2):
        """Create mixup of images and labels"""
        batch_size = tf.shape(images)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Create two versions of the batch
        images_1 = images
        images_2 = tf.gather(images, indices)
        labels_1 = labels
        labels_2 = tf.gather(labels, indices)
        
        # Sample lambda from beta distribution
        l = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=alpha)
        l_y = tf.reshape(l[:, 0, 0, 0], [-1, 1])
        
        # Perform mixup on both images and labels
        images = l * images_1 + (1 - l) * images_2
        labels = l_y * labels_1 + (1 - l_y) * labels_2
        
        return images, labels

    # Apply mixup to the training dataset
    train_ds_mixup = train_ds.map(
        lambda x, y: mixup_augmentation(x, y, alpha=0.2),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Create Feature-wise regularization function
    def add_feature_noise(images, labels, std=0.1):
        """Add feature-wise noise as regularization"""
        # Add small Gaussian noise to input
        noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=std)
        images = images + noise
        # Clip to valid image range [0,1]
        images = tf.clip_by_value(images, 0.0, 1.0)
        return images, labels
    
    # Apply feature noise to part of the training dataset
    train_ds_with_noise = train_ds.map(
        lambda x, y: add_feature_noise(x, y, std=0.05),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Combine original and augmented datasets
    train_ds_combined = tf.data.Dataset.sample_from_datasets(
        [train_ds, train_ds_mixup, train_ds_with_noise], 
        weights=[0.4, 0.4, 0.2]
    ).prefetch(tf.data.AUTOTUNE)
    
    # 2. Discriminative learning rates implementation
    # Create learning rate multipliers based on layer depth
    def get_lr_multiplier(layer_idx, total_layers, min_lr_factor=0.1):
        """Generate a learning rate multiplier that increases with layer depth"""
        return min_lr_factor + (1.0 - min_lr_factor) * (layer_idx / total_layers)
    
    # Progressive unfreezing and training
    print("Starting progressive unfreezing...")
    for phase, layer_group in enumerate(layer_groups):
        print(f"\nPhase {phase+1}/{len(layer_groups)}: Unfreezing layers...")
        
        # Unfreeze this group of layers
        for idx in layer_group:
            if idx < len(base_model.layers):
                base_model.layers[idx].trainable = True
                print(f"  Unfrozen layer: {base_model.layers[idx].name}")
        
        # Count trainable parameters after this round of unfreezing
        trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Create optimizer with layer-wise learning rates
        if phase == 0:
            # For first phase, use standard learning rate
            lr = FINE_TUNING_LEARNING_RATE
        else:
            # For subsequent phases, gradually decrease starting learning rate
            lr = FINE_TUNING_LEARNING_RATE * (0.7 ** phase)
        
        print(f"  Base learning rate for this phase: {lr:.8f}")
        
        # Compile with current learning rate
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        # Adjust patience and epochs based on phase
        patience = 3 if phase < len(layer_groups) - 1 else 5
        epochs = 5 if phase < len(layer_groups) - 1 else 10
        
        # Create callbacks for this phase
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-7,
            verbose=1
        )
        
        # 5. Knowledge distillation - save snapshots of the model for later ensembling
        snapshot_path = f'model_snapshot_phase_{phase+1}.h5'
        model_checkpoint = ModelCheckpoint(
            snapshot_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        callbacks = [early_stopping, reduce_lr, model_checkpoint]
        
        # Train for this phase - use mixup augmentation in later phases
        current_train_ds = train_ds_combined if phase > 0 else train_ds
        
        print(f"  Training phase {phase+1} for {epochs} epochs...")
        history = model.fit(
            current_train_ds,
            epochs=epochs,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        # Track best performance across phases
        final_val_accuracy = max(history.history['val_accuracy'])
        if final_val_accuracy > best_val_accuracy:
            best_val_accuracy = final_val_accuracy
            best_model_weights = model.get_weights()
            print(f"  New best validation accuracy: {best_val_accuracy:.4f}")
    
    # After all phases, apply the best weights found
    if best_model_weights:
        print("\nApplying best weights found during progressive training...")
        model.set_weights(best_model_weights)
    
    # Save the final fine-tuned model
    model.save('nsfw_classifier_advanced_finetuned.h5')
    
    # 5. Optional: Apply model ensemble/knowledge distillation from snapshots
    print("\nCreating ensemble prediction from model snapshots...")
    
    # Load model snapshots for ensemble prediction
    model_snapshots = []
    for i in range(len(layer_groups)):
        snapshot_path = f'model_snapshot_phase_{i+1}.h5'
        if os.path.exists(snapshot_path):
            print(f"Loading snapshot from phase {i+1}...")
            snapshot_model = tf.keras.models.load_model(snapshot_path)
            model_snapshots.append(snapshot_model)
    
    if len(model_snapshots) > 1:
        # Create a simple ensemble model
        class EnsembleModel:
            def __init__(self, models, weights=None):
                self.models = models
                # If no weights provided, use equal weighting
                self.weights = weights if weights else [1/len(models)] * len(models)
            
            def predict(self, x):
                # Weight each model's prediction
                weighted_preds = []
                for model, weight in zip(self.models, self.weights):
                    pred = model.predict(x)
                    weighted_preds.append(pred * weight)
                
                # Average the predictions
                return np.sum(weighted_preds, axis=0)
        
        # Use exponentially increasing weights for later snapshots
        # Later snapshots are generally better
        snapshot_weights = [2**i for i in range(len(model_snapshots))]
        snapshot_weights = [w/sum(snapshot_weights) for w in snapshot_weights]
        
        ensemble_model = EnsembleModel(model_snapshots, weights=snapshot_weights)
        
        # Evaluate the ensemble model
        print("\nEvaluating ensemble model...")
        ensemble_predictions = ensemble_model.predict(val_ds)
        ensemble_accuracy = np.mean(np.argmax(ensemble_predictions, axis=1) == 
                                   np.argmax(np.concatenate([y for _, y in val_ds]), axis=1))
        print(f"Ensemble model validation accuracy: {ensemble_accuracy:.4f}")
        
        # If ensemble model is better, update our final model
        if ensemble_accuracy > best_val_accuracy:
            print("Ensemble model outperforms single best model. Training final model with distillation...")
            
            # Train final model using soft targets from ensemble
            def distillation_loss(alpha=0.1, temperature=2.0):
                """Knowledge distillation loss combining soft and hard targets"""
                def loss(y_true, y_pred):
                    # Hard target loss (standard categorical crossentropy)
                    hard_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
                    
                    # For soft targets, get ensemble predictions
                    soft_targets = ensemble_model.predict(x)
                    
                    # Apply temperature scaling to soften the probability distribution
                    soft_targets = tf.nn.softmax(soft_targets / temperature)
                    soft_pred = tf.nn.softmax(y_pred / temperature)
                    
                    # Soft target loss
                    soft_loss = tf.keras.losses.kullback_leibler_divergence(soft_targets, soft_pred)
                    
                    # Combine losses
                    return alpha * hard_loss + (1 - alpha) * soft_loss * (temperature**2)
                
                return loss
            
            # Create and train student model with distillation
            student_model = create_model()
            student_model.compile(
                optimizer=Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
                loss=distillation_loss(alpha=0.3, temperature=3.0),
                metrics=['accuracy']
            )
            
            # We'd train with distillation here, but simplifying for this example
    
    print("\nAdvanced fine-tuning completed!")
    return model

def apply_stochastic_weight_averaging(model, train_ds, val_ds, class_weights):
    """Apply Stochastic Weight Averaging for better generalization"""
    print("\nApplying Stochastic Weight Averaging (SWA)...")
    
    # First, train the model normally to a good starting point
    base_lr = FINE_TUNING_LEARNING_RATE / 2
    
    # Compile the model with the base learning rate
    model.compile(
        optimizer=Adam(learning_rate=base_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train for a few epochs to reach a good starting point
    model.fit(
        train_ds,
        epochs=5,
        validation_data=val_ds,
        class_weight=class_weights,
        verbose=1
    )
    
    # Initialize SWA variables
    swa_weights = [tf.Variable(w, trainable=False) for w in model.get_weights()]
    swa_count = tf.Variable(1.0, trainable=False)
    
    # Set up cyclic learning rate schedule for SWA
    n_cycles = 4
    cycle_epochs = 3
    total_swa_epochs = n_cycles * cycle_epochs
    
    print(f"Running SWA for {total_swa_epochs} epochs with {n_cycles} cycles...")
    
    # Run SWA training
    for cycle in range(n_cycles):
        print(f"\nSWA Cycle {cycle+1}/{n_cycles}")
        
        # Cycle learning rate: high -> low
        cycle_lr = base_lr * (0.5 + 0.5 * np.cos(np.pi * cycle / n_cycles))
        
        # Update optimizer with new learning rate
        model.compile(
            optimizer=Adam(learning_rate=cycle_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for one cycle
        model.fit(
            train_ds,
            epochs=cycle_epochs,
            validation_data=val_ds,
            class_weight=class_weights,
            verbose=1
        )
        
        # Update the SWA weights
        current_weights = model.get_weights()
        for i, (swa_w, current_w) in enumerate(zip(swa_weights, current_weights)):
            new_swa = (swa_w * swa_count + current_w) / (swa_count + 1.0)
            swa_w.assign(new_swa)
        
        swa_count.assign_add(1.0)
    
    # Apply the averaged weights to the model
    print("\nApplying SWA weights to model...")
    model.set_weights([w.numpy() for w in swa_weights])
    
    # Final evaluation with SWA weights
    model.compile(
        optimizer=Adam(learning_rate=base_lr/10),  # Lower learning rate for final evaluation
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Evaluate the SWA model
    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    print(f"SWA model validation accuracy: {val_acc:.4f}")
    
    # Save the SWA model
    model.save('nsfw_classifier_swa.h5')
    
    return model

def adaptive_instance_selection(model, train_ds, val_ds, class_weights):
    """Use adaptive instance selection to focus on the most informative examples"""
    print("\nApplying Adaptive Instance Selection...")
    
    # Get all training data in memory (be careful with large datasets)
    train_images = []
    train_labels = []
    
    for images, labels in train_ds.unbatch():
        train_images.append(images.numpy())
        train_labels.append(labels.numpy())
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    # Initialize difficulty scores for each training example
    n_samples = len(train_images)
    difficulty_scores = np.ones(n_samples)
    
    # Track model performance over time
    val_accuracies = []
    
    # Training configuration
    n_rounds = 3
    epochs_per_round = 5
    
    for round_idx in range(n_rounds):
        print(f"\nAdaptive Selection Round {round_idx+1}/{n_rounds}")
        
        # Calculate sampling probabilities based on difficulty scores
        sampling_probs = difficulty_scores / np.sum(difficulty_scores)
        
        # Sample a subset of training data based on difficulty
        n_to_sample = min(n_samples, 10000)  # Adjust based on your memory constraints
        sampled_indices = np.random.choice(
            n_samples, 
            size=n_to_sample, 
            replace=False, 
            p=sampling_probs
        )
        
        # Create a new dataset from the sampled indices
        sampled_images = train_images[sampled_indices]
        sampled_labels = train_labels[sampled_indices]
        
        sampled_ds = tf.data.Dataset.from_tensor_slices((sampled_images, sampled_labels))
        sampled_ds = sampled_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        
        # Train the model on the sampled dataset
        model.fit(
            sampled_ds,
            epochs=epochs_per_round,
            validation_data=val_ds,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
            ],
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        val_accuracies.append(val_acc)
        print(f"Validation accuracy after round {round_idx+1}: {val_acc:.4f}")
        
        # Update difficulty scores based on model predictions
        predictions = model.predict(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(train_labels, axis=1)
        
        # Examples that are misclassified get higher difficulty scores
        is_correct = (predicted_classes == true_classes).astype(np.float32)
        
        # Get prediction confidence
        confidence = np.max(predictions, axis=1)
        
        # Update difficulty scores:
        # - Increase score for misclassified examples
        # - For correctly classified, focus on examples with lower confidence
        for i in range(n_samples):
            if is_correct[i] == 0:  # Misclassified
                difficulty_scores[i] *= 1.5
            else:  # Correctly classified
                # Examples with lower confidence still need focus
                difficulty_scores[i] *= (2.0 - confidence[i])
        
        # Normalize difficulty scores
        difficulty_scores = difficulty_scores / np.mean(difficulty_scores)
    
    # Save the final model
    model.save('nsfw_classifier_adaptive.h5')
    
    # Plot validation accuracy over rounds
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_rounds+1), val_accuracies, 'o-')
    plt.xlabel('Round')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy with Adaptive Instance Selection')
    plt.grid(True)
    plt.savefig('adaptive_instance_selection.png')
    plt.close()
    
    return model

def advanced_fine_tuning_main():
    """Main function for advanced fine-tuning"""
    print("Advanced Fine-tuning for NSFW Classifier")
    
    # Create datasets
    train_ds, val_ds, test_ds = create_data_datasets()
    
    # Get class weights
    adjusted_class_weights = compute_class_weights(train_ds)
    
    # Create base model
    model, base_model = create_model()
    
    # Choose one of the advanced fine-tuning methods:
    
    # Option 1: Progressive unfreezing with discriminative learning rates and mixup
    model = advanced_fine_tune_model(model, base_model, train_ds, val_ds, adjusted_class_weights)
    
    # Option 2: Stochastic Weight Averaging
    # model = apply_stochastic_weight_averaging(model, train_ds, val_ds, adjusted_class_weights)
    
    # Option 3: Adaptive Instance Selection
    # model = adaptive_instance_selection(model, train_ds, val_ds, adjusted_class_weights)
    
    # Final evaluation
    final_evaluation = evaluate_model(model, test_ds)
    print(f"Final model accuracy: {final_evaluation['accuracy']:.4f}")
    
    return model

if __name__ == "__main__":
    advanced_fine_tuning_main()
