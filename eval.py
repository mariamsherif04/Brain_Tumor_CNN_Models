def evaluate_model(model, history, test_ds, model_name):
    """
    Evaluates model on test set and prints training/validation metrics.
    """
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"{model_name} - Test accuracy: {test_acc:.4f}")
    print(f"{model_name} - Test loss: {test_loss:.4f}")

    # Final training & validation metrics
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_acc   = history.history['val_accuracy'][-1]
    final_val_loss  = history.history['val_loss'][-1]

    print(f"{model_name} - Final training accuracy: {final_train_acc:.4f}")
    print(f"{model_name} - Final training loss: {final_train_loss:.4f}")
    print(f"{model_name} - Final validation accuracy: {final_val_acc:.4f}")
    print(f"{model_name} - Final validation loss: {final_val_loss:.4f}")
  
