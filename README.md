# EQP_training_script
Training script for loading data and training and evaluating model.

When the training data isn't chunked to a size below ~1,000,000 points there appears to be some CPU bound operation in the loss.backward() call which slows down training
even though batch sizes are kept the same if the data is chunked smaller.
