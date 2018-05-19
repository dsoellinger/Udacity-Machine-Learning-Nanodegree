# Convolutional Neural Networks

## Model Validation

We typically break our dataset into three sets:

- Train Set
  We use this set for fitting the model's weights
  
- Validation Set
  Tells us if our chosen model is performing well. Usually evaluated in each epoch.
  Since the model is not used for deciding the weights, we can use it to detect overfitting.
  
- Test Set
  The idea of the test set is that when we go to test the model, we look at data we have truly never seen before. 
  Although the validation data are not used to train the model, the model is biased in favor of the validation set.
  
 
## Image Augmentation

We want our network to learn an invariant representation of our objects. This means that the position of an object should not matter.
In other words we want to have:

- Scale invariance
  Independence of the object's size
- Rotation invariance
  Rotation of the object should not matter

Combined both terms together we call this: Translation invariance