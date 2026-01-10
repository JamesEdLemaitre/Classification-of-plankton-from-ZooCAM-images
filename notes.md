# Step 1: Starting the challenge

To start the challenge, I decided to make a minimal working version. I only modified a bit the VanillaCNN into a VanillaCNN_V2 version to make it a bit more customizable. I trained the model and easily got a f1-score around 55% on the Kaggle public validation set. 

I decided not to lose more time on this model, as I didn't expect this version to give the best results. 

# Step 2: Foundation model: dinov3

I had heard about foundation models but never really had the time to delve into the subject. I decided to go for Dinov3 as I saw in the news that this model was recently published and had very good performances. It also allowed me to use the "small" version with only about 21M paramters, which fits on small GPUs. I used dinov3 as a backbone and added a simple classification head. 

I first decided to just use the basic training loop, with very basic input transformations and a small classification head. It gave me very good results, after a few iterations I was able to reach 70+% for f1-score. 

I also tried to make a more complex training loop in two phases. The phase 1 trains the head, then the back bone is trained with a smaller learning rate, fully or partially. I had similar results to the first approach, which make me thinks that I didn't make an optimal code. I decided not to lose more time on this and to come back if other approaches will not improve the ~74% f1-score reached on the Kaggle public validation set.

# Step 3: Timm mainstream models

My code allowed me to launch trainings very fast using ResNext-50, EfficientNet-b3 and EfficientNet-b5. These models might allowed me to reach 65% easily, some of the training reached 70%. I keep this result in mind, knowing that the training could be improved but won't spend more time on this for now, as I want to try other approaches.

# Step 4: Dataset reannotation and more complex pipeline

During the precedent steps, I made codes to extract information from the training and display example images of the classes and the predicted classes. It allowed me to notice that there are (probably, I'm not a plankton master) annotations mistakes in quantities that might affect results. I also noticed that many classes look the same, and that the models I trained until now were not too bad at understanding what a plankton looks like, but often confused a class with a very similar one. It can be difficult even for an untrained human to decide between two classes to classify an image. 

I thus decided to make bigger classes and train a model to classify the images as looking like:
- ... 

Using a dinov3 backbone, I was able to train a model with a f1-score > 85% and an accuracy > 94%. I decided that it was good enough, as I would probably be able to improve this score in the future. 

# Step 5: Real Foundation model

As the results given by the precedent step didn't achieve a better score (very close to the first score however), I want to follow the advice in the course slides: "We need data". Using my knowledge from my apprenticeship, I know that foundation models can be used to train on huge amount of data, regardless of wether this data is labelled neither if the labels are the same as the ones in the challenge dataset. 

I gathered 3 additionnal datasets, which should allow us to have a final dataset containing a much bigger amount of images from various sources, forcing the model to learn meaningful features to understand planktons. 

# To to also:

- Try the same or a similar approach to "Deep Learning Classification of Lake ZooPlankton"
Including: Try to extract features in a Dataframe using more traditionnal image processing tools  (Done, but needs much more tuning, see scripts/image_proc_feature_extraction)
