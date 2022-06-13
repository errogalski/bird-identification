# bird-identification
CNN model and Streamlit app that identifies the birds of North America

## Problem Statement
Building community interest in local wildlife is a great way to protect and preserve the natural beauty of the planet. Birds are an important part of every ecosystem, and many species are rare or endangered. My goal is to build a convolutional neural network that will identify as the bird species of North America in the hopes that it can be used to make science more accessible. It could eventually be paired with cameras at feeders and used to identify and then notify people which birds are visiting their feeders so that they don’t miss a special guest. 

## The Data 
The dataset was obtained from the Cornell Lab of Ornithology. It contained over 48,000 images split into 555 classes. 

## Pre-processing
The images in the dataset were provided at varying sizes and with the bird taking up varying amounts of the frame. In order to better train the model on the differences specific to the birds, all of the images were cropped using bounding box definitions provided in the dataset. Using the Keras image data preprocessing library I also applied image augmentations prior to training. Both a randomized vertical flip and rotation were applied to the images in order to better train for the various positions birds may be photographed in.

## The Model
The best performing model was a CNN model that used transfer learning. I used EfficientNet - B0 and 224 x 224 resolution. The model contained 555 classes, 371 layers, and performed with an accuracy of 81% after 200 epochs.

## Analysis
With the very fine details differentiating bird species it was found that the greater detail in the model, the better the results. Early iterations of the model attempted to use the VGG-16 then the Inception-V3 architectures. Both of these model were good for early attempts since they ran fairly quickly but due to their relative simplicity (especially VGG-16) they performed poorly. Efficient Net proved to yield the best results but was much more computationally expensive.

B0 with a resolution of 224x224 were selected because the higher resolution inputs and models like B7 ran into memory allocation issues on the computer I was using. Unfortunately Google Colab was limited by the size of the dataset and wasn't able to run any faster because of how long data transfer took.

Even with a lower resolution the model was able to achieve an accuracy score of 81%. Checking the outputs more closely I was able to find that generally images that were not correctly identified scored with a fairly low confidence and often times the model's second or third predicition would be correct. This was especially apparent with extremely similar looking birds such as trying to distinguish an American crow from a fish crow. 

## Conclusion and Next Steps
As a relatively early model the results from my bird identification CNN have been very promising. Using streamlit to display the top five results consistently gives the correct identification even on birds images that it struggles with. Because the dataset provided more images for birds that were more common, the model does a good job of weighing common birds species more heavily than similar looking rare species. However the results could likely be improved in future iterations even more by providing a location the picture was taken to weigh results based on location.

Eventually I would like to combine this with a object detection model that can ingest streamed video data so a user can set up a camera to watch their feeder and be alerted when a specific type of bird is spotted at their feeder. This could potentially be useful for both commercial users to run out and check their home feeders or to researchers interested in researching a specific type of bird and use this application to trigger recordings only of the bird they're interested in rather than needing to sift through hours of footage.
