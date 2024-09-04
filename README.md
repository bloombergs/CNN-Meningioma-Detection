# CNN-Meningioma-Detection
a Tumor(Meningioma) detection made with CNN from kaggle dataset

the project classifies into 2 class: meningioma and normal
![meningiomabrain](https://github.com/user-attachments/assets/ec59395a-6a68-465f-b6e3-c8cd1276bcc7)
![normalbrain](https://github.com/user-attachments/assets/65d2b679-a889-4600-9e19-57f0027a905f)

the project contains 15 image for each class,
images are processed using ETL,load the image;i normalized pixel value using reshape=1/255;change every image size to 200,200;set the class mode to categorical so we can use categorical_crossentropy for the loss function.

make the model,first layer we use conv2D for convution2D,16 neuron,inputshape we set to 200,200 with 3 rgb value and we use relu for the activation function;second layer we use maxpooling2d to lower the shape;and then repeat it 2 more time to make it even smaller;flatten the layer so we can use dense layer;and lastly for the last layer we use dense layer with softmax activation function.

compile the model,using adam as a optimizer;categorical_crossentropy as a loss function.

fit the data,put the trainflow data;10 epochs;put the validationflow for the validation data

make prediction,we iterate and split the testflow to into testimage and testlabel;before we predict we need to expand the dimension for batch size;and then we display the prediction and actual label.

plot the image,to plot the image we use pyplot from matplotlib library.

save the model,save the model using .h5 format.
