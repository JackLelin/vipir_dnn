# Unet_vipir
In this branch, the model itself remains largely the same as the prelim version, Transformer+Unet. 
But the loss function is changed. Instead of a segmentation problem, the extraction of traces is modeled as a regression problem. 
Two things are different
1. The DNN outputs two vectors, the first vector represent the presence of the trace/reflected pulse. The second vector represent the virtual height of reflection, or the virtual height of the pulse. Two vectors for O-trace and two vectors for X-trace. So 4 vectors in total
2. The loss function is changed. Now the probablity vector uses Cross entropy, and the virtual height vector uses mean square loss.

Compare to the prelim-version of the T-UNet, the output layer is connected to a linear layer to make the output a two vectors. 

The dataloader and training data generator will be modified accordingly. 
