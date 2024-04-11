# DBP_ChrisDoe_NeuriteTracing

**11/30/2023**

**Omnipose - 3D**

**Aim:**

To train an Omnipose 3D model to segment the neurons. The previous attempts of 2D omnipose did not result in good segmentation.  

- [ ] Create a notebook probably use ZeroCostDL4Mic as a template?
- [ ] Figure out which images to use 
- [ ] Check the cropping dimensions
- [ ] Run the notebooks on DGX 

**Training details:**

* The documentation mentions that the training and preprocessing need to be done on two separate notebooks/scripts.
* Training should be done only via CLI.
* Find the folder on Google Drive and the masks for all the images. 
* Image dimensions need to be understood well before training 
* Ground truth needs to be labeled matrices and not binary masks 

**Questions:**

* Image dimensions
* diameter
* --nclasses - 2 or 3 

Mario's suggestions for the above questions were,
* He suggested 50*50*50 image size and he has cropped single neurons from the images. The whole image contains 3-5 neurons 

**Training:** 

Attempts with ZeroCostDL4Mic did not work out because of 'no GPU access'. I also got an error that 'Patch size is greater than the source image'. I did not get this error when I unchecked the 'Default parameters' where the dimensions of the patch size were bigger than the source image. 

On changing that, I receive the following error - ' ValueError: A `Concatenate` layer requires inputs with matching shapes except for the concatenation axis. Received: input_shape=[(None, 24, 24, 24, 64), (None, 25, 25, 25, 128)]' 

The above error was set right when the image was given in dimensions of equal height and width. 


Somehow my google account has been banned from using the GPU allocation. Works well on my personal google account and also in Nodar's account. 
Now that I don't have GPU access. Probably I should try using the local GPU and in a Python environment. 

I re-created the python env for omnipose.  Just 'pip install omnipose' did not work as I was getting an error - 'cannot import ifft from scipy' hence I installed the latest source of omnipose by cloning the git. This helped in avoiding the above error. 

When I tried to train a model, I got the following errors, 

* `from aicsimageio import AICSImage` cannot be found so I used the following command - `pip install --upgrade --force-reinstall aicsimageio` This helped to fix the above error. 
* Then the I got the following error - `ModuleNotFoundError: No module named 'torch_optimizer`. This was fixed with the following - `pip install torch_optimizer`. 

After this, I was able to run the following command - `python -m omnipose --train --use_gpu --dir C:\Users\ssivagur\Documents\Projects\DBP\Doe\ModifiedInput --mask_filter _masks --n_epochs 400 --pretrained_model None  --learning_rate 0.1 --save_every 50 --save_each  --verbose  --look_one_level_down --all_channels --dim 3 --RAdam --batch_size 4 `

The above was running perfectly but I cancelled it since it was test run. I gave the following - `python -m omnipose --train --use_gpu --dir C:\Users\ssivagur\Documents\Projects\DBP\Doe\trainingData --mask_filter _masks --n_epochs 2 --pretrained_model None  --learning_rate 0.1 --save_every 50 --save_each  --verbose  --all_channels --dim 3 --RAdam --batch_size 4` but I got the following error, 

![image](https://github.com/broadinstitute/ssivagur/assets/64338533/2a06869d-b383-4c2e-9496-f48f47d27bc0)

Probably the above error was because I tried to train 4 images and also they were whole images. 

Next attempt - with single neuron images and 2 epochs and got the same error. 

Next attempt - tried it with a couple of images but even then I was getting the out-of-memory error. It works only with 1 pair of images. 

Trained a model from scratch using 'python -m omnipose --train --use_gpu --dir C:\Users\ssivagur\Documents\Projects\DBP\Doe\trainingData --mask_filter _masks --n_epochs 100 --pretrained_model None  --learning_rate 0.1 --save_every 50 --save_each  --verbose  --all_channels --dim 3 --RAdam --batch_size 1` and re-trained the model with new set of images. 

**Evaluation:**
I was not able to evaluate the model because of the 'permission error' but this was set right when run as an administrator. Resulted in an  [OS error](https://github.com/kevinjohncutler/omnipose/issues/82) when run as an administrator. 

The saved `_seg.npy` were opened with the following, 

`image = np.load('c:\\Users\\ssivagur\\Documents\\Projects\\DBP\\Doe\\omniposeOutput\\v_seg.npy', allow_pickle=True).item()['masks']`

The `_seg.npy` files are saved as a dictionary with images, masks, outlines, flows (and more). 

**With varied threshold settings of doeModel3 (dated 2022-10-06):**

I was not able to get the model that is shown on the document hence I chose the latest one after the date of 2022-09-30. 
Green - raw image; red - UNet predictions
![image](https://github.com/broadinstitute/ssivagur/assets/64338533/5986282d-1a2d-4c6f-8ab3-06277cc426cb)
![image](https://github.com/broadinstitute/ssivagur/assets/64338533/778dcdd6-57e8-499f-9d2e-08f03742e7f0)
![image](https://github.com/broadinstitute/ssivagur/assets/64338533/a921768e-0e49-404e-a8b6-c06d2c46b24c)

Lowering the threshold setting did not make it better at least in the above model that I tested. 

** Trying different settings in the APP:**

**APP1**

Default settings:
![image](https://github.com/broadinstitute/ssivagur/assets/64338533/8c460537-d83e-4590-9306-8a8877bae41a)
![image](https://github.com/broadinstitute/ssivagur/assets/64338533/19a74c69-f162-4774-99fc-a96a20772e5c)


More trials on altering the threshold of the different models that were available and also changing the threshold settings of the APP method can be found [here](https://docs.google.com/document/d/1qXLjTghT5RDeixhsCw1ibhmiC1gLLy8_Cs0kE93WZPc/edit). 

### **Next steps discussed:**
Since the previous models and the APP did not perform as expected, these are some of the next steps discussed in the order of priority 
* Try some preprocessing with APP2
* Image augmentations in 3D UNet 
* Weight maps - make in a such way that the model pays more attention to the thin structures than the thicker ones - distance transform; gives less importance to the thicker structures; more importance to the thinner ones and the middle ones stay somewhere in between. 
* Come up with new ground truth images 

### **Attempts:** 

* Preprocessing attempted on the neurite images - cropping and contrast enhancement 
* Image augmentations need to be improved 
