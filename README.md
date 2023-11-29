This repository stores open-source codes for the publication: [Recognizing hand use and hand role at home after stroke from egocentric video](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000361).  
***In the publication, three machine learning models, including a random forest classifier, SlowFast network, and Hand Object Detector, were trained to identify hand-object interaction in daily living for stroke survivors. This repository includes SlowFast network.***  
The inputs for the three models were the images of cropped hand regions of the detected hand bounding boxes using [Hand Object Detector](https://github.com/ddshan/hand_object_detector). 

# 2. SlowFast
This repository was modified from the [SlowFast network for MECCANO dataset](https://github.com/fpv-iplab/MECCANO) and took the weight to re-train the model to classify the interaction and hand roles of stroke survivors.  
Please download the files from [here](https://drive.google.com/drive/folders/1cc71O4XT0etAUXAAbIXLQANWnE7LHPnn?usp=sharing) and put them under _slowfast/Interaction_.  

Images can be stored under _slowfast/Interaction/frames/dataset_name/participant_id/video_id/_ and their text files containing hand bounding boxes were stored under _Shan_bbx_.  
The dataset structure can be found in [interaction.py](https://github.com/mft2023/SlowFast/blob/main/slowfast/datasets/interaction.py) and the training hyperparameters can be found in [SLOWFAST_8x8_R50_Interaction.yaml](https://github.com/mft2023/SlowFast/blob/main/slowfast/Interaction/SLOWFAST_8x8_R50_Interaction.yaml).
