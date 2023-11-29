# SlowFast
***SlowFast Network for Hand Object Interaction Detection and Hand Role Classification***  
This repository took the weights from the [SlowFast network for MECCANO dataset](https://github.com/fpv-iplab/MECCANO) and re-trained the model to classify the interaction and hand roles of stroke survivors.  
Please download the files from [here](https://drive.google.com/drive/folders/1cc71O4XT0etAUXAAbIXLQANWnE7LHPnn?usp=sharing) and put them under _slowfast/Interaction_.  

Images can be stored under _slowfast/Interaction/frames/dataset_name/participant_id/video_id/_ and their text files containing hand bounding boxes were stored under _Shan_bbx_.  
The dataset information can be found [interaction.py](https://github.com/mft2023/SlowFast/blob/main/slowfast/datasets/interaction.py) and [SLOWFAST_8x8_R50_Interaction.yaml](https://github.com/mft2023/SlowFast/blob/main/slowfast/Interaction/SLOWFAST_8x8_R50_Interaction.yaml).
