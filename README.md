This repository stores open-source codes for the publication: [Recognizing hand use and hand role at home after stroke from egocentric video](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000361).  
In the publication, three machine learning models, including a random forest classifier, SlowFast network, and Hand Object Detector, were trained to identify hand-object interaction in daily living for stroke survivors.  
The inputs for the three models were the images of cropped hand regions of the detected hand bounding boxes using [Hand Object Detector](https://github.com/ddshan/hand_object_detector). 

# Hand-Object Interaction and Hand Role Classification using SlowFast Network  
## 1. Please clone the original SlowFast GitHub and replace the original _slowfast_ folder with the _slowfast_ folder in this repository.
This repository was modified from the [SlowFast network for MECCANO dataset](https://github.com/fpv-iplab/MECCANO) and re-trained the model to classify interaction and hand roles of stroke survivors.  
## 2. Download pre-trained MECCANO weights  
Please download the weights from [here](https://drive.google.com/drive/folders/1cc71O4XT0etAUXAAbIXLQANWnE7LHPnn?usp=sharing) and put them under `slowfast/Interaction` folder.  
## 3. Prepare datasets and labels
The dataset structure, including images and labels, can be found in [interaction.py](https://github.com/mft2023/SlowFast/blob/main/slowfast/datasets/interaction.py).
Raw image data was stored in _Interaction_/_frames_/{dataset name}/{participant id}/{video id}/images.  
## 4. Decide hypperparameters in the yaml file
The training hyperparameters for interaction detection and hand role classification can be found in [SLOWFAST_8x8_R50_Interaction.yaml](https://github.com/mft2023/SlowFast/blob/main/slowfast/Interaction/SLOWFAST_8x8_R50_Interaction.yaml) and [SLOWFAST_8x8_R50_Manipulation.yaml](https://github.com/mft2023/SlowFast/blob/main/slowfast/Interaction/SLOWFAST_8x8_R50_Manipulation.yaml), respectively.

Training and testing modes can be enabled in the yaml files.  
Launch training and testing with:
```
python tools/run_net.py --cfg path_to_Interaction.yaml_file
```

# Cite
If you find this repository useful in your research, please consider citing:
```
@article{
    Author = {Meng-Fen Tsai,Rosalie H. Wang, and Zariffa, José},
    Title = {Recognizing hand use and hand role at home after stroke from egocentric video},
    Journal = {PLOS Digital Health 2.10: e0000361},
    Year = {2023}
}
```
