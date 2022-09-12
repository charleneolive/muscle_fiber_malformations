# Quantification of Muscle Fiber Malformations Using Edge Detection to Investigate Chronic Wound Healing

This is the github repository for the quantification approaches for muscle fiber malformations using edge detection.

## Running the Code

1. Clone the RCF repo

    ```
    git clone https://github.com/yun-liu/RCF-PyTorch.git
    ```
2. Download the pretrained model from the RCF repository.

3. Add the repository to the system path in `01_test_RCF.py`.

4. Edit the relevant parameters under the `Args` class and run `01_test_RCF.py`

5. Postprocess with the Piotr's Structured Forest matlab toolbox.

6. Edit the relevant paths in `path`, `original_path` and `save_path` in 02_edge_detection_nms.py and run `02_edge_detection_nms.py`.


## References

    @article{liu2019richer,
      title={Richer Convolutional Features for Edge Detection},
      author={Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Bian, Jia-Wang and Zhang, Le and Bai, Xiang and Tang, Jinhui},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      volume={41},
      number={8},
      pages={1939--1946},
      year={2019},
      publisher={IEEE}
    }

    
    @inproceedings{DollarICCV13edges,
      author    = {Piotr Doll\'ar and C. Lawrence Zitnick},
      title     = {Structured Forests for Fast Edge Detection},
      booktitle = {ICCV},
      year      = {2013},
    }

    @article{DollarARXIV14edges,
      author    = {Piotr Doll\'ar and C. Lawrence Zitnick},
      title     = {Fast Edge Detection Using Structured Forests},
      journal   = {ArXiv},
      year      = {2014},
    }

    @inproceedings{ZitnickECCV14edgeBoxes,
      author    = {C. Lawrence Zitnick and Piotr Doll\'ar},
      title     = {Edge Boxes: Locating Object Proposals from Edges},
      booktitle = {ECCV},
      year      = {2014},
    }
