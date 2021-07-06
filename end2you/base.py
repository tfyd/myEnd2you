import torch
import torch.optim as optim
import logging
import json 

from torch import nn
from pathlib import Path


class BasePhase:
    """ Base class for train/eval models."""
    
    def __init__(self,
                 model:nn.Module,
                 ckpt_path:str = None,
                 visual_ckpt_path:str = None,
                 audio_ckpt_path:str = None,
                 freeze_visual: bool = False,
                 freeze_audio: bool = False,
                 optimizer:optim = None):
        """ Initialize object class.
        
        Args:
            model (torch.nn.Module): Model to train/evaluate.
            ckpt_path (str): Path to the pretrain model.
            optimizer (torch.optim): Optimizer to use for training.
        """
        
        self.optimizer = optimizer
        self.model = model
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.visual_ckpt_path = Path(visual_ckpt_path) if visual_ckpt_path else None
        self.audio_ckpt_path = Path(audio_ckpt_path) if audio_ckpt_path else None
        self.freeze_visual = freeze_visual if visual_ckpt_path else False
        self.freeze_audio = freeze_audio if audio_ckpt_path else False

    
    def load_checkpoint(self):
        """ Loads model parameters (state_dict) from file_path. 
            If optimizer is provided, loads state_dict of
            optimizer assuming it is present in checkpoint.
        
        Args:
            checkpoint (str): Filename which needs to be loaded
            model (torch.nn.Module): Model for which the parameters are loaded
            optimizer (torch.optim): Optional: resume optimizer from checkpoint
        """
        logging.info("Restoring model from [{}]".format(str(self.ckpt_path)))
        
        if not Path(self.ckpt_path).exists():
            raise Exception("File doesn't exist [{}]".format(str(self.ckpt_path)))

        if torch.cuda.is_available():
            checkpoint = torch.load(str(self.ckpt_path))
        else:
            checkpoint = torch.load(str(self.ckpt_path), map_location='cpu')

        self.model.load_state_dict(checkpoint['state_dict'])
        
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optim_dict'])
        
        return checkpoint

    def load_visual_checkpoint(self):
        """ Loads model parameters (state_dict) from file_path.
            If optimizer is provided, loads state_dict of
            optimizer assuming it is present in checkpoint.

        Args:
            checkpoint (str): Filename which needs to be loaded
            model (torch.nn.Module): Model for which the parameters are loaded
            optimizer (torch.optim): Optional: resume optimizer from checkpoint
        """
        logging.info("Restoring model from [{}]".format(str(self.visual_ckpt_path)))

        if not Path(self.visual_ckpt_path).exists():
            raise Exception("File doesn't exist [{}]".format(str(self.visual_ckpt_path)))

        if torch.cuda.is_available():
            checkpoint = torch.load(str(self.visual_ckpt_path))
        else:
            checkpoint = torch.load(str(self.visual_ckpt_path), map_location='cpu')

        print(checkpoint['state_dict'].keys())
        model_dict = self.model.state_dict()
        print(model_dict.keys())
        # visual_ckpt_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict and k.startswith('visual')}
        visual_ckpt_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}

        print(visual_ckpt_dict.keys())

        model_dict.update(visual_ckpt_dict)
        self.model.load_state_dict(model_dict)

        if self.freeze_visual:
            for name, value in self.model.named_parameters():
                if name.startswith('visual'):
                    value.requires_grad = False


        # if self.optimizer:
        #    self.optimizer.load_state_dict(checkpoint['optim_dict'])

        return checkpoint

    def load_audio_checkpoint(self):
        """ Loads model parameters (state_dict) from file_path.
            If optimizer is provided, loads state_dict of
            optimizer assuming it is present in checkpoint.

        Args:
            checkpoint (str): Filename which needs to be loaded
            model (torch.nn.Module): Model for which the parameters are loaded
            optimizer (torch.optim): Optional: resume optimizer from checkpoint
        """
        logging.info("Restoring model from [{}]".format(str(self.audio_ckpt_path)))

        if not Path(self.audio_ckpt_path).exists():
            raise Exception("File doesn't exist [{}]".format(str(self.audio_ckpt_path)))

        if torch.cuda.is_available():
            checkpoint = torch.load(str(self.audio_ckpt_path))
        else:
            checkpoint = torch.load(str(self.audio_ckpt_path), map_location='cpu')

        print(checkpoint['state_dict'].keys())
        model_dict = self.model.state_dict()
        print(model_dict.keys())
        audio_ckpt_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.startswith('audio')}
        # audio_ckpt_dict = {k: v for k, v in checkpoint['state_dict'].items()}

        print(audio_ckpt_dict.keys())
        new_audio_dict = {}
        for k in audio_ckpt_dict.keys():
            if k.startswith('audio_model.c'):
                new_key = 'audio_model.model' + k[11:]
                new_audio_dict[new_key] = audio_ckpt_dict[k]
            else:
                new_audio_dict[k] = audio_ckpt_dict[k]
        print(new_audio_dict.keys())
        model_dict.update(new_audio_dict)
        self.model.load_state_dict(model_dict)
        # if self.optimizer:
        #     self.optimizer.load_state_dict(checkpoint['optim_dict'])
        if self.freeze_audio:
            for name, value in self.model.named_parameters():
                if name.startswith('audio'):
                    print("freeze ", name)
                    value.requires_grad = False

        return checkpoint
    
    def _save_dict_to_json(self, 
                           dictionary:dict, 
                           json_path:str):
        """ Saves dict of floats in json file
        
        Args:
            dictionary (dict): of float-castable values (np.float, int, float, etc.)
            json_path (string): path to json file
        """
        
        with open(json_path, 'w') as f:
            json.dump(dictionary, f, indent=4)
    