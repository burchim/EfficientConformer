# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch

# Models
from models.transducer import Transducer
from models.model_ctc import ModelCTC, InterCTC
from models.lm import LanguageModel

# Datasets
from utils.datasets import (
    LibriSpeechDataset,
    LibriSpeechCorpusDataset
)

# Preprocessing
from utils.preprocessing import (
    collate_fn_pad
)

def create_model(config):

    # Create Model
    if config["model_type"] == "Transducer":

        model = Transducer(
            encoder_params=config["encoder_params"],
            decoder_params=config["decoder_params"],
            joint_params=config["joint_params"],
            tokenizer_params=config["tokenizer_params"],
            training_params=config["training_params"],
            decoding_params=config["decoding_params"],
            name=config["model_name"]
        )

    elif config["model_type"] == "CTC":

        model = ModelCTC(
            encoder_params=config["encoder_params"],
            tokenizer_params=config["tokenizer_params"],
            training_params=config["training_params"],
            decoding_params=config["decoding_params"],
            name=config["model_name"]
        )

    elif config["model_type"] == "InterCTC":

        model = InterCTC(
            encoder_params=config["encoder_params"],
            tokenizer_params=config["tokenizer_params"],
            training_params=config["training_params"],
            decoding_params=config["decoding_params"],
            name=config["model_name"]
        )

    elif config["model_type"] == "LM":

        model = LanguageModel(
            lm_params=config["lm_params"],
            tokenizer_params=config["tokenizer_params"],
            training_params=config["training_params"],
            decoding_params=config["decoding_params"],
            name=config["model_name"]
        )

    else:

        raise Exception("Unknown model type")

    return model

def load_datasets(training_params, tokenizer_params, args):

    # Training Datasets
    training_datasets = {

        "LibriSpeech": {
            "class": LibriSpeechDataset,
            "split": {
                "training": "train",
                "training-clean": "train-clean",
                "validation-clean": None,
                "validation-other": None,
                "test-clean": None,
                "test-other": None,
                "eval_time": None,
                "eval_time_encoder": None,
                "eval_time_decoder": None,
            }
        },

        "LibriSpeechCorpus": {
            "class": LibriSpeechCorpusDataset,
            "split": {
                "training": "train",
                "validation-clean": None,
                "validation-other": None,
                "test-clean": None,
                "test-other": None,
                "eval_time": None,
                "eval_time_encoder": None,
                "eval_time_decoder": None,
            }
        }
    }

    # Evaluation Datasets
    evaluation_datasets = {

        "LibriSpeech": {
            "class": LibriSpeechDataset,
            "split": {
                "training": ["dev-clean", "dev-other"],
                "training-clean": ["dev-clean", "dev-other"],
                "validation-clean": "dev-clean",
                "validation-other": "dev-other",
                "test-clean": "test-clean",
                "test-other": "test-other",
                "eval_time": "dev-clean",
                "eval_time_encoder": "dev-clean",
                "eval_time_decoder": "dev-clean",
            }
        },

        "LibriSpeechCorpus": {
            "class": LibriSpeechCorpusDataset,
            "split": {
                "training": "val",
                "validation-clean": "val",
                "validation-other": "val",
                "test-clean": "test",
                "test-other": "test",
                "eval_time": "val",
                "eval_time_encoder": "val",
                "eval_time_decoder": "val",
            }
        }
    }

    # Select Dataset and Split
    training_dataset = training_datasets[training_params["training_dataset"]]["class"]
    training_split = training_datasets[training_params["training_dataset"]]["split"][args.mode]
    evaluation_dataset = evaluation_datasets[training_params["evaluation_dataset"]]["class"]
    evaluation_split = evaluation_datasets[training_params["evaluation_dataset"]]["split"][args.mode]

    # Training Dataset
    if training_split:

        if args.rank == 0:
            print("Loading training dataset : {} {}".format(training_params["training_dataset"], training_split))

        dataset_train =  training_dataset(training_params["training_dataset_path"], training_params, tokenizer_params, training_split, args)

        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=args.world_size,rank=args.rank)
        else:
            sampler = None

        dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=training_params["batch_size"], shuffle=(not args.distributed), num_workers=args.num_workers, collate_fn=collate_fn_pad, drop_last=True, sampler=sampler, pin_memory=False)
        
        if args.rank == 0:
            print("Loaded :", dataset_train.dataset.__len__(), "samples", "/", dataset_train.__len__(), "batches")
    else:
        dataset_train = None


    # Evaluation Dataset
    if  evaluation_split:

        # Multiple Evaluation datasets
        if isinstance(evaluation_split, list):

            dataset_eval = {}

            for split in evaluation_split:

                if args.rank == 0:
                    print("Loading evaluation dataset : {} {}".format(training_params["evaluation_dataset"], split))

                dataset = evaluation_dataset(training_params["evaluation_dataset_path"], training_params, tokenizer_params, split, args)

                if args.distributed:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size,rank=args.rank)
                else:
                    sampler = None

                dataset = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_eval, shuffle=(not args.distributed), num_workers=args.num_workers, collate_fn=collate_fn_pad, sampler=sampler, pin_memory=False)
                
                if args.rank == 0:
                    print("Loaded :", dataset.dataset.__len__(), "samples", "/", dataset.__len__(), "batches")

                dataset_eval[split] = dataset

        # One Evaluation dataset
        else:

            if args.rank == 0:
                print("Loading evaluation dataset : {} {}".format(training_params["evaluation_dataset"], evaluation_split))

            dataset_eval = evaluation_dataset(training_params["evaluation_dataset_path"], training_params, tokenizer_params, evaluation_split, args)

            if args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset_eval, num_replicas=args.world_size,rank=args.rank)
            else:
                sampler = None

            dataset_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size_eval, shuffle=(not args.distributed), num_workers=args.num_workers, collate_fn=collate_fn_pad, sampler=sampler, pin_memory=False)
            
            if args.rank == 0:
                print("Loaded :", dataset_eval.dataset.__len__(), "samples", "/", dataset_eval.__len__(), "batches")
    else:
        dataset_eval = None
    
    return dataset_train, dataset_eval