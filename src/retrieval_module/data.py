import tarfile
from huggingface_hub import hf_hub_download
import webdataset as wds
from braceexpand import braceexpand
import ujson
import os
import pandas as pd
from PIL import Image
import faiss
import torch
from datasets import load_dataset

import mmh3


_original_torch_load = torch.load
def patched_load(args, **kwargs):
    kwargs['mmap'] = False
    return _original_torch_load(args, **kwargs)
torch.load = patched_load
import json
import argparse
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, CLIPTokenizer, CLIPImageProcessor, AutoImageProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import glob
import csv
from datasets import load_from_disk, concatenate_datasets

from cropper_module.cropper import highlight_entity

CURRENT_ID=0

def load_csv_data(test_file):
    test_list = []
    with open(test_file, "r") as f:
        reader = csv.reader(f)
        test_header = next(reader)
        for row in reader:
            try: 
                # if (row[test_header.index("question_type")] == "automatic" or row[test_header.index("question_type")] == "templated" or row[test_header.index("question_type")] == "multi_answer" or row[test_header.index("question_type")] == "infoseek"): 
                test_list.append(row)
            except:
                # print row and line number
                print(row, reader.line_num)
                raise ValueError("Error in loading csv data")
    return test_list, test_header

def get_test_question(preview_index, test_list, test_header):
    return {test_header[i]: test_list[preview_index][i] for i in range(len(test_header))}

class ViquaeDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        # print(f'Loading from {args.query_path}...')
        with open('/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/viquae_dataset/test.jsonl', 'r') as f:
            self.data = []
            for line in f:
                self.data.append(json.loads(line))
        print('Loading completed')

        self.args = args

        self.query_images_root = "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/viquae_images/images"

    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample['image']
        assert not os.path.isabs(image_path)
        image_path = os.path.join(self.query_images_root, image_path)
        sample['image_path'] = image_path
        if os.path.exists(image_path):
            image_query = Image.open(image_path).convert('RGB')
        else:
            # Black Image
            image_query = Image.new('RGB', (224, 224), color=(0, 0, 0))
            print(f"[!] Warning: image path {image_path} not found. Using black image as placeholder.")

        sample['image_query'] = image_query
        sample['question'] = sample['original_question']
        sample['wikipedia_url'] = sample['url']
        sample['data_id'] = sample['id']
        sample['reference'] = sample['output']['answer']

        return sample
    
    def __len__(self):
        return len(self.data)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print(f'Loading from {args.query_path if not hasattr(args, "use_omgm") or not args.use_omgm else args.query_path_omgm}...')
        with open(args.query_path, 'r') as f:
            self.data = ujson.load(f)

        # Optional: load OMGM csv if requested (existing behavior)
        if hasattr(args, 'use_omgm') and args.use_omgm:
            self.omgm_data, self.header = load_csv_data(args.query_path_omgm)
            self.cineca_indexed_data = {}
            for item in self.data:
                self.cineca_indexed_data[item["dataset_image_ids"]] = item
        
        # Optional: load a JSON with precomputed crop info (all_results_merged.json or custom)
        # The JSON can be either:
        #  - a list of entries with 'dataset_image_ids' and 'bbox' fields, or
        #  - a dict keyed by dataset_image_ids to entry
        self.cropped_map = {}
        try:
            # precedence: explicit arg -> conventional merged path -> no map
            cropped_path = None
            if hasattr(args, 'cropped_json_path') and args.cropped_json_path:
                cropped_path = args.cropped_json_path
            else:
                # default consolidated results path
                default_path = "/leonardo_scratch/large/userexternal/mmorini0/bbox_filtering_results/all_results_merged.json"
                if os.path.exists(default_path):
                    cropped_path = default_path

            if cropped_path and os.path.exists(cropped_path):
                print(f"Loading cropped map from {cropped_path}")
                with open(cropped_path, 'r') as f:
                    loaded = json.load(f)
                # Normalize into dict keyed by dataset_image_ids
                if isinstance(loaded, dict):
                    # assume already keyed
                    self.cropped_map = {str(k): v for k, v in loaded.items()}
                elif isinstance(loaded, list):
                    for entry in loaded:
                        # try multiple possible key fields
                        key = None
                        for kname in ('dataset_image_ids', 'dataset_image_id', 'image_id', 'dataset_image_id|0'):
                            if kname in entry:
                                key = entry[kname]
                                break
                        # fallback: try 'image_path' or 'image_id' in entry
                        if key is None:
                            key = entry.get('dataset_image_ids') or entry.get('image_id') or entry.get('image_path')
                        if key is not None:
                            self.cropped_map[str(key)] = entry
                print(f"Loaded cropped_map with {len(self.cropped_map)} entries")
        except Exception as e:
            print(f"Warning: failed loading cropped_json_path: {e}")
            self.cropped_map = {}

        # NOTE: Attention bboxes are now computed dynamically in retrieval.py
        # The following code is commented out but kept for reference
        # if hasattr(args, 'attention_bboxes') and args.attention_bboxes:
        #     try:
        #         with open(args.attention_bboxes, 'r') as f:
        #             self.attention_bboxes = json.load(f)
        #         print(f"Loaded attention bboxes for {len(self.attention_bboxes)} samples")
        #     except Exception as e:
        #         print(f"Warning: failed loading attention_bboxes: {e}")
        #         self.attention_bboxes = None
        self.attention_bboxes = None  # Placeholder for dynamic computation

        print(f'Loading completed...')
        self.start_idx = 0

    def split(self, start_idx, end_idx):
        self.start_idx = start_idx
        if hasattr(self.args, 'use_omgm') and self.args.use_omgm:
            self.omgm_data = self.omgm_data[start_idx:end_idx]
        else:
            self.data = self.data[start_idx:end_idx]

    def __getitem__(self, idx):
        if hasattr(self.args, 'use_omgm') and self.args.use_omgm:
            sample = get_test_question(idx, self.omgm_data, self.header)
            if 'unique_id' in sample:
                sample['unique_id'] = sample['unique_id']
            else:    
                sample['unique_id'] = "E-VQA_{}".format(self.start_idx + idx)
            cineca_sample = self.cineca_indexed_data[sample["dataset_image_ids"].split("|")[0]]
            image_path = cineca_sample["related_images"]
            if 'retrieval' in cineca_sample and 'evidence_section_id' in cineca_sample:
                sample['retrieval'] = cineca_sample['retrieval']
                sample['evidence_section_id'] = cineca_sample['evidence_section_id']
        else:
            sample = self.data[idx]
            image_path = sample["related_images"]
        
        sample['image_path'] = image_path

        if os.path.isabs(image_path) and os.path.exists(image_path):
            image_query = Image.open(image_path).convert('RGB')
        else:
            # Black Image
            image_query = Image.new('RGB', (224, 224), color=(0, 0, 0))
            print(f"[!] Warning: image path {image_path} not found. Using black image as placeholder.")

        sample['image_query'] = image_query

        cropped_img, annotated_image = self._get_cropped_image(sample)

        # if the area is less than self.args.max_bbox_ratio, then keep the cropped image
        crop_area = cropped_img.size[0] * cropped_img.size[1] if cropped_img is not None else 0
        image_area = image_query.size[0] * image_query.size[1]
        sample['bbox_ratio'] = crop_area / image_area
        if hasattr(self.args, 'max_bbox_ratio') and self.args.max_bbox_ratio is not None and float(sample['bbox_ratio']) > self.args.max_bbox_ratio:
            cropped_img = None

        sample['image_cropped'] = cropped_img
        sample['annotated_image'] = annotated_image

        return sample
    
    def __len__(self):
        if hasattr(self.args, 'use_omgm') and self.args.use_omgm:
            return len(self.omgm_data)
        return len(self.data)
    
    def _get_cropped_image(self, sample):
        image_query = sample['image_query']

        cropped_img = None
        annotated_image = None
        
        # NOTE: Attention bbox logic has been moved to retrieval.py for dynamic computation
        # When use_attention_bbox is True, bboxes are computed dynamically during inference
        # The code below is commented out but kept for reference
        # if hasattr(self.args, 'use_attention_bbox') and self.args.use_attention_bbox:
        #     if hasattr(self.args, 'same_grounding_dino_setting') and self.args.same_grounding_dino_setting:
        #         if hasattr(self, 'cropped_map') and self.cropped_map:
        #             key = sample.get('dataset_image_ids')
        #             cropped_entry = None
        #             if key is not None:
        #                 cropped_entry = self.cropped_map.get(str(key))
        #                 if cropped_entry is None and '|' in str(key):
        #                     cropped_entry = self.cropped_map.get(str(key).split('|')[0])
        #             if not cropped_entry:
        #                 return None, None
        #     try:
        #         attention_bbox = self.attention_bboxes.get(sample['unique_id'])[self.args.attention_bbox_method]
        #         sample['bbox_2d'] = attention_bbox
        #         cropped_img = image_query.crop(attention_bbox)
        #         annotated_image, _ = highlight_entity(image_query, attention_bbox)
        #     except Exception as e:
        #         print(f"Warning: failed processing attention bbox for {sample['unique_id']}: {e}")
        #         cropped_img = None
        #         annotated_image = None
        
        # Skip cropped_map loading when use_attention_bbox is True (handled dynamically in retrieval.py)
        if hasattr(self.args, 'use_attention_bbox') and self.args.use_attention_bbox:
            return None, None
        
        # Original cropped_map logic for non-attention-bbox cases
        if True:  # Keep original else block logic
            try:
                if hasattr(self, 'cropped_map') and self.cropped_map:
                    key = sample.get('dataset_image_ids')
                    # try direct match, then try first part before '|' if present
                    cropped_entry = None
                    if key is not None:
                        cropped_entry = self.cropped_map.get(str(key))
                        if cropped_entry is None and '|' in str(key):
                            cropped_entry = self.cropped_map.get(str(key).split('|')[0])
                    if cropped_entry:
                        bbox = cropped_entry.get('bbox')
                        if bbox and len(bbox) >= 4:
                            # bbox may be floats; convert to int crop coords safely
                            try:
                                x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
                                # clamp to image size
                                w, h = image_query.size
                                x1 = max(0, min(w-1, x1))
                                x2 = max(0, min(w, x2))
                                y1 = max(0, min(h-1, y1))
                                y2 = max(0, min(h, y2))
                                if x2 > x1 and y2 > y1:
                                    sample['is_small'] = cropped_entry.get('is_small')
                                    sample['bbox_2d'] = bbox
                                    sample['entity'] = cropped_entry.get('entity').split(' ')[1].replace(".","")
                                    
                                    cropped_img = image_query.crop((x1, y1, x2, y2))
                                    annotated_image, _ = highlight_entity(image_query, (x1, y1, x2, y2))
                            except Exception:
                                cropped_img = None
                                annotated_image = None
            except Exception:
                cropped_img = None
                annotated_image = None


        # Attach cropped image (or None) to sample under 'image_cropped'
        return cropped_img, annotated_image

def calculate_splits(dataset_len):
    # Robust parsing of PART/TOTAL_PART environment variables
    part = int(os.environ.get('PART', '0'))
    total_part_env = os.environ.get('TOTAL_PART', None)
    total_part = (int(total_part_env) + 1) if total_part_env is not None else 1
    print(f'Computing split {part} of the dataset... (total_part={total_part})')
    slicing = dataset_len // total_part
    if (part+1) == total_part:
        start_idx = slicing * part
        end_idx = dataset_len
    else:
        start_idx = slicing * part
        end_idx = slicing * part + slicing
    print(f'Processing Element from {start_idx} to {end_idx}...')

    return start_idx, end_idx

def load_csv_to_dict(csv_path):
    mapping = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            value = row[1].strip() if len(row) > 1 else ''
            mapping[key] = value
    return mapping

class DatasetInfoseekOMGM(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        print(f'Loading from {args.query_path_omgm}...')
        self.data, self.header = load_csv_data(args.query_path_omgm)
        print(f'Loading completed...')

        # Optional: load cropped_map same as Dataset (use args.cropped_json_path or default)
        self.cropped_map = {}
        try:
            cropped_path = None
            if hasattr(args, 'cropped_json_path') and args.cropped_json_path:
                cropped_path = args.cropped_json_path
            else:
                default_path = "/leonardo_scratch/large/userexternal/mmorini0/bbox_filtering_results/all_results_merged.json"
                if os.path.exists(default_path):
                    cropped_path = default_path

            if cropped_path and os.path.exists(cropped_path):
                print(f"Loading cropped map from {cropped_path}")
                with open(cropped_path, 'r') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.cropped_map = {str(k): v for k, v in loaded.items()}
                elif isinstance(loaded, list):
                    for entry in loaded:
                        key = entry.get('dataset_image_ids') or entry.get('image_id') or entry.get('image_path')
                        if key is not None:
                            self.cropped_map[str(key)] = entry
                print(f"Loaded cropped_map with {len(self.cropped_map)} entries")
        except Exception as e:
            print(f"Warning: failed loading cropped_json_path: {e}")
            self.cropped_map = {}

        self.image_path_mapping = load_csv_to_dict('/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/oven_eval/image_downloads/ovenid2impath.csv')
        self.img_base_path = "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/oven_eval/image_downloads/"
    
    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

    def __getitem__(self, idx):
        sample = get_test_question(idx, self.data, self.header)

        dataset_image_ids = sample["dataset_image_ids"]
            
        image_path = os.path.join(self.img_base_path, self.image_path_mapping.get(dataset_image_ids))
        image = Image.open(image_path)

        sample['image_query'] = image
        sample['image_path'] = image_path
    
        return sample
    
    def __len__(self):
        return len(self.data)

def get_query_dataset(args):
    global CURRENT_ID
    if args.dataset_name == 'evqa':
        dataset = Dataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'viquae':
        dataset = ViquaeDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'mrag':
        dataset = MRAGBenchDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'okvqa':
        dataset = OKVQADataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'oven':
        dataset = OVENDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'blink':
        dataset = BLINKDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'mmvp':
        dataset = MMVPDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'real_world_qa':
        dataset = RealWorldQADataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'qbench':
        dataset = QBenchDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'vstar':
        dataset = VStarDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'ade':
        dataset = AdeDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'omni':
        dataset = OmniDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'coco':
        dataset = COCODataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'textvqa':
        dataset = TextVQADataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'chartqa':
        dataset = ChartQADataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'ocrbench':
        dataset = OCRBenchDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'pope':
        dataset = POPEDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'chair':
        dataset = ChairDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'amber':
        dataset = AmberDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'amber_disc':
        dataset = AmberDiscDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'scienceqa':
        dataset = ScienceQADataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))    
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'gqa':
        dataset = GQADataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'mathvista':
        dataset = MathVistaDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'ai2d':
        dataset = AI2DDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'mme':
        dataset = MMEDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'mmebench_en':
        dataset = MMEBenchENDataset(args)
        start_idx, end_idx = calculate_splits(len(dataset))
        dataset.split(start_idx, end_idx)
    elif args.dataset_name == 'infoseek':
        if args.use_omgm:
            dataset = DatasetInfoseekOMGM(args)
            start_idx, end_idx = calculate_splits(len(dataset))
            dataset.split(start_idx, end_idx)
        else:
            shard_list = braceexpand(args.query_path)
            start_idx, end_idx = calculate_splits(73620)

            def process_sample(sample):
                global CURRENT_ID
                
                for key in sample:
                    if isinstance(sample[key], bytes):
                        sample[key] = sample[key].decode('utf-8')
                        
                sample['image_query'] = sample.pop('img.jpg')
                sample['data_id'] = sample['__key__']

                return sample
                
            
            def should_process(sample):
                global CURRENT_ID
                should_keep = start_idx <= CURRENT_ID < end_idx
                CURRENT_ID += 1
                return should_keep

            dataset = wds.DataPipeline(
                wds.SimpleShardList(shard_list),
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                # wds.split_by_node,
                # wds.split_by_worker,
                wds.select(should_process),
                wds.decode("pil", handler=wds.warn_and_continue),
                wds.map(process_sample, handler=wds.warn_and_continue),
                #wds.select(lambda x: x is not None, handler=wds.warn_and_continue)
            )
    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")
        
    return dataset, start_idx, end_idx



class MRAGBenchDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dataset = load_from_disk("/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/mrag_bench")

    def __getitem__(self, index):
        
        sample = self.dataset[index]

        sample['data_id'] = sample['id']
        sample['image_query'] = sample['image'].convert("RGB")
        sample['image_path'] = None  # No path available in this dataset format
        sample['gt_choice'] = sample['answer_choice']

        question = sample['question']
        choices_A = sample['A']
        choices_B = sample['B']
        choices_C = sample['C']
        choices_D = sample['D']

        prompt = f"Answer with the option's letter from the given choices directly.\n"

        question += f"\n Choices:\nA: {choices_A}\nB: {choices_B}\nC: {choices_C}\nD: {choices_D}"
        question = prompt + question
        sample['question'] = question


        return sample

    def __len__(self):
        return len(self.dataset)
    
    def split(self, start_idx, end_idx):
        self.dataset = self.dataset.select(range(start_idx, end_idx))
    
        
class OKVQADataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.data = []
        with open('/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/M2KR/okvqa/okvqa_test.jsonl', 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

    def __getitem__(self, index):
        sample = self.data[index]
        image_path = sample['image_path']
        sample['image_query'] = Image.open(image_path).convert("RGB")

        sample['wikipedia_url'] = ''
        sample['data_id'] = str(sample['data_id'])

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]


class OVENDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.data = []
        with open('/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/oven_eval/annotations/oven_query_val.jsonl', 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

        self.image_path_mapping = load_csv_to_dict('/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/oven_eval/image_downloads/ovenid2impath.csv')
        self.img_base_path = "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/oven_eval/image_downloads/"

    def __getitem__(self, index):
        sample = self.data[index]
        image_path = os.path.join(self.img_base_path, self.image_path_mapping.get(sample['image_id'], ''))
        sample['image_path'] = image_path
        try:
            sample['image_query'] = Image.open(image_path).convert("RGB")
        except:
            # Black image
            sample['image_query'] = Image.new("RGB", (224, 224), color="black")

        sample['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{sample['entity_text'].replace(' ', '_')}"
        sample['data_id'] = str(sample['data_id'])
        sample['answer'] = sample['entity_text']

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

class BLINKDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        all_sets = []

        categories = ["Counting", "IQ_Test", "Object_Localization", "Relative_Depth", "Relative_Reflectance", "Spatial_Relation"]

        for cat in categories:
            print(f"Caricamento categoria: {cat}")
            # Carichiamo il subset specifico
            ds = load_from_disk(f"/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/blink/{cat}")
            all_sets.append(ds)

        # Uniamo tutto in un unico oggetto Dataset
        self.data = concatenate_datasets(all_sets)

    def __getitem__(self, index):
        sample = self.data[index]
        sample['data_id'] = str(sample['idx'])
        if sample['image_1'] is not None:
            sample['image_query'] = sample['image_1'].convert('RGB')
        else:
            sample['image_query'] = Image.new('RGB', (224, 224), color=(0, 0, 0))
            print(f"[!] Warning: image for data_id {sample['data_id']} is None. Using black image as placeholder.")
        sample['wikipedia_url'] = ''
        sample['question'] = sample['prompt'] + "\nPlease answer directly with only the letter of the correct option and nothing else."
        sample['category'] = sample['idx'].split('_')[1]
        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class MMVPDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        images = {}
        images_path = {}
        for i in range(300):
            file_path = hf_hub_download(repo_id="MMVP/MMVP", filename=f"{i+1}.jpg", subfolder="MMVP Images", repo_type="dataset")
            images[i] = Image.open(file_path).convert('RGB')
            images_path[i] = file_path

        self.data = []
        file_path = hf_hub_download(repo_id="MMVP/MMVP", filename="Questions.csv", repo_type="dataset")
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if row[0]=="lndex":
                    continue
                if row[0]=="Index":
                    continue
                self.data.append({
                    "question": str(row[1]),
                    "imageId": int(row[0])-1,
                    "options": str(row[2]).replace('(a)', 'A.').replace('(b)', 'B.'),
                    # "text_options": self.give_options(str(row[2])),
                    "answer": str(row[3]).replace('(a)', 'A').replace('(b)', 'B'),
                    "image_query": images[int(row[0])-1],
                    "image_path": images_path[int(row[0])-1]
                })

    @staticmethod
    def give_options(input_string):
        parts = input_string.split("(")
        result = [part.split(")")[1].strip() for part in parts[1:]]
        return result
    

    def __getitem__(self, idx):
        sample = self.data[idx]

        sample['data_id'] = f"MMVP_{idx}"
        sample['wikipedia_url'] = ''

        sample['question'] += '\n' + sample['options'] + "\nPlease answer directly with only the letter of the correct option and nothing else."

        return sample

    def __len__(self):
        return len(self.data)

    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

class RealWorldQADataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.data = load_from_disk("/leonardo_scratch/large/userexternal/mmorini0/real_world_qa")

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample['data_id'] = f"RealWorldQA_{idx}"
        sample['wikipedia_url'] = ''

        sample['image_query'] = sample['image']

        return sample
    
    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))


class QBenchDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.images = {}
        file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="images_llvisionqa.tar", repo_type="dataset")
        extract_dir = os.path.dirname(file_path)
        if not os.path.exists(extract_dir+"/images"):
            with tarfile.open(file_path, "r:") as tar:
                tar.extractall(path=extract_dir)

        files_in_dir = os.listdir(extract_dir+"/images/")
        for file in files_in_dir:
            self.images[file] = extract_dir+"/images/"+file

        self.data = []
        dev_file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="llvisionqa_dev.json", repo_type="dataset")

        with open(dev_file_path, "r") as json_file:
            self.data.extend(json.load(json_file))

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        image_path = self.images[sample['img_path']]
        sample['image_path'] = image_path
        if os.path.exists(image_path):
            sample['image_query'] = Image.open(image_path).convert('RGB')
        else:
            # Black Image
            sample['image_query'] = Image.new('RGB', (224, 224), color=(0, 0, 0))
            print(f"[!] Warning: image path {image_path} not found. Using black image as placeholder.")

        sample['data_id'] = f"QBench_{idx}"
        sample['wikipedia_url'] = ''

        candidates = sample['candidates'] # ['Dull', 'Normal', 'Colorful']
        options = ""
        for i, candidate in enumerate(candidates):
            options += f"{chr(65+i)}. {candidate}\n"

        sample['question'] += '\n' + options + "\nPlease answer directly with only the letter of the correct option and nothing else."

        correct_ans = sample['correct_ans'] # e.g., "Colorful"
        correct_letter = chr(65 + candidates.index(correct_ans)) # e.g., "C"
        sample['answer'] = correct_letter

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

class VStarDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.data = load_from_disk("/leonardo_scratch/large/userexternal/mmorini0/vstar_bench")

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['question'] = sample['text']
        file_path = os.path.join("/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/datasets--craigwu--vstar_bench/snapshots/d9ae62c903da0c98336e85c5ee89cd863b04b4da/", sample['image'])
        sample['image_path'] = file_path
        if os.path.exists(file_path):
            sample['image_query'] = Image.open(file_path).convert('RGB')
        else:
            # Black Image
            sample['image_query'] = Image.new('RGB', (224, 224), color=(0, 0, 0))
            print(f"[!] Warning: image path {file_path} not found. Using black image as placeholder.")
        
        sample['data_id'] = f"VStar_{sample['question_id']}"
        sample['wikipedia_url'] = ''

        sample['answer'] = sample['label']

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class AdeDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_from_disk("/leonardo_scratch/large/userexternal/mmorini0/ade_bench")

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = sample['image'].convert('RGB')
        
        sample['data_id'] = f"Ade_{sample['unique_id']}"
        sample['wikipedia_url'] = ''

        sample['category'] = sample['sub_task']

        sample['question'] = sample['prompt'].replace("Select", "\nSelect") + "\nAnswer with the option's letter from the given choices directly."

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class OmniDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.dataset = load_from_disk("/leonardo_scratch/large/userexternal/mmorini0/cv_bench")

        self.data = []
        for sample in self.dataset:
            if 'Omni3D' in sample['source']:
                self.data.append(sample)



    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = sample['image'].convert('RGB')
        
        sample['data_id'] = f"CVBench_{sample['idx']}"
        sample['wikipedia_url'] = ''

        sample['category'] = sample['task']

        sample['question'] = sample['prompt']

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx: end_idx]

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_from_disk("/leonardo_scratch/large/userexternal/mmorini0/coco_bench")



    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = sample['image'].convert('RGB')
        
        sample['data_id'] = f"COCO_{sample['unique_id']}"
        sample['wikipedia_url'] = ''

        sample['category'] = sample['sub_task']

        sample['question'] = sample['prompt']

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class TextVQADataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_dataset("lmms-lab/textvqa", cache_dir="/leonardo_scratch/large/userexternal/mmorini0", split="validation")

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = sample['image'].convert('RGB')
        
        sample['data_id'] = f"TextVQA_{sample['question_id']}"
        sample['wikipedia_url'] = ''

        sample['answer'] = sample['answers']

        qs = sample["question"]
        qs += "\nReference OCR tokens:"
        for i in range(len(sample["ocr_tokens"])):
            token = sample["ocr_tokens"][i]
            if i==0:
                qs += f" {token}"
            else:
                qs += f", {token}"
        qs += f"\nAnswer the question using a single word or very few words. Answer: "

        sample['question'] = qs

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class ChartQADataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_dataset("lmms-lab/ChartQA", cache_dir="/leonardo_scratch/large/userexternal/mmorini0", split='test')



    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = sample['image'].convert('RGB')

        sample['data_id'] = f"ChartQA_{idx}"

        sample['question'] = sample['question'] + "\nAnswer with a single word or very few words. Answer: "

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class OCRBenchDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_dataset("echo840/OCRBench", split="test", cache_dir="/leonardo_scratch/large/userexternal/mmorini0")

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = sample['image'].convert('RGB')
        
        sample['data_id'] = f"OCRBench_{idx}"
        sample['wikipedia_url'] = ''

        sample['category'] = sample['question_type']

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class POPEDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_dataset("lmms-lab/POPE", split="test", cache_dir="/leonardo_scratch/large/userexternal/mmorini0")

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = sample['image'].convert('RGB')
        
        sample['data_id'] = f"POPE_{sample['question_id']}"
        sample['wikipedia_url'] = ''

        sample['category'] = sample['category']

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class ChairDataset(torch.utils.data.Dataset):

    def __init__(self, args):

        self.args = args

        self.data = [json.loads(q) for q in open(os.path.expanduser('/leonardo_scratch/large/userexternal/mmorini0/CHAIR-eval/data/chair-500.jsonl'), "r")]

        self.image_base_path = "/leonardo_scratch/large/userexternal/mmorini0/CHAIR-eval/data/chair-500"

    def __getitem__(self, idx):
        sample = self.data[idx]

        image_path = os.path.join(self.image_base_path, sample['image'])
        sample['image_path'] = image_path
        sample['image_query'] = Image.open(sample['image_path']).convert('RGB')

        sample['data_id'] = f"CHAIR_{sample['question_id']}"
        sample['wikipedia_url'] = ''

        sample['image_id'] = sample['image']

        sample['question'] = sample['text']

        return sample
    
    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

    def __len__(self):
        return len(self.data)

class AmberDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        with open('/leonardo_scratch/large/userexternal/mmorini0/AMBER/data/query/query_generative.json', 'r') as f:
            self.data = json.load(f)

        self.image_base_path = '/leonardo_scratch/large/userexternal/mmorini0/AMBER/image'
        
    def __getitem__(self, idx):
        sample = self.data[idx]

        image_path = os.path.join(self.image_base_path, sample['image'])
        sample['image_path'] = image_path
        sample['image_query'] = Image.open(sample['image_path']).convert('RGB')

        sample['data_id'] = f"AMBER_{sample['id']}"

        sample['wikipedia_url'] = ''
        sample['question'] = sample['query']

        return sample
        

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

class AmberDiscDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        with open('/leonardo_scratch/large/userexternal/mmorini0/AMBER/data/query/query_discriminative.json', 'r') as f:
            self.data = json.load(f)

        self.image_base_path = '/leonardo_scratch/large/userexternal/mmorini0/AMBER/image'
        
    def __getitem__(self, idx):
        sample = self.data[idx]

        image_path = os.path.join(self.image_base_path, sample['image'])
        sample['image_path'] = image_path
        sample['image_query'] = Image.open(sample['image_path']).convert('RGB')

        sample['data_id'] = f"AMBER_{sample['id']}"

        sample['wikipedia_url'] = ''
        sample['question'] = sample['query']

        return sample
        

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

class ScienceQADataset(torch.utils.data.Dataset): #TODO: da finire
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.dataset = load_dataset("derek-thomas/ScienceQA", split="test", cache_dir="/leonardo_scratch/large/userexternal/mmorini0")

        self.data = []

        for sample in self.dataset:
            if sample['image'] is not None:
                self.data.append(sample)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = sample['image'].convert('RGB')
        
        sample['data_id'] = f"ScienceQA_{idx}"
        sample['wikipedia_url'] = ''

        sample['category'] = sample['grade']
        sample['text_answer'] = sample['choices'][sample['answer']]

        sample['answer'] = chr(ord('A')+sample['answer'])

        qs = sample['question']

        for i in range(len(sample["choices"])):
            option = sample["choices"][i]
            qs += f"\n{chr(ord('A')+i)}. {option}"

        qs += f"\nPlease answer directly with only the letter of the correct option and nothing else."

        sample['question'] = qs

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data[start_idx:end_idx]

class GQADataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.img_dataset = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", cache_dir="/leonardo_scratch/large/userexternal/mmorini0")
        self.images = {}
        for row in self.img_dataset:
            self.images[row['id']] =  row['image']

        self.data = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev", cache_dir="/leonardo_scratch/large/userexternal/mmorini0")

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = self.images[sample['imageId']]
        
        sample['data_id'] = f"GQA_{sample['id']}"
        sample['wikipedia_url'] = ''

        sample['question'] = sample['question'] + "\nAnswer the question using a single word or phrase."

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class MathVistaDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_dataset("AI4Math/MathVista", split="testmini", cache_dir="/leonardo_scratch/large/userexternal/mmorini0")

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['data_id'] = f"MathVista_{sample['pid']}"
        qs = sample['query']

        if sample["question_type"] == "multi_choice":
            qs += f"\nGive the answer in the format of A, B, C or D.\nAnswer: "
        else:
            qs += f"\nAnswer the question using a single word or phrase."
        
        sample['question'] = qs

        sample['image_query'] = sample['decoded_image']

        sample['category'] = sample['metadata']['category']
        sample['type'] = sample['question_type']

        gt_answer = sample["answer"]
        if sample["question_type"] == "multi_choice":
            reverse_dict = {}
            for ind, item in enumerate(sample["choices"]):
                reverse_dict[item] = chr(ord('A')+ind)
            gt_answer = reverse_dict[gt_answer]
        
        sample['answer'] = gt_answer
        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class AI2DDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_dataset("lmms-lab/ai2d", split="test", cache_dir="/leonardo_scratch/large/userexternal/mmorini0")


    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['image_query'] = sample['image'].convert('RGB')
        
        sample['data_id'] = f"AI2D_{idx}"
        sample['wikipedia_url'] = ''

        qs = sample["question"]
        keys = ["A", "B", "C", "D"]
        for i in range(len(sample["options"])):
            option = sample["options"][i]
            key = keys[i]
            qs += f"\n{key}. {option}"

        qs += "Answer with the option's letter from the given choices directly."
        
        sample['question'] = qs

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

class MMEDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_dataset("lmms-lab/MME", split="test", cache_dir="/leonardo_scratch/large/userexternal/mmorini0")

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample['question'] = sample['question'] + "\nAnswer the question using only yes or no."

        sample['image_query'] = sample['image'].convert('RGB')
        sample['data_id'] = f"MME_{sample['question_id'].replace('/', '_')}"

        return sample
    
    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))
    
class MMEBenchENDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.data = load_dataset("lmms-lab/MMBench_EN", split="dev", cache_dir='/leonardo_scratch/large/userexternal/mmorini0')

    @staticmethod
    def hash_image(image):
        image = image.resize((256, 256))
        image_bytes = image.tobytes()
        hash_value = mmh3.hash(image_bytes)
        hex_hash = format(hash_value, '08x')
        return hex_hash

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if sample["hint"] != "nan":
            qs = sample["hint"] + "\n" + sample["question"] + f" Options:"
        else:
            qs = sample["question"] + f" Options:"
        for keys in ["A", "B", "C", "D"]:
            if sample[keys] != "nan":
                qs += (f"\n{keys}. "+sample[keys])
        
        qs += "\nAnswer with the option's letter from the given choices directly."
        
        sample['question'] = qs

        sample['image_query'] = sample['image'].convert('RGB')
        sample['data_id'] = f"MMEBenchEN_{sample['index']}"

        image_hash = self.hash_image(sample['image_query'])

        sample['image_query'] = sample['image']

        source_id = (sample["question"]+ " " + image_hash + " " + sample["source"])

        sample['source_id'] = source_id

        return sample

    def __len__(self):
        return len(self.data)
    
    def split(self, start_idx, end_idx):
        self.data = self.data.select(range(start_idx, end_idx))

if __name__ == "__main__":
    # dataset_names = ['blink', 'mmvp', 'real_world_qa', 'qbench', 'vstar', 'ade', 'omni', 'coco', 'textvqa', 'chartqa', 'ocrbench', 'pope', 'chair', 'amber', 'amber_disc', 'scienceqa', 'gqa', 'mathvista', 'ai2d', 'mme', 'mmebench_en']
    # dataset_names = ["viquae", "mrag", "okvqa", "oven"]
    dataset_names = ['textvqa']
    from argparse import Namespace
    args = Namespace()
    dataset_lens = {}
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        dataset, _, _ = get_query_dataset(args)
        dataset_lens[dataset_name] = len(dataset)
        print(f"Dataset {dataset_name} loaded with {len(dataset)} samples.")
        del dataset

    print(dataset_lens)