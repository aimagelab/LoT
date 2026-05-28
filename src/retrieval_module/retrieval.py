import time
from scipy.ndimage import gaussian_filter
import copy
import types
import gc
import math
from matplotlib import pyplot as plt
import numpy as np
import time
import PIL
import spacy
import webdataset as wds
from braceexpand import braceexpand
import re
import ujson
import tqdm
import os
import subprocess
import pandas as pd
from PIL import Image
import torch
import cv2
from scipy.ndimage import binary_erosion, binary_dilation
from typing import Dict, Tuple, Optional
import random


_original_torch_load = torch.load
def patched_load(args, **kwargs):
    kwargs['mmap'] = False
    return _original_torch_load(args, **kwargs)
torch.load = patched_load
import argparse

from transformers import InternVLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration

from accelerate import infer_auto_device_map, init_empty_weights
from spacy.lang.en import English

import retrieval_module.prompts as prompts
from retrieval_module.retriever import Retriever, uniform_passages_of_sentences
from retrieval_module.data import *
from retrieval_module.entity_extraction import extract_question_target
from critique.critique_model import CritiqueModel
from cropper_module.cropper import Cropper, highlight_entity
from cropper_module.extract_entity import find_visual_entity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from retrieval_module.new_forward import *
from accelerate.hooks import remove_hook_from_module
#clip_path= "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/BAAI/EVA-CLIP-8B/models--BAAI--EVA-CLIP-8B/snapshots/0e4dca944e8ece27eb9dfe4a488c0ed0c4644fc9"
#clip_processor= "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/openai/clip-vit-large-patch14/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

PASSAGE_DELIMITER = "\n\n\n"

# --- SINK CONFIGURATION for Attention Bbox Extraction ---
SINK_DIMS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": [458, 2570],
    "Qwen/Qwen2.5-VL-3B-Instruct": [318, 1874, 1819],
    "Qwen/Qwen2.5-VL-32B-Instruct": [4675, 3094],
    "OpenGVLab/InternVL3_5-1B-hf": [35, 13],
    "OpenGVLab/InternVL3_5-4B-hf": [4, 396, 0],
    "OpenGVLab/InternVL3_5-8B-hf": [2276, 233],
    "Qwen/Qwen3-VL-2B-Instruct": [1793, 1999, 1401],
    "Qwen/Qwen3-VL-4B-Instruct": [0],
    "Qwen/Qwen3-VL-8B-Instruct": [1838],
    "aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning": [788, 1384, 4062],
    "Qwen/Qwen2-VL-2B-Instruct": [1073, 534, 940],
    "Qwen/Qwen2-VL-7B-Instruct": [2570, 458],
    "llava-hf/llava-1.5-7b-hf": [2533, 1415],
    "OpenGVLab/InternVL3_5-38B-hf": [731]
}
SINK_PERCENTILE = 25  # Fixed percentile for filtering visual sink tokens

nlp = spacy.load("en_core_web_sm")

# --- Bbox Extraction Helper Functions ---
def extract_bbox_weighted_centroid(attention_map: np.ndarray, std_multiplier: float = 2.0) -> Tuple[int, int, int, int]:
    """
    Weighted Centroid method for bbox extraction.
    
    Computes the centroid weighted by attention values and uses standard deviation
    to determine the bounding box size.
    """
    attn_norm = attention_map / (attention_map.sum() + 1e-8)
    
    h, w = attention_map.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    cx = np.sum(x_coords * attn_norm)
    cy = np.sum(y_coords * attn_norm)
    
    std_x = np.sqrt(np.sum(((x_coords - cx) ** 2) * attn_norm))
    std_y = np.sqrt(np.sum(((y_coords - cy) ** 2) * attn_norm))
    
    x1 = max(0, int(cx - std_multiplier * std_x))
    y1 = max(0, int(cy - std_multiplier * std_y))
    x2 = min(w, int(cx + std_multiplier * std_x))
    y2 = min(h, int(cy + std_multiplier * std_y))
    
    return x1, y1, x2, y2


def extract_bbox_morphological(attention_map: np.ndarray, threshold: float = 0.3, kernel_size: int = 7) -> Tuple[int, int, int, int]:
    """
    Morphological Operations method for bbox extraction.
    
    Binarizes the attention map and applies morphological closing to get a connected region.
    """
    binary = attention_map > threshold
    
    kernel = np.ones((kernel_size, kernel_size))
    closed = binary_dilation(binary, kernel)
    closed = binary_erosion(closed, kernel)
    
    coords = np.argwhere(closed)
    
    if len(coords) == 0:
        return 0, 0, attention_map.shape[1], attention_map.shape[0]
    
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    
    return int(x1), int(y1), int(x2), int(y2)


def compute_average_bbox(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Compute the average of two bounding boxes."""
    x1 = int((bbox1[0] + bbox2[0]) / 2)
    y1 = int((bbox1[1] + bbox2[1]) / 2)
    x2 = int((bbox1[2] + bbox2[2]) / 2)
    y2 = int((bbox1[3] + bbox2[3]) / 2)
    return x1, y1, x2, y2


def visualiza_attn_map(image, attn_map, output_file):
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        ax[1].imshow(attn_map, cmap='hot')
        ax[1].set_title("Stitched Attention Map")
        ax[1].axis('off')
        ax[2].imshow(image)
        ax[2].imshow(gaussian_filter(attn_map, sigma=1.5), cmap='hot', alpha=0.6)
        ax[2].set_title("Overlay")
        ax[2].axis('off')
        plt.savefig(output_file)
        plt.close()

def get_args():
    parser = argparse.ArgumentParser(description="RAG inference")

#    parser.add_argument("--text_index_path", type=str, required=False, help="Path to the FAISS text index file")
#    parser.add_argument("--text_index_json_path", type=str, required=False, help="Path to the JSON file with document metadata")
    parser.add_argument("--img_index_path", type=str, required=False)
    parser.add_argument("--img_index_json_path", type=str, required=False)
    parser.add_argument("--wiki_KB", type=str, required=False)
    parser.add_argument("--KB_images", type=str, required=False)
    parser.add_argument("--query_path", type=str, required=False)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset_name", type=str, choices=["evqa", "infoseek", "viquae", 'mrag', 'okvqa', 'oven', 'blink', 'mmvp', 'real_world_qa', 'qbench', 'vstar', 'ade', 'omni', 'coco', 'textvqa', 'chartqa', 'ocrbench', 'pope', 'chair', 'amber', 'amber_disc', 'scienceqa', 'gqa', 'mathvista', 'ai2d', 'mme', 'mmebench_en'], required=False, default='None')
    parser.add_argument("--dataset_split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--output_root", type=str, default="./")

    parser.add_argument("--use_omgm", action='store_true')
    parser.add_argument("--reconstruct_omgm", action='store_true')
    parser.add_argument("--query_path_omgm", type=str, default=None)
    parser.add_argument('--omgm_retriever', action='store_true')
    parser.add_argument('--omgm_retriever_crop', action='store_true')
    parser.add_argument('--omgm_step1', action='store_true')
    parser.add_argument('--omgm_step1_result_file', type=str, default=None)
    parser.add_argument('--omgm_step2', action='store_true')
    parser.add_argument('--omgm_step2_result_file', type=str, default=None)

    parser.add_argument("--experiment_type", type=str, choices=["with_retrieval", "no_retrieval"], default="with_retrieval")
    parser.add_argument("--pre_computed_retrieval_path", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=1, help="Number of top documents to retrieve")

    parser.add_argument('--use_google_lens', action='store_true')
    parser.add_argument('--use_oracle', action='store_true')

    parser.add_argument('--only_question', action='store_true') # Per zero-shot

    parser.add_argument("--all_filtered", action='store_true') # If true, when no filtered retrieval, use only question without any prompt

    # Passages Evaluation args
    parser.add_argument("--critique_model_name", type=str, default="/leonardo_scratch/large/userexternal/fcocchi0/reag/reag_reflectiva/sft_bs-2_acc-8_lr-2e-06_wr-0.03_user-m/checkpoint-7000")
    parser.add_argument('--eval_passages', action='store_true')
    parser.add_argument('--eval_passages_w_images', action='store_true')
    parser.add_argument('--eval_passages_batch_size', type=int, default=5, help="Batch size for passages evaluation")
    parser.add_argument("--yes_prob_thr", type=float, default=0.5, help="Threshold for yes probability in passage relevance evaluation")

    # Reasoning
    parser.add_argument("--extract_reasoning", action='store_true')
    parser.add_argument("--force_reasoning", action='store_true')
    parser.add_argument("--strict_parsing", action='store_true')

    parser.add_argument("--short_prompt", action='store_true')
    parser.add_argument("--multiple_choice", action='store_true')
    parser.add_argument("--one_word", action='store_true')
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument('--model_max_length', type=int, default=16384)
    parser.add_argument('--min_pixels', type=int, default=3136)
    parser.add_argument('--max_pixels', type=int, default=301056)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--repetition_penalty', type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k_sampling', type=int, default=None)
    parser.add_argument('--min_p', type=float, default=None)

    parser.add_argument('--cropper_model_name', type=str, default="IDEA-Research/grounding-dino-base",)
    parser.add_argument('--crop_query_img', action='store_true')
    parser.add_argument('--cropped_json_path', type=str, default=None)
    parser.add_argument('--max_bbox_ratio', type=float, default=None)

    # Self-Elicit
    parser.add_argument('--self_elicit_alpha', type=float, default=0.5)
    parser.add_argument('--self_elicit_image_markers', action='store_true')
    parser.add_argument('--self_elicit_image_markers_w_fallback', action='store_true')
    parser.add_argument('--self_elicit_image_bbox', action='store_true')
    parser.add_argument('--self_elicit_image_crop', action='store_true')
    parser.add_argument('--self_elicit_image_only_crop', action='store_true')
    parser.add_argument('--self_elicit_image_add_crop', action='store_true')
    parser.add_argument('--self_elicit_image_add_original', action='store_true')
    parser.add_argument('--self_elicit_image_w_bbox', action='store_true')
    parser.add_argument('--self_elicit_image_add_bbox', action='store_true')
    parser.add_argument('--self_elicit_image_add_crop_markers', action='store_true')

    parser.add_argument("--use_attention_bbox", action='store_true', help="Whether to use attention-based bbox for cropping. If true, bbox will be computed dynamically using model attention maps.")
    # parser.add_argument("--attention_bboxes", type=str, help="Path to load attention-based bboxes. ", default="/leonardo_scratch/large/userexternal/mmorini0/OMGM/attention_bboxes.json")
    parser.add_argument("--attention_bbox_method", type=str, default="weighted_centroid", help="Method used to extract the bbox (e.g., 'weighted_centroid', 'morphological_t0.3_k7', 'average')")
    parser.add_argument("--attention_bbox_layer_range", type=str, default="middle_half", help="Layer range for attention bbox extraction") # , choices=['all', 'last_half', 'first_half', 'middle_half']
    # parser.add_argument('--same_grounding_dino_setting', action='store_true', help="If using attention bbox, whether to use the same setting as grounding dino for bbox extraction. (e.g. same fallback on original image)")

    parser.add_argument("--self_elicit", action='store_true')
    parser.add_argument("--self_elicit_gen", action='store_true')
    parser.add_argument("--self_elicit_gen_all", action='store_true')
    parser.add_argument("--self_elicit_gen_passage", action='store_true')
    parser.add_argument("--self_elicit_gen_sen2pas", action='store_true')

    #parser.add_argument("--self_elicit_max_passages", type=int, default=3, help="Max passages for self-elicit to avoid OOM with attentions")
    # parser.add_argument('--self_elicit_config', type=str, default="/leonardo/home/userexternal/mmorini0/SelfElicit/src/self_elicit/config_critic.yaml", required=False)
    
    parser.add_argument("--small_bbox", action='store_true')

    parser.add_argument("--few_shot_examples", action='store_true')

    parser.add_argument("--crop_in_retrieval", action='store_true')

    parser.add_argument("--re_rank_qwen", action='store_true')
    parser.add_argument("--re_rank_top_k", type=int, default=5, help="Number of top documents to retrieve")

    parser.add_argument("--attention_text_layer_range", type=str, default=None, help="Layer range for attention bbox extraction") # , choices=['all', 'last_half', 'first_half', 'middle_half']

    parser.add_argument('--sink_tau', type=float, default=None, help="If set, filter visual sink tokens based on this attention threshold (percentile) instead of fixed top-k.")
    parser.add_argument('--weighted_centroid_std_multiplier', type=float, default=2.0, help="Standard deviation multiplier for weighted centroid bbox extraction.")



    # REBUTTAL ARGS
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--gdino_bbox', action='store_true')
    parser.add_argument('--comp', action='store_true')

    args = parser.parse_args()

    args.text_elicit = args.self_elicit or args.self_elicit_gen or args.self_elicit_gen_passage or args.self_elicit_gen_sen2pas

        # Print args
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n\n")
    
    return args

def log_memory(tag):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[{tag}] Alloc: {allocated:.2f}GB | Res: {reserved:.2f}GB | Peak: {max_alloc:.2f}GB")


class InferenceModel:
    def __init__(self, args):
        self.args = args

        self.token_vis_start = None
        self.token_vis_end = None


    @staticmethod
    def get_n_match(string, substring):
        """
        Count the number of occurrences of a substring within a string.

        Parameters
        ----------
        string : str
            The string to search within.

        substring : str
            The substring to count occurrences of.

        Returns
        -------
        count : int
            Number of occurrences of the substring.
        """
        all_starts = []
        start = 0
        while True:
            start = string.find(substring, start)
            if start == -1:
                break
            all_starts.append(start)
            start += 1  # Increment start to avoid overlapping matches
        return len(all_starts)

    def get_inputs(self, question, image_query, image_cropped=None, annotated_image=None, bbox_2d=None, entity=None, text_eliciting=False):
        def find_question(content, question):
            #Return position of the question in the content
            for i, c in enumerate(content):
                if c.get('type') == 'text' and question in c.get('text'):
                    return i
                
            return None

        messages = []
        
        SYSTEM_PROMPT = prompts.SYSTEM_PROMPT
        
        if not text_eliciting:
            if self.args.self_elicit_image_markers and not (self.args.self_elicit_image_only_crop and image_cropped is None) or self.args.self_elicit_image_markers_w_fallback or self.args.self_elicit_image_add_crop_markers and image_cropped is not None:
                SYSTEM_PROMPT += prompts.SELF_ELICIT_SYSTEM_PROMPT_VQA_IMG

            if self.args.self_elicit or self.args.self_elicit_gen or self.args.self_elicit_gen_passage or self.args.self_elicit_gen_sen2pas or self.args.self_elicit_gen_all:
                SYSTEM_PROMPT += prompts.SELF_ELICIT_SYSTEM_PROMPT_VQA_TEXT

            # if self.args.self_elicit_image_bbox and entity is not None:
            #     SYSTEM_PROMPT += prompts.SELF_ELICIT_IMAGE_BBOX_SYSTEM_PROMPT_VQA.format(entity=entity, color="red")

        if self.args.few_shot_examples:
            examples = "Here are some examples of the format you should follow in the answer:\n"
            if self.args.dataset_name == 'infoseek':
                few_shots = prompts.INFOSEEK_FEW_SHOTS_ELICITED if text_eliciting else prompts.INFOSEEK_FEW_SHOTS
            elif self.args.dataset_name == 'oven' and not text_eliciting:
                few_shots = prompts.OVEN_FEW_SHOTS
            else:
                few_shots = []
            for ex in few_shots:
                context = "\n".join(f"- {para}" for para in ex['context'])
                examples += f"""\
Question: {ex['question']}
Context:
{context}
Answer: {ex['answer']}
"""
            SYSTEM_PROMPT += examples

        if SYSTEM_PROMPT:
            messages.append(
                {
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': SYSTEM_PROMPT}
                    ]
                }
            )
        
        
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type":"text", "text": question}
                ]
            }
        )

        images = [image_query]

        if not text_eliciting:
            if self.args.self_elicit_image_markers and not (self.args.self_elicit_image_only_crop and image_cropped is None) or self.args.self_elicit_image_markers_w_fallback:
                messages[-1]['content'].insert(0, {"type": "text", "text":"<START_IMPORTANT_IMG>"})
                messages[-1]['content'].insert(2, {"type": "text", "text":"<END_IMPORTANT_IMG>\n"})

            # if self.args.self_elicit_image_crop and image_cropped is not None and bbox_2d is not None and entity is not None:
            #     messages[-1]["content"].append({"type": "text", "text": PASSAGE_DELIMITER + prompts.PROMPT_SELF_ELICT_IMAGE_CROP.format(bbox_2d=bbox_2d, entity=entity) + '\n'})
            #     messages[-1]["content"].append({"type": "image"})

            if (self.args.self_elicit_image_add_crop or self.args.self_elicit_image_add_bbox or self.args.self_elicit_image_add_crop_markers) and image_cropped is not None:
                
                pos = find_question(messages[-1]['content'], question)
            
                if self.args.self_elicit_image_add_crop_markers and image_cropped is not None:
                    messages[-1]['content'].insert(pos, {"type": "text", "text":"<END_IMPORTANT_IMG>\n"})
                
                messages[-1]['content'].insert(pos, {"type": "image"})
                
                if self.args.self_elicit_image_add_crop_markers and image_cropped is not None:
                    messages[-1]['content'].insert(pos, {"type": "text", "text":"<START_IMPORTANT_IMG>"})
            
            if self.args.self_elicit_image_add_original:
                pos = find_question(messages[-1]['content'], question)
                messages[-1]['content'].insert(pos, {"type": "image"})


            if self.args.short_prompt:
                messages[-1]["content"].append({"type": "text", "text": '\nGive a short answer.\nShort answer: '})
            if self.args.multiple_choice:
                messages[-1]["content"].append({"type": "text", "text": '\nGive the answer in the format of A, B, C or D.\nAnswer: '})
            if self.args.one_word:
                messages[-1]["content"].append({"type": "text", "text": '\nGive the answer in few words.\nAnswer: '})

            if self.args.self_elicit_image_only_crop and image_cropped is not None:
                images = [image_cropped]
            elif self.args.self_elicit_image_w_bbox and annotated_image is not None:
                images = [annotated_image]
            else: # Fall back to original behavior if there is no crop
                images = [image_query]
            
            # if self.args.self_elicit_image_crop and image_cropped is not None:
            #     images.append(image_cropped)

            if self.args.self_elicit_image_add_crop and image_cropped is not None:
                images.append(image_cropped)
            
            if self.args.self_elicit_image_add_crop_markers and image_cropped is not None:
                images.append(image_cropped)
            
            if self.args.self_elicit_image_add_bbox and annotated_image is not None:
                images.append(annotated_image)

            if self.args.self_elicit_image_add_original:
                images.append(image_query)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text = [text],
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
            max_length=self.args.model_max_length,
            truncation=True,
        )
        
        return inputs

    def find_text_token_spans(self, input_ids, target_text, raise_if_not_found=False):
        """
        Locate spans of a target text within tokenized input.

        Parameters
        ----------
        input_ids : list of int
            Tokenized input as a 1D list of token IDs.

        target_text : str
            Target text to find within the tokenized input.

        raise_if_not_found : bool, optional
            If True, raise an error if the target text is not found.

        Returns
        -------
        spans : list of tuple
            List of (start, end) indices for each occurrence of the target text.
            If the target is truncated, returns (start, -1) to indicate incomplete match.
        """
        # Ensure input_ids is a list of integers
        assert (type(input_ids) == list) and (
            type(input_ids[0]) == int
        ), "input_ids should be a 1-d list, make sure it's not a tensor."

        # Decode input tokens to text and encode the target text into tokens
        tokenizer = self.processor.tokenizer
        source = tokenizer.decode(input_ids)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        target = tokenizer.decode(target_ids)
        
        # Check if target is in source
        if target not in source:
            # Try to find at least the beginning of the target (first 20 chars)
            target_prefix = target_text[:20].strip()
            if target_prefix in source:
                # Find the start position of the prefix
                start = 0
                n_match_prefix = self.get_n_match(source, target_prefix)
                while start < len(input_ids) and n_match_prefix > 0:
                    start += 1
                    source_seg = tokenizer.decode(input_ids[start:])
                    n_match_cur = self.get_n_match(source_seg, target_prefix)
                    
                    if n_match_cur < n_match_prefix:
                        start -= 1
                        return [(start, -1)]  # Found start, but truncated
            
            if raise_if_not_found:
                assert False, f"'{target}' not found in input"
            return []
        
        # Initialize variables for finding spans
        n_match_left = self.get_n_match(source, target)
        spans = []
        start = 0

        while True:
            start += 1
            if start >= len(input_ids):
                break
                
            source_seg = tokenizer.decode(input_ids[start:])
            n_match_cur = self.get_n_match(source_seg, target)

            # If the number of matches decreases, start of a match is found
            if n_match_cur < n_match_left:
                # assert (
                #     n_match_left - n_match_cur == 1
                # ), f"{n_match_left - n_match_cur} matches in a same token"
                n_match_left = n_match_cur
                start -= 1
                # Find the end of the match
                end = max(start + len(target_ids) - 5, start)
                while end < len(input_ids):
                    end += 1
                    seg_text = tokenizer.decode(input_ids[start:end])
                    if target in seg_text:
                        break
                    # Check if we've gone too far without finding complete match
                    if end - start > len(target_ids) + 10:
                        # Likely truncated at the end
                        end = -1
                        break
                # Save the span and update the start position
                spans.append((start, end))
                if end == -1:
                    break
                start = end

            # Exit condition
            if n_match_left == 0:
                break

        return spans

    def get_context_token_span(
        self,
        context,
        input_ids
    ):
        """
        Identify the token span of the context within the tokenized input.

        Parameters
        ----------
        context : str
            The context passage for answering the question.

        question : str
            The question to answer.

        Returns
        -------
        context_span : tuple of int
            A tuple (start, end) representing the token span of the context.
        """
        context_spans = self.find_text_token_spans(input_ids, context)
        if len(context_spans) == 0:
            
            context_spans = [[0, -1]]
        return context_spans[0]

    def get_sentence_token_spans(self, context_ids):
        tokenizer = self.processor.tokenizer
        context_text = self.processor.tokenizer.decode(context_ids)
        context_tokens_text = [
            tokenizer.decode([token_id]).replace(" ", "") for token_id in context_ids
        ]
        sents = [sent.text for sent in spacy.load("en_core_web_sm")(context_text).sents]
        # if a sent is all " ", then merge it with the next sent
        for i in range(len(sents)):
            # if sents[i].strip() == "":
            if len(sents[i].strip()) <= 5:
                if i < len(sents) - 1:
                    sents[i + 1] = sents[i] + sents[i + 1]
                    sents[i] = ""
                else:
                    sents[i - 1] = sents[i - 1] + sents[i]
                    sents[i] = ""
        sents = [sent for sent in sents if sent != ""]

        # find sentence token spans
        sent_token_spans = []
        tk_start_idx = 0

        for i, sent in enumerate(sents):
            sent = sent.lstrip(" ")
            sent_num_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
            # find the end token index
            sent_text = sent.replace(" ", "")
            span_text = self.processor.tokenizer.decode(
                context_ids[tk_start_idx : tk_start_idx + sent_num_tokens]
            ).replace(" ", "")
            span_include_sent = span_text.find(sent_text) >= 0
            sent_include_span = sent_text.find(span_text) >= 0
            len_span = sent_num_tokens
            if span_include_sent and sent_include_span:  # pass
                pass
            elif span_include_sent and not sent_include_span:  # span is longer
                while True:
                    len_span -= 1
                    if tk_start_idx + len_span >= len(context_tokens_text):
                        break 
                    del_token = context_tokens_text[tk_start_idx + len_span]
                    span_text = span_text.rstrip(del_token)
                    if span_text.find(sent_text) < 0:  # span is shorter than sent
                        # len_span += 1
                        span_text = span_text + del_token
                        break
            elif not span_include_sent:  # span is shorter
                while True:
                    if tk_start_idx + len_span >= len(context_tokens_text):
                        break
                    add_token = context_tokens_text[tk_start_idx + len_span]
                    len_span += 1
                    span_text = span_text + add_token
                    if span_text.find(sent_text) >= 0:
                        break

            tk_end_idx = tk_start_idx + len_span
            sent_token_spans.append((tk_start_idx, tk_end_idx))
            tk_start_idx = tk_end_idx

            if not span_text.endswith(sent_text):  # last token contains the next sentence
                tk_start_idx -= 1

        assert len(sent_token_spans) == len(sents)

        return sent_token_spans, sents

    def get_passage_token_spans(self, context_ids, passage_delimiter=PASSAGE_DELIMITER):
        tokenizer = self.processor.tokenizer
        context_text = self.processor.tokenizer.decode(context_ids)
        passages = context_text.split(passage_delimiter)

        passage_token_spans = []
        tk_start_idx = 0

        for i, passage in enumerate(passages):
            passage = passage.lstrip(" ")
            passage_num_tokens = len(tokenizer.encode(passage, add_special_tokens=False))

            tk_end_idx = tk_start_idx + passage_num_tokens
            passage_token_spans.append((tk_start_idx, tk_end_idx))
            tk_start_idx = tk_end_idx

        assert len(passage_token_spans) == len(passages)

        return passage_token_spans, passages

    @torch.inference_mode() 
    def generate_self_elicit(self, question, image_query, context):
        
        inputs = self.get_inputs(question, image_query, text_eliciting=True)
        input_ids = inputs.input_ids.cpu().tolist()[0]
        
        
        target_device = self.model_self_elicit.device
        model_inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # 2. Calcolo Indici Contesto
        start = time.time()
        context_span = self.get_context_token_span(context, input_ids)
        
        # Preparazione spans (identica)
        passage_spans, passages = None, None
        sent_spans, sents = None, None
        context_ids = input_ids[context_span[0] : context_span[1]]
        # ... Logica scelta sentence/passage ...
        start = time.time()
        if self.args.self_elicit_gen_passage:
            sent_spans, sents = self.get_passage_token_spans(context_ids)
        elif self.args.self_elicit_gen_sen2pas:
            passage_spans, passages = self.get_passage_token_spans(context_ids)
            sent_spans, sents = self.get_sentence_token_spans(context_ids) # Esempio default
        else:
            sent_spans, sents = self.get_sentence_token_spans(context_ids) # Esempio default

        end = time.time()
        elapsed = end - start

        # Setup range layer
        # Qwen2.5: self.model_self_elicit.model.language_model.layers
        lm_layers = self.model_self_elicit.model.language_model.layers
        n_layers = len(lm_layers)

        layer_range = self.args.attention_text_layer_range
        
        if layer_range is not None and layer_range[0] == '[':
            try:
                start_layer, end_layer = map(int, layer_range[1:-1].split(','))
                if not (0 <= start_layer < end_layer <= n_layers):
                    raise ValueError
            except Exception:
                print(f"Invalid layer_range format: {layer_range}. Falling back to 'middle_half'.")
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
        elif layer_range is not None:
            if layer_range == 'all':
                start_layer, end_layer = 0, n_layers
            elif layer_range == 'last_half':
                start_layer, end_layer = n_layers // 2, n_layers
            elif layer_range == 'first_half':
                start_layer, end_layer = 0, n_layers // 2
            elif layer_range == 'middle_half':
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
            else:
                start_layer, end_layer = 0, n_layers - (n_layers // 4)
        else:
            start_layer = int(0.5 * n_layers)
            end_layer = int(1.0 * n_layers)

        attention_scores_data = {}
        sink_data = {}

        def create_optimized_hook(layer_idx, ctx_start, ctx_end):
            def hook(module, input, output):
                # log_memory(f"⚡ ATTN_HOOK_{layer_idx} (Inside)")

                
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    att_tensor = output[1] 
                    
                    # 1. SLICING & REDUCTION
                    # Attenzione all'ultimo token (-1) verso il contesto
                    relevant_att = att_tensor[0, :, -1, ctx_start:ctx_end]
                    averaged = relevant_att.mean(dim=0)
                    
                    # 2. SAVE TO CPU
                    attention_scores_data[layer_idx] = averaged.detach().cpu().float().numpy()
                    
                    # 3. MEMORY HACK
                    # return (output[0], None) + output[2:] # Opzionale: kill tensor
                
            return hook

        def create_hidden_hook(layer_idx, ctx_start, ctx_end, sink_dims_list):
            def hook(module, input, output):
                # output is tuple, first element is hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                ctx_hidden = hidden_states[0, ctx_start + 1:ctx_end, :]
                
                # Compute sink scores
                sink_vals = ctx_hidden[:, sink_dims_list]
                max_sink_val = sink_vals.abs().max(dim=1).values
                rms = torch.sqrt(ctx_hidden.pow(2).mean(dim=1))
                sink_score = max_sink_val / (rms + 1e-6)
                
                sink_data[layer_idx] = {
                    'sink_scores': sink_score.detach().cpu().float().numpy()
                }
            return hook
        
        model_name = self.args.model_name
        if model_name not in SINK_DIMS:
            print(f"Warning: SINK_DIMS not defined for model {model_name}. Using default.")
            sink_dims = [318, 1874, 1819]  # Default to 3B model dims
        else:
            sink_dims = SINK_DIMS[model_name]
        hooks = []
        for i, layer in enumerate(lm_layers):
            if start_layer <= i < end_layer:
                h = layer.self_attn.register_forward_hook(
                    create_optimized_hook(i, context_span[0], context_span[1])
                )
                hooks.append(h)
                h = layer.register_forward_hook(
                    create_hidden_hook(i, context_span[0], context_span[1], sink_dims)
                )
                hooks.append(h)
        
        # torch.cuda.empty_cache()

        try:
            with torch.inference_mode():
                # Qwen forward args: input_ids, attention_mask, pixel_values etc.
                self.model_self_elicit(**model_inputs, output_attentions=False, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
            del model_inputs
            
        elicited_context, evidence_sents, evidence_spans = self._process_attention_scores(
            attention_scores_data,
            sents,
            sent_spans,
            threshold=self.args.self_elicit_alpha,
            passages=passages,
            passage_spans=passage_spans
        )

        return elicited_context, evidence_sents, evidence_spans, elapsed

    def _process_attention_scores(self, attention_scores_dict, sents, sent_spans, threshold=0.5, passages=None, passage_spans=None):
        """
        Metodo helper interno che aggrega i punteggi (già estratti e su CPU) e seleziona le frasi.
        Sostituisce la vecchia funzione nested 'self_elicit'.
        """
        def find_passage_from_sentence(sent_span, passage_spans):
            for i, pas_span in enumerate(passage_spans):
                if sent_span[0] >= pas_span[0] and sent_span[1] <= pas_span[1]:
                    return i
            return -1
        
        if not attention_scores_dict:
            # Fallback se qualcosa è andato storto e non abbiamo attenzioni
            return " ".join(sents), [], []

        import numpy as np

        # Ordiniamo i layer per sicurezza (anche se il dict non è ordinato, vogliamo coerenza)
        sorted_layers = sorted(attention_scores_dict.keys())
        
        # Creiamo matrice [num_layers, context_len]
        # Nota: ogni array nel dict è già [context_len]
        att_layer_scores = np.array([attention_scores_dict[l] for l in sorted_layers])

        # Normalizzazione attraverso i layer (come nel codice originale)
        att_layer_scores /= (att_layer_scores.sum(axis=1, keepdims=True) + 1e-9)

        # Aggregazione finale sui token (media sui layer)
        att_token_scores = att_layer_scores.mean(axis=0)

        # Aggregazione per frasi
        sent_scores = []
        for sent_span in sent_spans:
            # Gestione caso span vuoto o indici errati
            if sent_span[1] > sent_span[0]:
                # Slice sicuro sugli score dei token
                score = att_token_scores[sent_span[0] : sent_span[1]].mean()
            else:
                score = 0.0
            sent_scores.append(score)
            
        sent_scores = np.array(sent_scores)

        # Selezione frasi sopra soglia
        target_sent_index = (sent_scores >= sent_scores.max() * threshold).nonzero()[0]
        target_passage_index = set()
        if self.args.self_elicit_gen_sen2pas and passages is not None and passage_spans is not None:
            for idx in target_sent_index:
                sent = sents[idx]
                is_valid_sent = len(sent.replace(" ", "")) > 5 and sent.strip() != '# Wiki Article:'.strip()
                pas_idx = find_passage_from_sentence(sent_spans[idx], passage_spans)
                if pas_idx != -1 and is_valid_sent:
                    target_passage_index.add(pas_idx)

        # Costruzione output
        elicited_context = ""
        evidence_sents = []
        evidence_spans = []
        
        marker_impstart = '<START_IMPORTANT_TXT>'
        marker_impend = '<END_IMPORTANT_TXT>'

        if self.args.self_elicit_gen_sen2pas and passages is not None and passage_spans is not None:
            for i, pas in enumerate(passages):
                if i in target_passage_index:
                    elicited_context += f"{marker_impstart} {pas} {marker_impend}{PASSAGE_DELIMITER}"
                    evidence_sents.append(pas)
                    evidence_spans.append(passage_spans[i])
                else:
                    elicited_context += f"{pas}{PASSAGE_DELIMITER}"
            return elicited_context, evidence_sents, evidence_spans

        sent_end = PASSAGE_DELIMITER if self.args.self_elicit_gen_passage else ""

        for i, sent in enumerate(sents):
            is_valid_sent = len(sent.replace(" ", "")) > 5 and sent.strip() != '# Wiki Article:'.strip()
            if i in target_sent_index and is_valid_sent:
                elicited_context += f"{marker_impstart} {sent} {marker_impend}{sent_end}"
                evidence_sents.append(sent)
                evidence_spans.append(sent_spans[i])
            else:
                elicited_context += f"{sent}{sent_end}"

        return elicited_context, evidence_sents, evidence_spans

    @torch.inference_mode() 
    def generate(self, question, image_query, image_cropped=None, annotated_image=None, bbox_2d=None, entity=None):

        inputs = self.get_inputs(question, image_query, image_cropped, annotated_image, bbox_2d, entity)
        inputs = inputs.to(self.model.device)

        gen_kwargs = {}
        if self.args.temperature is not None:
            gen_kwargs['temperature'] = self.args.temperature
        if self.args.top_p is not None:
            gen_kwargs['top_p'] = self.args.top_p
        if self.args.top_k_sampling is not None:
            gen_kwargs['top_k'] = self.args.top_k_sampling
        if self.args.repetition_penalty is not None:
            gen_kwargs['repetition_penalty'] = self.args.repetition_penalty
        if self.args.min_p is not None:
            gen_kwargs['min_p'] = self.args.min_p

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=self.args.max_new_tokens, 
            stop_strings=["</answer>"] if self.args.extract_reasoning else None,
            tokenizer=self.processor.tokenizer,
            use_cache=True,
            **gen_kwargs
        )
        # Subtract the len of <think> if reasoning is extracted, use tokenizer to be sure of how manuy tokens it is (should be 3)
        in_ids_lengths = [len(in_ids) for in_ids in inputs.input_ids]

        generated_ids_trimmed = [
            out_ids[in_ids_len :] for in_ids, out_ids, in_ids_len in zip(inputs.input_ids, generated_ids, in_ids_lengths)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Free memory
        # del inputs
        # del generated_ids
        # del generated_ids_trimmed
        # gc.collect()
        # torch.cuda.empty_cache()
        # Calculate number of image tokens in input
        # Find position of self.vis_start and self.vis_end in the input_ids
        input_ids_list = inputs.input_ids.cpu().tolist()[0]
        num_image_tokens = 0
        if self.token_vis_start and self.token_vis_end:
            try:
                start_idx = input_ids_list.index(self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_start))
                end_idx = input_ids_list.index(self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_end))
                num_image_tokens = end_idx - start_idx - 1  # Subtract the start and end tokens themselves
            except ValueError:
                # If the tokens are not found, we can set num_image_tokens to 0 or handle it as needed
                num_image_tokens = 0
        
        
        
        return output_text[0], in_ids_lengths[0], len(generated_ids_trimmed[0].cpu()), num_image_tokens

class InferenceModelQwen2_5_VL(InferenceModel):
    def __init__(self, args):
        super().__init__(args)

        n_gpus = torch.cuda.device_count()

        print("Loading Qwen2.5-VL model...")
        
        self.processor = AutoProcessor.from_pretrained(
            args.model_name,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            padding_side="left",
            trust_remote_code=True
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            output_attentions=False,
            trust_remote_code=True,
        )
        
        language_model_ref = self.model.model.language_model

        if args.text_elicit:
            num_layers = len(language_model_ref.layers)
            n_layers = num_layers
            half_layers = num_layers // 2
            
            device_map = "balanced"

            # 1. Carica il modello per Self-Elicit con Eager Attention per avere attention_mask corretta
            self.model_self_elicit = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_name,
                attn_implementation="flash_attention_2", # Necessario per ricevere attention_mask in eager_attention_forward
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                output_attentions=False,
                trust_remote_code=True
            )

            # Accesso rapido ai sottomoduli del modello elicit
            elicit_lm = self.model_self_elicit.model.language_model
            elicit_visual = self.model_self_elicit.model.visual

            self.model_self_elicit.gradient_checkpointing_disable()

            # Configurazione Attenzione Ibrida (Flash / Eager)
            # Copiamo la config dal primo layer
            config_wo_attn = copy.deepcopy(elicit_lm.layers[0].self_attn.config)
            config_wo_attn._attn_implementation = "flash_attention_2"
            config_wo_attn.output_attentions = False
            
            config_w_attn = copy.deepcopy(elicit_lm.layers[0].self_attn.config)
            config_w_attn._attn_implementation = "eager"
            config_w_attn.output_attentions = True

            layer_range = self.args.attention_text_layer_range
        
            if layer_range is not None and layer_range[0] == '[':
                try:
                    start_layer, end_layer = map(int, layer_range[1:-1].split(','))
                    if not (0 <= start_layer < end_layer <= n_layers):
                        raise ValueError
                except Exception:
                    print(f"Invalid layer_range format: {layer_range}. Falling back to 'middle_half'.")
                    start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
            elif layer_range is not None:
                if layer_range == 'all':
                    start_layer, end_layer = 0, n_layers
                elif layer_range == 'last_half':
                    start_layer, end_layer = n_layers // 2, n_layers
                elif layer_range == 'first_half':
                    start_layer, end_layer = 0, n_layers // 2
                elif layer_range == 'middle_half':
                    start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
                else:
                    start_layer, end_layer = 0, n_layers - (n_layers // 4)
            else:
                start_layer = int(0.5 * n_layers)
                end_layer = int(1.0 * n_layers)

            for i, layer in enumerate(elicit_lm.layers):
                if i >= start_layer and i < end_layer:
                    layer.self_attn.config = config_w_attn
                else:
                    layer.self_attn.config = config_wo_attn

            for block in elicit_visual.blocks:
                if hasattr(block.attn, 'config'):
                    block.attn.config.output_attentions = False

            for i, layer in enumerate(elicit_lm.layers):
                if i >= start_layer and i < end_layer:
                    layer.self_attn.config.output_attentions = True
                    remove_hook_from_module(layer, "attn_output_hook")
                    
                    # Monkey Patching con le funzioni equivalenti per Qwen2.5
                    layer.forward = types.MethodType(qwen2_5_vl_decoderlayer_forward, layer)
                    layer.self_attn.forward = types.MethodType(qwen2_5_vl_self_attn_forward, layer.self_attn)
                    
                    layer.input_layernorm.forward = types.MethodType(qwen2_5_vl_layernorm_forward, layer.input_layernorm)
                    layer.post_attention_layernorm.forward = types.MethodType(qwen2_5_vl_layernorm_forward, layer.post_attention_layernorm)
                    
                    layer.mlp.forward = types.MethodType(qwen2_5_vl_mlp_forward, layer.mlp)
                else:
                    layer.self_attn.config.output_attentions = False
                
            for p in self.model_self_elicit.parameters():
                p.requires_grad = False

            # 5. Configurazione finale di runtime
            self.model_self_elicit.config.use_cache = False
            # Assicuriamoci di settarlo anche nel LM interno
            self.model_self_elicit.model.config.use_cache = False 
            self.model_self_elicit.eval()
        
        # Configurazione modello principale (inference veloce)
        self.model.config.use_cache = True
        self.model.model.config.use_cache = True # Qwen struttura nidificata
        self.model.eval()
        
        print("Qwen2.5-VL model loaded.")

        self.token_vis_start = "<|vision_start|>"
        self.token_vis_end = "<|vision_end|>"

    @torch.inference_mode()
    def extract_attention_bbox(
        self, 
        image: Image.Image, 
        question: str, 
        entity: str,
        layer_range: str = 'middle_half'
    ) -> Optional[Dict[str, Tuple[int, int, int, int]]]:
        """
        Extract bounding boxes from attention maps using entity-to-vision attention.
        
        This method performs a forward pass to extract attention from entity tokens
        to visual tokens, filters out sink tokens, and computes bboxes using multiple methods.
        
        Args:
            image: PIL Image
            question: Question text containing the entity
            entity: Entity to focus on (e.g., "building", "car")
            layer_range: Which layers to use for attention extraction
                        ('middle_half', 'last_half', 'first_half', 'all')
        
        Returns:
            Dictionary with multiple bbox methods:
            {
                'weighted_centroid': (x1, y1, x2, y2),
                'morphological_t0.3_k7': (x1, y1, x2, y2),
                'morphological_t0.1_k7': (x1, y1, x2, y2),
                ...
                'average': (x1, y1, x2, y2)
            }
            Returns None if extraction fails.
        """
        from qwen_vl_utils import process_vision_info
        
        # Check if model supports this feature (needs sink dims)
        model_name = self.args.model_name
        if model_name not in SINK_DIMS:
            print(f"Warning: SINK_DIMS not defined for model {model_name}. Using default.")
            sink_dims = [318, 1874, 1819]  # Default to 3B model dims
        else:
            sink_dims = SINK_DIMS[model_name]

        SYSTEM_PROMPT = prompts.SYSTEM_PROMPT
        
        # 1. Prepare input
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": question}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.cpu().tolist()[0]
        
        # 2. Find entity tokens
        if entity is not None:
            entity_spans = self.find_text_token_spans(input_ids, entity)
            if not entity_spans:
                print(f"Warning: Entity '{entity}' not found in input. Cannot extract bbox.")
                return None
            entity_start, entity_end = entity_spans[0]
        else: # If teh entity is None we simply extract the attention weights correpsonding to the last token
            entity_start, entity_end = -1, len(input_ids)
        
        # Handle truncated entity
        if entity_end == -1:
            entity_end = len(input_ids)
        
        # 3. Vision tokens indices
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_start)
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_end)
        input_ids_tensor = inputs.input_ids[0]
        
        vision_start_positions = (input_ids_tensor == vision_start_id).nonzero(as_tuple=True)[0]
        vision_end_positions = (input_ids_tensor == vision_end_id).nonzero(as_tuple=True)[0]
        
        if len(vision_start_positions) == 0 or len(vision_end_positions) == 0:
            print("Warning: Could not find vision token boundaries.")
            return None
            
        vision_start_idx = vision_start_positions[0].item()
        vision_end_idx = vision_end_positions[0].item()
        
        # 5. Layer Selection
        n_layers = len(self.model.model.language_model.layers)
        if layer_range[0] == '[':
            try:
                start_layer, end_layer = map(int, layer_range[1:-1].split(','))
                if not (0 <= start_layer < end_layer <= n_layers):
                    raise ValueError
            except Exception:
                print(f"Invalid layer_range format: {layer_range}. Falling back to 'middle_half'.")
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
        else:
            if layer_range == 'all':
                start_layer, end_layer = 0, n_layers
            elif layer_range == 'last_half':
                start_layer, end_layer = n_layers // 2, n_layers
            elif layer_range == 'first_half':
                start_layer, end_layer = 0, n_layers // 2
            elif layer_range == 'middle_half':
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
            else:
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)

        # 6. Monkey patch for attention extraction from entity tokens
        # Save original forwards to restore later
        original_forwards = {}
        for i, layer in enumerate(self.model.model.language_model.layers):
            if start_layer <= i < end_layer:
                original_forwards[i] = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(
                    create_qwen2_5_vl_self_attn_forward(entity_start, entity_end), 
                    layer.self_attn
                )
        
        attention_data = {}
        sink_data = {}
        accumulated_activations = []

        def hook_fn(module, input, output):
            accumulated_activations.append(output[0].detach().cpu())


        def create_attn_hook(layer_idx, vis_start, vis_end):
            def hook(module, input, output):
                if len(output) > 1 and output[1] is not None:
                    att_tensor = output[1]
                    # Attention from entity tokens to vision tokens
                    # att_tensor shape: [batch, heads, entity_len, seq_len]
                    entity_to_vision = att_tensor[0, :, :, vis_start + 1:vis_end]
                    # Average over entity tokens, then over heads
                    per_head = entity_to_vision.mean(dim=1)  # [heads, n_vision]
                    mean_heads = per_head.mean(dim=0)  # [n_vision]
                    attention_data[layer_idx] = {
                        'mean': mean_heads.detach().cpu().float().numpy()
                    }
            return hook

        def create_hidden_hook(layer_idx, vis_start, vis_end, sink_dims_list):
            def hook(module, input, output):
                # output is tuple, first element is hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                vis_hidden = hidden_states[0, vis_start + 1:vis_end, :]
                
                # Compute sink scores
                sink_vals = vis_hidden[:, sink_dims_list]
                max_sink_val = sink_vals.abs().max(dim=1).values
                rms = torch.sqrt(vis_hidden.pow(2).mean(dim=1))
                sink_score = max_sink_val / (rms + 1e-6)
                
                sink_data[layer_idx] = {
                    'sink_scores': sink_score.detach().cpu().float().numpy()
                }
            return hook

        hooks = []
        layers = self.model.model.language_model.layers
        for i in range(start_layer, end_layer):
            h1 = layers[i].self_attn.register_forward_hook(
                create_attn_hook(i, vision_start_idx, vision_end_idx)
            )
            hooks.append(h1)
            h_sink = layers[i].register_forward_hook(
                create_hidden_hook(i, vision_start_idx, vision_end_idx, sink_dims)
            )
            hooks.append(h_sink)
            if i == start_layer:  # Only register the activation hook on the first layer we modify
                h = layers[i].register_forward_hook(hook_fn)
                hooks.append(h)
        
        # Move inputs to model device
        target_device = self.model.model.visual.patch_embed.proj.weight.device
        model_inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        try:
            self.model(**model_inputs, output_attentions=False)
        finally:
            # Remove hooks
            for h in hooks:
                h.remove()
            # Restore original forwards
            for i, orig_forward in original_forwards.items():
                layers[i].self_attn.forward = orig_forward
            del model_inputs
            # torch.cuda.empty_cache()

        if not attention_data:
            print("Warning: No attention data extracted.")
            return None

        # 7. Aggregate attention and sink scores
        sorted_layers = sorted(attention_data.keys())
        mean_agg = np.array([attention_data[l]['mean'] for l in sorted_layers]).mean(axis=0)
        
        sorted_layers_sink = sorted(sink_data.keys())
        if sorted_layers_sink:
            sink_scores_agg = np.array([sink_data[l]['sink_scores'] for l in sorted_layers_sink]).mean(axis=0)
        else:
            sink_scores_agg = np.zeros_like(mean_agg)

        grid_thw = inputs.image_grid_thw[0]
        h_grid, w_grid = int(grid_thw[1]), int(grid_thw[2])

        # 8. Reshape to grid
        expected_total = h_grid * w_grid
        n_vis_tokens = mean_agg.shape[0]
        
        if n_vis_tokens * 4 == expected_total:
            grid_shape = (h_grid // 2, w_grid // 2)
        elif n_vis_tokens * 2 == expected_total:
            grid_shape = (h_grid, w_grid // 2)
        elif n_vis_tokens == expected_total:
            grid_shape = (h_grid, w_grid)
        else:
            side = int(np.sqrt(n_vis_tokens))
            grid_shape = (side, side)

        try:
            attn_map = mean_agg.reshape(grid_shape)
            sink_map = sink_scores_agg.reshape(grid_shape)
        except ValueError as e:
            print(f"Warning: Could not reshape attention map: {e}")
            return None

        # 9. Normalize attention map
        attn_map_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # 10. Filter using percentile threshold
        actual_threshold = np.percentile(sink_map, SINK_PERCENTILE)
        is_sink_grid = sink_map >= actual_threshold
        
        # Also mark border tokens as sinks
        if is_sink_grid.shape[0] > 4 and is_sink_grid.shape[1] > 4:
            is_sink_grid[0, :] = True
            is_sink_grid[1, :] = True
            is_sink_grid[-1, :] = True
            is_sink_grid[-2, :] = True
            is_sink_grid[:, 0] = True
            is_sink_grid[:, 1] = True
            is_sink_grid[:, -1] = True
            is_sink_grid[:, -2] = True
        
        # Zero out sink tokens
        attn_map_filtered = attn_map_norm.copy()
        attn_map_filtered[is_sink_grid] = 0.0
        if attn_map_filtered.max() > 0:
            attn_map_filtered = attn_map_filtered / attn_map_filtered.max()
        
        # 11. Resize to image dimensions
        target_size = (image.size[0], image.size[1])
        attn_map_filtered_resized = cv2.resize(attn_map_filtered, target_size, interpolation=cv2.INTER_CUBIC)
        
        # 12. Extract bboxes using multiple methods
        # Weighted centroid
        try:
            bbox_weighted = extract_bbox_weighted_centroid(attn_map_filtered_resized, std_multiplier=self.args.weighted_centroid_std_multiplier)
        except Exception:
            bbox_weighted = (0, 0, image.size[0], image.size[1])
        
        results = {'weighted_centroid': bbox_weighted}

        # # Multiple morphological variants
        # morphological_configs = [
        #     (0.3, 7),   # Original (baseline)
        #     (0.1, 7),   # Lower threshold (larger bbox)
        #     (0.3, 15),  # Larger kernel (more expansion)
        #     (0.1, 15),  # Both (most inclusive)
        #     (0.0, 7),
        #     (0.0, 15),
        #     (0.0, 31),
        # ]
        
        # for threshold, kernel_size in morphological_configs:
        #     key = f'morphological_t{threshold}_k{kernel_size}'
        #     try:
        #         bbox_morpho = extract_bbox_morphological(attn_map_filtered_resized, threshold=threshold, kernel_size=kernel_size)
        #         results[key] = bbox_morpho
        #     except Exception:
        #         results[key] = (0, 0, image.size[0], image.size[1])
        
        # # Compute average between weighted_centroid and baseline morphological
        # bbox_morpho_baseline = results['morphological_t0.3_k7']
        # bbox_average = compute_average_bbox(bbox_weighted, bbox_morpho_baseline)
        # results['average'] = bbox_average
        
        # Debug: save raw tensors for later visualization in notebook
        if os.environ.get("DEBUG", "0") == "1":
            debug_dir = os.path.join(os.path.dirname(__file__), '..', '..', f'debug_attention_bbox_{self.args.dataset_name}')
            os.makedirs(debug_dir, exist_ok=True)
            entity_clean = entity.replace(" ", "_").replace("/", "_").replace("'", "") if entity is not None else 'None'
            sample_id = f'{entity_clean}_{abs(hash(question))}'
            sample_dir = os.path.join(debug_dir, sample_id)
            os.makedirs(sample_dir, exist_ok=True)
            
            import pickle
            
            # Save image
            image.save(os.path.join(sample_dir, 'image.png'))
            
            # Save attention maps as numpy
            np.save(os.path.join(sample_dir, 'attn_map_raw.npy'), attn_map_norm)
            np.save(os.path.join(sample_dir, 'attn_map_filtered.npy'), attn_map_filtered)
            np.save(os.path.join(sample_dir, 'attn_map_filtered_resized.npy'), attn_map_filtered_resized)
            np.save(os.path.join(sample_dir, 'sink_map.npy'), sink_map)
            np.save(os.path.join(sample_dir, 'is_sink_grid.npy'), is_sink_grid)
            
            # Save bboxes and metadata
            metadata = {
                'entity': entity,
                'question': question,
                'bbox_weighted': bbox_weighted,
                'results': results,
                'image_size': image.size,
                'grid_shape': (h_grid, w_grid),
                'vision_start_idx': vision_start_idx,
                'vision_end_idx': vision_end_idx,
            }
            with open(os.path.join(sample_dir, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save hidden state activations if available
            if accumulated_activations:
                try:
                    hidden_states = accumulated_activations[0].detach().cpu()
                    torch.save(hidden_states, os.path.join(sample_dir, 'hidden_states.pt'))
                    if sink_dims:
                        np.save(os.path.join(sample_dir, 'sink_dims.npy'), np.array(sink_dims))
                except Exception as e:
                    print(f"Warning: Could not save hidden state activations: {e}")
            
            print(f"Debug tensors saved to: {sample_dir}")
        
        return results
    
    def visualize_hidden_state_activations(
        self,
        hidden_states: torch.Tensor,
        image: Image.Image,
        vision_start_idx: int,
        vision_end_idx: int,
        sink_dims: Optional[list] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        output_dir: str = None,
        prefix: str = "activation_viz",
        n_highlight: int = 5
    ):
        """
        Visualize hidden state activations for BOS, sink and non-sink tokens.
        
        Produces:
        - 1 figure with 3 bar plots (BOS, sink, non-sink) on a shared y-scale
        - 1 image with multiple sink (red) and non-sink (cyan) tokens highlighted
        
        Args:
            hidden_states: Tensor [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
            image: PIL Image
            vision_start_idx: Start index of vision tokens
            vision_end_idx: End index of vision tokens
            sink_dims: Dimensions identifying sink tokens (optional)
            grid_shape: Vision token grid shape (h, w), computed if None
            output_dir: Output directory (default: debug_attention_bbox)
            prefix: File name prefix
            n_highlight: Number of tokens per category to highlight on image
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug_attention_bbox')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get sink_dims if not provided
        if sink_dims is None:
            model_name = self.args.model_name
            if model_name not in SINK_DIMS:
                print(f"Warning: SINK_DIMS not defined for model {model_name}. Using default.")
                sink_dims = [318, 1874, 1819]
            else:
                sink_dims = SINK_DIMS[model_name]
        
        # Move to CPU and handle shape
        hidden_states_cpu = hidden_states.detach().cpu()
        if hidden_states_cpu.ndim == 3:
            hidden_states_cpu = hidden_states_cpu[0]  # [seq_len, hidden_dim]
        
        # Extract vision tokens
        vision_states = hidden_states_cpu[vision_start_idx + 1:vision_end_idx]
        
        # Compute sink scores for vision tokens
        sink_dims_tensor = torch.tensor(sink_dims, dtype=torch.long)
        sink_vals = vision_states[:, sink_dims_tensor]
        max_sink_val = sink_vals.abs().max(dim=1).values
        rms = torch.sqrt(vision_states.pow(2).mean(dim=1))
        sink_scores = max_sink_val / (rms + 1e-6)
        
        # Identify sink and non-sink tokens using percentile threshold
        actual_threshold = np.percentile(sink_scores.to(dtype=torch.float32).numpy(), SINK_PERCENTILE)
        is_sink = sink_scores >= actual_threshold
        
        # Find representative sink and non-sink tokens
        sink_token_indices = torch.where(is_sink)[0]
        non_sink_token_indices = torch.where(~is_sink)[0]
        
        if len(sink_token_indices) == 0:
            print("Warning: No sink tokens found. Using first vision token as sink.")
            sink_token_indices = torch.tensor([0])
        
        if len(non_sink_token_indices) == 0:
            print("Warning: No non-sink tokens found. Using last vision token as non-sink.")
            non_sink_token_indices = torch.tensor([len(vision_states) - 1])
        
        # Select top-n sink tokens by sink score, and n non-sink tokens spread out
        n_sink_highlight = min(n_highlight, len(sink_token_indices))
        top_sink_scores, top_sink_order = sink_scores[sink_token_indices].sort(descending=True)
        highlighted_sink_idxs = sink_token_indices[top_sink_order[:n_sink_highlight]].tolist()
        
        n_nonsink_highlight = min(n_highlight, len(non_sink_token_indices))
        # Spread non-sink tokens evenly across the range
        step = max(1, len(non_sink_token_indices) // n_nonsink_highlight)
        highlighted_nonsink_idxs = non_sink_token_indices[::step][:n_nonsink_highlight].tolist()
        
        # Pick the first of each category for the bar chart
        main_sink_idx = highlighted_sink_idxs[0]
        main_nonsink_idx = highlighted_nonsink_idxs[0]
        
        # Extract states for visualization
        bos_state = hidden_states_cpu[0]
        sink_state = vision_states[main_sink_idx]
        non_sink_state = vision_states[main_nonsink_idx]
        
        # Normalize all states by RMS
        bos_rms = torch.sqrt(bos_state.pow(2).mean())
        bos_state_norm = bos_state / (bos_rms + 1e-6)
        
        sink_rms_val = torch.sqrt(sink_state.pow(2).mean())
        sink_state_norm = sink_state / (sink_rms_val + 1e-6)
        
        non_sink_rms_val = torch.sqrt(non_sink_state.pow(2).mean())
        non_sink_state_norm = non_sink_state / (non_sink_rms_val + 1e-6)
        
        # Get top-k dimensions for each
        k = 5
        bos_values, bos_top_indices = torch.topk(bos_state_norm.abs(), k=k)
        sink_values, sink_top_indices = torch.topk(sink_state_norm.abs(), k=k)
        non_sink_values, non_sink_top_indices = torch.topk(non_sink_state_norm.abs(), k=k)
        
        # Compute shared y-axis limit across all 3 plots
        global_ymax = max(
            bos_state_norm.abs().max().item(),
            sink_state_norm.abs().max().item(),
            non_sink_state_norm.abs().max().item()
        ) * 1.05
        
        # Create figure with 3 activation plots
        fig, axes = plt.subplots(3, 1, figsize=(14, 15))
        
        # Helper function to create activation plot
        def plot_activations(ax, state_norm, topk_indices, title, ymax, bar_color, sink_dims_list):
            all_vals = state_norm.abs().to(dtype=torch.float32).numpy()
            dim_indices = np.arange(len(all_vals))
            topk_set = set(topk_indices.cpu().tolist())
            
            # Find peak value and its dimension
            peak_val = all_vals.max()
            peak_dim = int(all_vals.argmax())

            bar_width = 0.5
            
            bars = ax.bar(dim_indices, all_vals, color=bar_color, width=bar_width)
            
            for idx in topk_set:
                ax.axvline(x=idx, color='gray', linestyle='--', linewidth=1, alpha=0.4)
                ax.bar(idx, all_vals[idx], color=bar_color, width=bar_width*5)
            
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Dimension", fontsize=13)
            ax.set_ylabel("|Activation| (RMS-normalized)", fontsize=13)
            ax.set_xlim([0, len(all_vals)])
            ax.set_ylim([0, ymax])
            
            # X ticks: only 0, sink dims, and last dimension
            n_dims = len(all_vals)
            tick_positions = sorted(set([0] + sink_dims_list + [n_dims - 1]))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(t) for t in tick_positions], fontsize=10)
            
            # Legend with peak value
            ax.plot([], [], ' ', label=f'Peak: dim {peak_dim} = {peak_val:.2f}')
            ax.legend(loc='upper right', fontsize=11, framealpha=0.8)
        
        sink_dims_list = sink_dims_tensor.tolist()
        plot_activations(axes[0], bos_state_norm, bos_top_indices, 
                        "BOS Token Activations", global_ymax, 'black', sink_dims_list)
        plot_activations(axes[1], sink_state_norm, sink_top_indices, 
                        f"Visual Sink Token (idx={main_sink_idx})", global_ymax, 'red', sink_dims_list)
        plot_activations(axes[2], non_sink_state_norm, non_sink_top_indices, 
                        f"Visual Non-Sink Token (idx={main_nonsink_idx})", global_ymax, 'cyan', sink_dims_list)
        
        plt.tight_layout()
        activations_path = os.path.join(output_dir, f'{prefix}_hidden_activations.png')
        plt.savefig(activations_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Activation plots saved to: {activations_path}")
        
        # Create image with multiple highlighted tokens
        try:
            if grid_shape is None:
                n_vis_tokens = vision_end_idx - vision_start_idx - 1
                grid_side = int(np.sqrt(n_vis_tokens))
                grid_h, grid_w = grid_side, grid_side
            else:
                grid_h, grid_w = grid_shape
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(image)
            
            img_width, img_height = image.size
            cell_width = img_width / grid_w
            cell_height = img_height / grid_h
            
            # Draw all highlighted sink tokens (red)
            for i, s_idx in enumerate(highlighted_sink_idxs):
                row = s_idx // grid_w
                col = s_idx % grid_w
                x = col * cell_width
                y = row * cell_height
                rect = Rectangle((x, y), cell_width, cell_height, 
                                  linewidth=3, edgecolor='red', facecolor='red', alpha=0.55,
                                  label='Sink Tokens' if i == 0 else None)
                ax.add_patch(rect)
            
            # Draw all highlighted non-sink tokens (cyan)
            for i, ns_idx in enumerate(highlighted_nonsink_idxs):
                row = ns_idx // grid_w
                col = ns_idx % grid_w
                x = col * cell_width
                y = row * cell_height
                rect = Rectangle((x, y), cell_width, cell_height, 
                                  linewidth=3, edgecolor='cyan', facecolor='cyan', alpha=0.55,
                                  label='Non-Sink Tokens' if i == 0 else None)
                ax.add_patch(rect)
            
            ax.set_title("Sink (red) vs Non-Sink (cyan) Visual Tokens", fontsize=16)
            ax.axis('off')
            ax.legend(loc='upper right', fontsize=12)
            
            plt.tight_layout()
            image_path = os.path.join(output_dir, f'{prefix}_tokens_highlighted.png')
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Highlighted tokens image saved to: {image_path}")
            
        except Exception as e:
            print(f"Warning: Could not create highlighted tokens image: {e}")
        
        return {
            'activations_path': activations_path,
            'bos_top_dims': bos_top_indices.tolist(),
            'sink_token_idx': main_sink_idx,
            'sink_top_dims': sink_top_indices.tolist(),
            'non_sink_token_idx': main_nonsink_idx,
            'non_sink_top_dims': non_sink_top_indices.tolist(),
            'highlighted_sink_idxs': highlighted_sink_idxs,
            'highlighted_nonsink_idxs': highlighted_nonsink_idxs
        }
 

class InferenceModelInternVL(InferenceModel):
    def __init__(self, args):
        super().__init__(args)

        n_gpus = torch.cuda.device_count()

        print("Loading internvl model...")
        self.processor = AutoProcessor.from_pretrained(
            args.model_name,
            model_max_length=args.model_max_length,
            padding_side="left",
            use_fast=True,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            trust_remote_code=True
        )

        self.model = InternVLForConditionalGeneration.from_pretrained( #AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            output_attentions=False,
            trust_remote_code=True,
        )

        if args.text_elicit:
            num_layers = len(self.model.language_model.layers)
            # Split only the second half of the layers on the available GPUs in sequence
            half_layers = num_layers // 2
            device_map = "balanced"


            # 1. Carica il modello con Eager Attention per avere attention_mask corretta
            self.model_self_elicit = InternVLForConditionalGeneration.from_pretrained(
                args.model_name,
                attn_implementation="flash_attention_2", # Necessario per ricevere attention_mask in eager_attention_forward
                torch_dtype=torch.bfloat16,
                device_map=device_map, #device_map_strategy,
                output_attentions=False,
            )

            self.model_self_elicit.gradient_checkpointing_disable()

            config_wo_attn = copy.deepcopy(self.model_self_elicit.model.language_model.layers[0].self_attn.config)
            config_wo_attn._attn_implementation = "flash_attention_2"
            config_wo_attn.output_attentions = False
            config_w_attn = copy.deepcopy(self.model_self_elicit.model.language_model.layers[0].self_attn.config)
            config_w_attn._attn_implementation = "eager"
            config_w_attn.output_attentions = True

            for i, layer in enumerate(self.model_self_elicit.model.language_model.layers):
                if i >= half_layers:
                    layer.self_attn.config = config_w_attn
                else:
                    layer.self_attn.config = config_wo_attn

            # 3. Disabilita le attenzioni per l'encoder visuale (già protetto da Flash Attention)
            for block in self.model_self_elicit.vision_tower.encoder.layer:
                block.attention.config.output_attentions = False

            # 4. Configura l'output delle attenzioni solo per la seconda metà dei layer testuali
            n_layers = len(self.model_self_elicit.model.language_model.layers)
            half_layers = n_layers // 2

            for i, layer in enumerate(self.model_self_elicit.model.language_model.layers):
                if i >= half_layers:
                    layer.self_attn.config.output_attentions = True
                    remove_hook_from_module(layer, "attn_output_hook") # DA DEBUGGARE
                    layer.forward = types.MethodType(qwen3_decoderlayer_forward, layer)
                    layer.self_attn.forward = types.MethodType(qwen3_self_attn_forward, layer.self_attn)
                    layer.self_attn.q_norm.forward = types.MethodType(qwen3_layernorm_forward, layer.self_attn.q_norm)
                    layer.self_attn.k_norm.forward = types.MethodType(qwen3_layernorm_forward, layer.self_attn.k_norm)
                    layer.input_layernorm.forward = types.MethodType(qwen3_layernorm_forward, layer.input_layernorm)
                    layer.post_attention_layernorm.forward = types.MethodType(qwen3_layernorm_forward, layer.post_attention_layernorm)
                    layer.mlp.forward = types.MethodType(qwen3_mlp_forward, layer.mlp)
                else:
                    layer.self_attn.config.output_attentions = False
                

            # ???
            for p in self.model_self_elicit.parameters():
                p.requires_grad = False

            # 5. Configurazione finale di runtime
            self.model_self_elicit.model.config.use_cache = False
            self.model_self_elicit.model.language_model.config.use_cache = False
            self.model_self_elicit.eval()
        
        self.model.model.config.use_cache = True
        self.model.model.language_model.config.use_cache = True
        self.model.eval()
        # self.model.cuda()
        print("InternVL model loaded.")
        self.token_vis_start = "<img>"
        self.token_vis_end = "</img>"

    @torch.inference_mode()
    def extract_attention_bbox(
        self, 
        image: Image.Image, 
        question: str, 
        entity: str,
        layer_range: str = 'middle_half'
    ) -> Optional[Dict[str, Tuple[int, int, int, int]]]:
        """
        Extract bounding boxes from attention maps using entity-to-vision attention.
        
        This method performs a forward pass to extract attention from entity tokens
        to visual tokens, filters out sink tokens, and computes bboxes using multiple methods.
        
        Args:
            image: PIL Image
            question: Question text containing the entity
            entity: Entity to focus on (e.g., "building", "car")
            layer_range: Which layers to use for attention extraction
                        ('middle_half', 'last_half', 'first_half', 'all') or number range like '[2,5]'
        
        Returns:
            Dictionary with multiple bbox methods:
            {
                'weighted_centroid': (x1, y1, x2, y2),
                'morphological_t0.3_k7': (x1, y1, x2, y2),
                'morphological_t0.1_k7': (x1, y1, x2, y2),
                ...
                'average': (x1, y1, x2, y2)
            }
            Returns None if extraction fails.
        """
        from qwen_vl_utils import process_vision_info
        
        # Check if model supports this feature (needs sink dims)
        model_name = self.args.model_name
        if model_name not in SINK_DIMS:
            print(f"Warning: SINK_DIMS not defined for model {model_name}. Using default.")
            sink_dims = [318, 1874, 1819]  # Default to 3B model dims
        else:
            sink_dims = SINK_DIMS[model_name]

        SYSTEM_PROMPT = prompts.SYSTEM_PROMPT
        
        # 1. Prepare input
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": question}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.cpu().tolist()[0]
        
        # 2. Find entity tokens
        if entity is not None:
            entity_spans = self.find_text_token_spans(input_ids, entity)
            if not entity_spans:
                entity_start, entity_end = -1, len(input_ids)
            else:
                entity_start, entity_end = entity_spans[0]
        else: # If teh entity is None we simply extract the attention weights correpsonding to the last token
            entity_start, entity_end = -1, len(input_ids)
        
        # Handle truncated entity
        if entity_end == -1:
            entity_end = len(input_ids)

        # 3. Vision tokens indices
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_start)
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_end)
        input_ids_tensor = inputs.input_ids[0]
        
        vision_start_positions = (input_ids_tensor == vision_start_id).nonzero(as_tuple=True)[0]
        vision_end_positions = (input_ids_tensor == vision_end_id).nonzero(as_tuple=True)[0]
        
        if len(vision_start_positions) == 0 or len(vision_end_positions) == 0:
            print("Warning: Could not find vision token boundaries.")
            return None
            
        vision_start_idx = vision_start_positions[0].item()
        vision_end_idx = vision_end_positions[0].item()
        
        # 5. Layer Selection
        n_layers = len(self.model.model.language_model.layers)
        if layer_range[0] == '[':
            try:
                start_layer, end_layer = map(int, layer_range[1:-1].split(','))
                if not (0 <= start_layer < end_layer <= n_layers):
                    raise ValueError
            except Exception:
                print(f"Invalid layer_range format: {layer_range}. Falling back to 'middle_half'.")
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
        else:
            if layer_range == 'all':
                start_layer, end_layer = 0, n_layers
            elif layer_range == 'last_half':
                start_layer, end_layer = n_layers // 2, n_layers
            elif layer_range == 'first_half':
                start_layer, end_layer = 0, n_layers // 2
            elif layer_range == 'middle_half':
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
            else:
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
                print(f"Invalid layer_range value: {layer_range}. Falling back to 'middle_half'.")
        # 6. Monkey patch for attention extraction from entity tokens
        # Save original forwards to restore later
        original_forwards = {}
        for i, layer in enumerate(self.model.model.language_model.layers):
            if start_layer <= i < end_layer:
                original_forwards[i] = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(
                    create_qwen3_self_attn_forward(entity_start, entity_end), 
                    layer.self_attn
                )
        
        attention_data = {}
        sink_data = {}

        def create_attn_hook(layer_idx, vis_start, vis_end):
            def hook(module, input, output):
                if len(output) > 1 and output[1] is not None:
                    att_tensor = output[1]
                    # Attention from entity tokens to vision tokens
                    # att_tensor shape: [batch, heads, entity_len, seq_len]
                    entity_to_vision = att_tensor[0, :, :, vis_start + 1:vis_end]
                    # Average over entity tokens, then over heads
                    per_head = entity_to_vision.mean(dim=1)  # [heads, n_vision]
                    mean_heads = per_head.mean(dim=0)  # [n_vision]
                    attention_data[layer_idx] = {
                        'mean': mean_heads.detach().cpu().float().numpy()
                    }
            return hook

        def create_hidden_hook(layer_idx, vis_start, vis_end, sink_dims_list):
            def hook(module, input, output):
                # output is tuple, first element is hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                vis_hidden = hidden_states[0, vis_start + 1:vis_end, :]
                
                # Compute sink scores
                sink_vals = vis_hidden[:, sink_dims_list]
                max_sink_val = sink_vals.abs().max(dim=1).values
                rms = torch.sqrt(vis_hidden.pow(2).mean(dim=1))
                sink_score = max_sink_val / (rms + 1e-6)
                
                sink_data[layer_idx] = {
                    'sink_scores': sink_score.detach().cpu().float().numpy()
                }
            return hook

        hooks = []
        layers = self.model.model.language_model.layers
        for i in range(start_layer, end_layer):
            h1 = layers[i].self_attn.register_forward_hook(
                create_attn_hook(i, vision_start_idx, vision_end_idx)
            )
            hooks.append(h1)
            h_sink = layers[i].register_forward_hook(
                create_hidden_hook(i, vision_start_idx, vision_end_idx, sink_dims)
            )
            hooks.append(h_sink)
        
        # Move inputs to model device
        target_device = self.model.device
        model_inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        try:
            self.model(**model_inputs, output_attentions=False)
        finally:
            # Remove hooks
            for h in hooks:
                h.remove()
            # Restore original forwards
            for i, orig_forward in original_forwards.items():
                layers[i].self_attn.forward = orig_forward
            del model_inputs
            # torch.cuda.empty_cache()

        if not attention_data:
            print("Warning: No attention data extracted.")
            return None

        # 7. Aggregate attention and sink scores
        sorted_layers = sorted(attention_data.keys())
        mean_agg = np.array([attention_data[l]['mean'] for l in sorted_layers]).mean(axis=0)
        
        sorted_layers_sink = sorted(sink_data.keys())
        if sorted_layers_sink:
            sink_scores_agg = np.array([sink_data[l]['sink_scores'] for l in sorted_layers_sink]).mean(axis=0)
        else:
            sink_scores_agg = np.zeros_like(mean_agg)

        # 8. InternVL 3.5 Specific Reshape Logic
        # InternVL constants
        TOKENS_PER_TILE = 256
        GRID_SIDE = 16  # sqrt(256)

        n_vis_tokens = mean_agg.shape[0]
        
        # InternVL structure: [Global(256)] + [Tile1(256)] + ... + [TileN(256)]
        # We verify if the token count makes sense
        if n_vis_tokens % TOKENS_PER_TILE != 0:
            print(f"Warning: Token count {n_vis_tokens} is not a multiple of {TOKENS_PER_TILE}. Might not be InternVL or tokens are truncated.")
            return None

        num_tiles = (n_vis_tokens // TOKENS_PER_TILE) - 1
        
        if num_tiles <= 0:
             # Case where there is only Global view (rare for high res) or something wrong
             # We just treat the only block as a single tile
             num_tiles = 1
             local_tokens = mean_agg
        else:
            # Drop the first 256 tokens (Global View) because they don't map linearly 
            # to high-res spatial coordinates. We only want the Local Tiles.
            local_tokens = mean_agg[TOKENS_PER_TILE:]

        # --- Derive Tile Layout (H x W) ---
        # InternVL decides layout based on aspect ratio. We must reverse-engineer it.
        # We look for h * w = num_tiles that best matches image aspect ratio.
        img_w, img_h = image.size
        target_ratio = img_w / img_h
        
        best_h, best_w = 1, num_tiles
        min_error = float('inf')

        # Find factors of num_tiles
        for h in range(1, int(math.sqrt(num_tiles)) + 1):
            if num_tiles % h == 0:
                w = num_tiles // h
                # Check (h, w)
                ratio_1 = w / h
                err_1 = abs(target_ratio - ratio_1)
                if err_1 < min_error:
                    min_error = err_1
                    best_h, best_w = h, w
                
                # Check (w, h) - swap
                ratio_2 = h / w
                err_2 = abs(target_ratio - ratio_2)
                if err_2 < min_error:
                    min_error = err_2
                    best_h, best_w = w, h

        # --- Stitching the Tiles ---
        # Each tile is 256 tokens -> becomes 16x16 grid
        # Final map size in "token space" will be (best_h * 16, best_w * 16)
        
        stitched_map = np.zeros((best_h * GRID_SIDE, best_w * GRID_SIDE), dtype=np.float32)
        
        # Reshape local tokens into tiles: [num_tiles, 16, 16]
        # Note: We must be careful with 'C' (row-major) vs order. 
        # InternVL ViT outputs [Batch, Sequence, Dim], sequence is usually row-major tiles.
        try:
            tiles = local_tokens.reshape(num_tiles, GRID_SIDE, GRID_SIDE)
        except ValueError:
             print("Reshape failed. Token count mismatch.")
             return None

        for i in range(num_tiles):
            # Calculate position of this tile in the grid
            row = i // best_w
            col = i % best_w
            
            y_start = row * GRID_SIDE
            x_start = col * GRID_SIDE
            
            stitched_map[y_start : y_start+GRID_SIDE, x_start : x_start+GRID_SIDE] = tiles[i]

        attn_map = stitched_map


        # Assuming sink_map logic is similar or you skip it for InternVL 
        # (InternVL doesn't have explicit sink registers like Qwen usually)
        # If you want to keep sink map logic, repeat the stitching process for sink_scores_agg
        if sink_scores_agg is not None and len(sink_scores_agg) > 0:
             # Same slicing logic
             if num_tiles > 0 and len(sink_scores_agg) > TOKENS_PER_TILE:
                 sink_local = sink_scores_agg[TOKENS_PER_TILE:]
             else:
                 sink_local = sink_scores_agg
             
             sink_tiles = sink_local.reshape(num_tiles, GRID_SIDE, GRID_SIDE)
             stitched_sink = np.zeros_like(stitched_map)
             for i in range(num_tiles):
                row = i // best_w
                col = i % best_w
                y_start, x_start = row * GRID_SIDE, col * GRID_SIDE
                stitched_sink[y_start:y_start+GRID_SIDE, x_start:x_start+GRID_SIDE] = sink_tiles[i]
             sink_map = stitched_sink
        else:
             sink_map = np.zeros_like(attn_map)

        # 9. Normalize attention map
        attn_map_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # 10. Filter using percentile threshold
        actual_threshold = np.percentile(sink_map, SINK_PERCENTILE)
        is_sink_grid = sink_map >= actual_threshold
        
        # Also mark border tokens as sinks
        if is_sink_grid.shape[0] > 4 and is_sink_grid.shape[1] > 4:
            is_sink_grid[0, :] = True
            is_sink_grid[1, :] = True
            is_sink_grid[-1, :] = True
            is_sink_grid[-2, :] = True
            is_sink_grid[:, 0] = True
            is_sink_grid[:, 1] = True
            is_sink_grid[:, -1] = True
            is_sink_grid[:, -2] = True
        
        # Zero out sink tokens
        attn_map_filtered = attn_map_norm.copy()
        attn_map_filtered[is_sink_grid] = 0.0
        if attn_map_filtered.max() > 0:
            attn_map_filtered = attn_map_filtered / attn_map_filtered.max()
        
        # 11. Resize to image dimensions
        target_size = (image.size[0], image.size[1])
        attn_map_filtered_resized = cv2.resize(attn_map_filtered, target_size, interpolation=cv2.INTER_CUBIC)

        # visualiza_attn_map(image, cv2.resize(attn_map_norm, target_size, interpolation=cv2.INTER_CUBIC), 'debug_attn_map_w_sink.png')
        # visualiza_attn_map(image, attn_map_filtered_resized, 'debug_attn_map_wo_sink.png')
        
        # 12. Extract bboxes using multiple methods
        # Weighted centroid
        try:
            bbox_weighted = extract_bbox_weighted_centroid(attn_map_filtered_resized, std_multiplier=2.0)
        except Exception:
            bbox_weighted = (0, 0, image.size[0], image.size[1])
        
        # Multiple morphological variants
        morphological_configs = [
            (0.3, 7),   # Original (baseline)
            (0.1, 7),   # Lower threshold (larger bbox)
            (0.3, 15),  # Larger kernel (more expansion)
            (0.1, 15),  # Both (most inclusive)
            (0.0, 7),
            (0.0, 15),
            (0.0, 31),
        ]
        
        results = {'weighted_centroid': bbox_weighted}
        
        for threshold, kernel_size in morphological_configs:
            key = f'morphological_t{threshold}_k{kernel_size}'
            try:
                bbox_morpho = extract_bbox_morphological(attn_map_filtered_resized, threshold=threshold, kernel_size=kernel_size)
                results[key] = bbox_morpho
            except Exception:
                results[key] = (0, 0, image.size[0], image.size[1])
        
        # Compute average between weighted_centroid and baseline morphological
        bbox_morpho_baseline = results['morphological_t0.3_k7']
        bbox_average = compute_average_bbox(bbox_weighted, bbox_morpho_baseline)
        results['average'] = bbox_average

        if os.environ.get("DEBUG", "0") == "1":
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            
            # 1. Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            
            # 2. Raw attention map overlay
            axes[0, 1].imshow(image)
            attn_overlay_raw = cv2.resize(attn_map_norm, (image.size[0], image.size[1]), interpolation=cv2.INTER_CUBIC)
            axes[0, 1].imshow(attn_overlay_raw, alpha=0.6, cmap='jet')
            axes[0, 1].set_title("Raw Attention Map Overlay")
            axes[0, 1].axis('off')
            
            # 3. Filtered attention map overlay (without sinks)
            axes[1, 0].imshow(image)
            axes[1, 0].imshow(attn_map_filtered_resized, alpha=0.6, cmap='jet')
            axes[1, 0].set_title("Filtered Attention Map (No Sinks)")
            axes[1, 0].axis('off')
            
            # 4. Weighted centroid bbox on image
            axes[1, 1].imshow(image)
            x1, y1, x2, y2 = bbox_weighted
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none')
            axes[1, 1].add_patch(rect)
            axes[1, 1].set_title("Weighted Centroid BBox")
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            debug_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug_attention_bbox')
            os.makedirs(debug_dir, exist_ok=True)
            entity = entity.replace(" ", "_").replace("/", "_").replace("'", "") if entity is not None else 'None'
            debug_path = os.path.join(debug_dir, f'attn_bbox_debug_{entity}_{hash(question)}.png')
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Debug attention bbox plot saved to: {debug_path}")

        
        return results

class InferenceModelQwen2VL(InferenceModel):
    def __init__(self, args):
        super().__init__(args)

        n_gpus = torch.cuda.device_count()

        print("Loading Qwen2-VL model...")
        
        self.processor = AutoProcessor.from_pretrained(
            args.model_name,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            padding_side="left",
            trust_remote_code=True
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            output_attentions=False,
            trust_remote_code=True,
        )
        
        language_model_ref = self.model.model.language_model

        if args.text_elicit:
            num_layers = len(language_model_ref.layers)
            half_layers = num_layers // 2
            
            device_map = "balanced"

            # 1. Carica il modello per Self-Elicit con Eager Attention per avere attention_mask corretta
            self.model_self_elicit = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_name,
                attn_implementation="flash_attention_2", # Necessario per ricevere attention_mask in eager_attention_forward
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                output_attentions=False,
                trust_remote_code=True
            )

            # Accesso rapido ai sottomoduli del modello elicit
            elicit_lm = self.model_self_elicit.model.language_model
            elicit_visual = self.model_self_elicit.model.visual

            self.model_self_elicit.gradient_checkpointing_disable()

            # Configurazione Attenzione Ibrida (Flash / Eager)
            # Copiamo la config dal primo layer
            config_wo_attn = copy.deepcopy(elicit_lm.layers[0].self_attn.config)
            config_wo_attn._attn_implementation = "flash_attention_2"
            config_wo_attn.output_attentions = False
            
            config_w_attn = copy.deepcopy(elicit_lm.layers[0].self_attn.config)
            config_w_attn._attn_implementation = "eager"
            config_w_attn.output_attentions = True

            for i, layer in enumerate(elicit_lm.layers):
                if i >= half_layers:
                    layer.self_attn.config = config_w_attn
                else:
                    layer.self_attn.config = config_wo_attn

            for block in elicit_visual.blocks:
                if hasattr(block.attn, 'config'):
                    block.attn.config.output_attentions = False

            # 4. Configura l'output delle attenzioni e Monkey Patching
            n_layers = len(elicit_lm.layers)
            half_layers = n_layers // 2

            for i, layer in enumerate(elicit_lm.layers):
                if i >= half_layers:
                    layer.self_attn.config.output_attentions = True
                    remove_hook_from_module(layer, "attn_output_hook")
                    
                    # Monkey Patching con le funzioni equivalenti per Qwen2.5
                    layer.forward = types.MethodType(qwen2vl_decoderlayer_forward, layer)
                    layer.self_attn.forward = types.MethodType(qwen2vl_self_attn_forward, layer.self_attn)
                    
                    layer.input_layernorm.forward = types.MethodType(qwen2_5_vl_layernorm_forward, layer.input_layernorm)
                    layer.post_attention_layernorm.forward = types.MethodType(qwen2_5_vl_layernorm_forward, layer.post_attention_layernorm)
                    
                    layer.mlp.forward = types.MethodType(qwen2_5_vl_mlp_forward, layer.mlp)
                else:
                    layer.self_attn.config.output_attentions = False
                
            for p in self.model_self_elicit.parameters():
                p.requires_grad = False

            # 5. Configurazione finale di runtime
            self.model_self_elicit.config.use_cache = False
            # Assicuriamoci di settarlo anche nel LM interno
            self.model_self_elicit.model.config.use_cache = False 
            self.model_self_elicit.eval()
        
        # Configurazione modello principale (inference veloce)
        self.model.config.use_cache = True
        self.model.model.config.use_cache = True # Qwen struttura nidificata
        self.model.eval()
        
        print("Qwen2.5-VL model loaded.")

        self.token_vis_start = "<|vision_start|>"
        self.token_vis_end = "<|vision_end|>"

    @torch.inference_mode()
    def extract_attention_bbox(
        self, 
        image: Image.Image, 
        question: str, 
        entity: str,
        layer_range: str = 'middle_half'
    ) -> Optional[Dict[str, Tuple[int, int, int, int]]]:
        """
        Extract bounding boxes from attention maps using entity-to-vision attention.
        
        This method performs a forward pass to extract attention from entity tokens
        to visual tokens, filters out sink tokens, and computes bboxes using multiple methods.
        
        Args:
            image: PIL Image
            question: Question text containing the entity
            entity: Entity to focus on (e.g., "building", "car")
            layer_range: Which layers to use for attention extraction
                        ('middle_half', 'last_half', 'first_half', 'all')
        
        Returns:
            Dictionary with multiple bbox methods:
            {
                'weighted_centroid': (x1, y1, x2, y2),
                'morphological_t0.3_k7': (x1, y1, x2, y2),
                'morphological_t0.1_k7': (x1, y1, x2, y2),
                ...
                'average': (x1, y1, x2, y2)
            }
            Returns None if extraction fails.
        """
        from qwen_vl_utils import process_vision_info
        
        # Check if model supports this feature (needs sink dims)
        model_name = self.args.model_name
        if model_name not in SINK_DIMS:
            print(f"Warning: SINK_DIMS not defined for model {model_name}. Using default.")
            sink_dims = [318, 1874, 1819]  # Default to 3B model dims
        else:
            sink_dims = SINK_DIMS[model_name]

        SYSTEM_PROMPT = prompts.SYSTEM_PROMPT
        
        # 1. Prepare input
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": question}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.cpu().tolist()[0]
        
        # 2. Find entity tokens
        if entity is not None:
            entity_spans = self.find_text_token_spans(input_ids, entity)
            if not entity_spans:
                print(f"Warning: Entity '{entity}' not found in input. Cannot extract bbox.")
                return None
            entity_start, entity_end = entity_spans[0]
        else: # If teh entity is None we simply extract the attention weights correpsonding to the last token
            entity_start, entity_end = -1, len(input_ids)
        
        # Handle truncated entity
        if entity_end == -1:
            entity_end = len(input_ids)
        
        # 3. Vision tokens indices
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_start)
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_end)
        input_ids_tensor = inputs.input_ids[0]
        
        vision_start_positions = (input_ids_tensor == vision_start_id).nonzero(as_tuple=True)[0]
        vision_end_positions = (input_ids_tensor == vision_end_id).nonzero(as_tuple=True)[0]
        
        if len(vision_start_positions) == 0 or len(vision_end_positions) == 0:
            print("Warning: Could not find vision token boundaries.")
            return None
            
        vision_start_idx = vision_start_positions[0].item()
        vision_end_idx = vision_end_positions[0].item()
        
        # 5. Layer Selection
        n_layers = len(self.model.model.language_model.layers)
        if layer_range[0] == '[':
            try:
                start_layer, end_layer = map(int, layer_range[1:-1].split(','))
                if not (0 <= start_layer < end_layer <= n_layers):
                    raise ValueError
            except Exception:
                print(f"Invalid layer_range format: {layer_range}. Falling back to 'middle_half'.")
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
        else:
            if layer_range == 'all':
                start_layer, end_layer = 0, n_layers
            elif layer_range == 'last_half':
                start_layer, end_layer = n_layers // 2, n_layers
            elif layer_range == 'first_half':
                start_layer, end_layer = 0, n_layers // 2
            elif layer_range == 'middle_half':
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
            else:
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)

        # 6. Monkey patch for attention extraction from entity tokens
        # Save original forwards to restore later
        original_forwards = {}
        for i, layer in enumerate(self.model.model.language_model.layers):
            if start_layer <= i < end_layer:
                original_forwards[i] = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(
                    create_qwen2vl_self_attn_forward(entity_start, entity_end), 
                    layer.self_attn
                )
        
        attention_data = {}
        sink_data = {}

        def create_attn_hook(layer_idx, vis_start, vis_end):
            def hook(module, input, output):
                if len(output) > 1 and output[1] is not None:
                    att_tensor = output[1]
                    # Attention from entity tokens to vision tokens
                    # att_tensor shape: [batch, heads, entity_len, seq_len]
                    entity_to_vision = att_tensor[0, :, :, vis_start + 1:vis_end]
                    # Average over entity tokens, then over heads
                    per_head = entity_to_vision.mean(dim=1)  # [heads, n_vision]
                    mean_heads = per_head.mean(dim=0)  # [n_vision]
                    attention_data[layer_idx] = {
                        'mean': mean_heads.detach().cpu().float().numpy()
                    }
            return hook

        def create_hidden_hook(layer_idx, vis_start, vis_end, sink_dims_list):
            def hook(module, input, output):
                # output is tuple, first element is hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                vis_hidden = hidden_states[0, vis_start + 1:vis_end, :]
                
                # Compute sink scores
                sink_vals = vis_hidden[:, sink_dims_list]
                max_sink_val = sink_vals.abs().max(dim=1).values
                rms = torch.sqrt(vis_hidden.pow(2).mean(dim=1))
                sink_score = max_sink_val / (rms + 1e-6)
                
                sink_data[layer_idx] = {
                    'sink_scores': sink_score.detach().cpu().float().numpy()
                }
            return hook

        hooks = []
        layers = self.model.model.language_model.layers
        for i in range(start_layer, end_layer):
            h1 = layers[i].self_attn.register_forward_hook(
                create_attn_hook(i, vision_start_idx, vision_end_idx)
            )
            hooks.append(h1)
            h_sink = layers[i].register_forward_hook(
                create_hidden_hook(i, vision_start_idx, vision_end_idx, sink_dims)
            )
            hooks.append(h_sink)
        
        # Move inputs to model device
        target_device = self.model.model.visual.patch_embed.proj.weight.device
        model_inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        try:
            self.model(**model_inputs, output_attentions=False)
        finally:
            # Remove hooks
            for h in hooks:
                h.remove()
            # Restore original forwards
            for i, orig_forward in original_forwards.items():
                layers[i].self_attn.forward = orig_forward
            del model_inputs
            # torch.cuda.empty_cache()

        if not attention_data:
            print("Warning: No attention data extracted.")
            return None

        # 7. Aggregate attention and sink scores
        sorted_layers = sorted(attention_data.keys())
        mean_agg = np.array([attention_data[l]['mean'] for l in sorted_layers]).mean(axis=0)
        
        sorted_layers_sink = sorted(sink_data.keys())
        if sorted_layers_sink:
            sink_scores_agg = np.array([sink_data[l]['sink_scores'] for l in sorted_layers_sink]).mean(axis=0)
        else:
            sink_scores_agg = np.zeros_like(mean_agg)

        grid_thw = inputs.image_grid_thw[0]
        h_grid, w_grid = int(grid_thw[1]), int(grid_thw[2])

        # 8. Reshape to grid
        expected_total = h_grid * w_grid
        n_vis_tokens = mean_agg.shape[0]
        
        if n_vis_tokens * 4 == expected_total:
            grid_shape = (h_grid // 2, w_grid // 2)
        elif n_vis_tokens * 2 == expected_total:
            grid_shape = (h_grid, w_grid // 2)
        elif n_vis_tokens == expected_total:
            grid_shape = (h_grid, w_grid)
        else:
            side = int(np.sqrt(n_vis_tokens))
            grid_shape = (side, side)

        try:
            attn_map = mean_agg.reshape(grid_shape)
            sink_map = sink_scores_agg.reshape(grid_shape)
        except ValueError as e:
            print(f"Warning: Could not reshape attention map: {e}")
            return None

        # 9. Normalize attention map
        attn_map_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # 10. Filter using percentile threshold
        actual_threshold = np.percentile(sink_map, SINK_PERCENTILE)
        is_sink_grid = sink_map >= actual_threshold
        
        # Also mark border tokens as sinks
        if is_sink_grid.shape[0] > 4 and is_sink_grid.shape[1] > 4:
            is_sink_grid[0, :] = True
            is_sink_grid[1, :] = True
            is_sink_grid[-1, :] = True
            is_sink_grid[-2, :] = True
            is_sink_grid[:, 0] = True
            is_sink_grid[:, 1] = True
            is_sink_grid[:, -1] = True
            is_sink_grid[:, -2] = True
        
        # Zero out sink tokens
        attn_map_filtered = attn_map_norm.copy()
        attn_map_filtered[is_sink_grid] = 0.0
        if attn_map_filtered.max() > 0:
            attn_map_filtered = attn_map_filtered / attn_map_filtered.max()
        
        # 11. Resize to image dimensions
        target_size = (image.size[0], image.size[1])
        attn_map_filtered_resized = cv2.resize(attn_map_filtered, target_size, interpolation=cv2.INTER_CUBIC)
        
        # 12. Extract bboxes using multiple methods
        # Weighted centroid
        try:
            bbox_weighted = extract_bbox_weighted_centroid(attn_map_filtered_resized, std_multiplier=2.0)
        except Exception:
            bbox_weighted = (0, 0, image.size[0], image.size[1])
        
        # Multiple morphological variants
        morphological_configs = [
            (0.3, 7),   # Original (baseline)
            (0.1, 7),   # Lower threshold (larger bbox)
            (0.3, 15),  # Larger kernel (more expansion)
            (0.1, 15),  # Both (most inclusive)
            (0.0, 7),
            (0.0, 15),
            (0.0, 31),
        ]
        
        results = {'weighted_centroid': bbox_weighted}
        
        for threshold, kernel_size in morphological_configs:
            key = f'morphological_t{threshold}_k{kernel_size}'
            try:
                bbox_morpho = extract_bbox_morphological(attn_map_filtered_resized, threshold=threshold, kernel_size=kernel_size)
                results[key] = bbox_morpho
            except Exception:
                results[key] = (0, 0, image.size[0], image.size[1])
        
        # Compute average between weighted_centroid and baseline morphological
        bbox_morpho_baseline = results['morphological_t0.3_k7']
        bbox_average = compute_average_bbox(bbox_weighted, bbox_morpho_baseline)
        results['average'] = bbox_average
        
        # Debug visualization if DEBUG env var is set
        if os.environ.get("DEBUG", "0") == "1":
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            
            # 1. Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            
            # 2. Raw attention map overlay
            axes[0, 1].imshow(image)
            attn_overlay_raw = cv2.resize(attn_map_norm, (image.size[0], image.size[1]), interpolation=cv2.INTER_CUBIC)
            axes[0, 1].imshow(attn_overlay_raw, alpha=0.6, cmap='jet')
            axes[0, 1].set_title("Raw Attention Map Overlay")
            axes[0, 1].axis('off')
            
            # 3. Filtered attention map overlay (without sinks)
            axes[1, 0].imshow(image)
            axes[1, 0].imshow(attn_map_filtered_resized, alpha=0.6, cmap='jet')
            axes[1, 0].set_title("Filtered Attention Map (No Sinks)")
            axes[1, 0].axis('off')
            
            # 4. Weighted centroid bbox on image
            axes[1, 1].imshow(image)
            x1, y1, x2, y2 = bbox_weighted
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none')
            axes[1, 1].add_patch(rect)
            axes[1, 1].set_title("Weighted Centroid BBox")
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            debug_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug_attention_bbox')
            os.makedirs(debug_dir, exist_ok=True)
            entity = entity.replace(" ", "_").replace("/", "_").replace("'", "") if entity is not None else 'None'
            debug_path = os.path.join(debug_dir, f'attn_bbox_debug_{entity}_{hash(question)}.png')
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Debug attention bbox plot saved to: {debug_path}")
        
        return results
   

class InferenceModelQwen3_VL(InferenceModel):
    def __init__(self, args):
        super().__init__(args)

        n_gpus = torch.cuda.device_count()

        print("Loading Qwen3-VL model...")
        
        self.processor = AutoProcessor.from_pretrained(
            args.model_name,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            padding_side="left",
            trust_remote_code=True
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            output_attentions=False,
            trust_remote_code=True,
        )
        
        language_model_ref = self.model.model.language_model

        if args.text_elicit:
            num_layers = len(language_model_ref.layers)
            half_layers = num_layers // 2
            
            device_map = "balanced"

            # 1. Carica il modello per Self-Elicit con Eager Attention per avere attention_mask corretta
            self.model_self_elicit = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_name,
                attn_implementation="flash_attention_2", # Necessario per ricevere attention_mask in eager_attention_forward
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                output_attentions=False,
                trust_remote_code=True
            )

            # Accesso rapido ai sottomoduli del modello elicit
            elicit_lm = self.model_self_elicit.model.language_model
            elicit_visual = self.model_self_elicit.model.visual

            self.model_self_elicit.gradient_checkpointing_disable()

            # Configurazione Attenzione Ibrida (Flash / Eager)
            # Copiamo la config dal primo layer
            config_wo_attn = copy.deepcopy(elicit_lm.layers[0].self_attn.config)
            config_wo_attn._attn_implementation = "flash_attention_2"
            config_wo_attn.output_attentions = False
            
            config_w_attn = copy.deepcopy(elicit_lm.layers[0].self_attn.config)
            config_w_attn._attn_implementation = "eager"
            config_w_attn.output_attentions = True

            for i, layer in enumerate(elicit_lm.layers):
                if i >= half_layers:
                    layer.self_attn.config = config_w_attn
                else:
                    layer.self_attn.config = config_wo_attn

            for block in elicit_visual.blocks:
                if hasattr(block.attn, 'config'):
                    block.attn.config.output_attentions = False

            # 4. Configura l'output delle attenzioni e Monkey Patching
            n_layers = len(elicit_lm.layers)
            half_layers = n_layers // 2

            for i, layer in enumerate(elicit_lm.layers):
                if i >= half_layers:
                    layer.self_attn.config.output_attentions = True
                    remove_hook_from_module(layer, "attn_output_hook")
                    
                    # Monkey Patching con le funzioni equivalenti per Qwen2.5
                    layer.forward = types.MethodType(qwen3_decoderlayer_forward, layer)
                    layer.self_attn.forward = types.MethodType(qwen3_self_attn_forward, layer.self_attn)
                    
                    layer.input_layernorm.forward = types.MethodType(qwen2_5_vl_layernorm_forward, layer.input_layernorm)
                    layer.post_attention_layernorm.forward = types.MethodType(qwen2_5_vl_layernorm_forward, layer.post_attention_layernorm)
                    
                    layer.mlp.forward = types.MethodType(qwen2_5_vl_mlp_forward, layer.mlp)
                else:
                    layer.self_attn.config.output_attentions = False
                
            for p in self.model_self_elicit.parameters():
                p.requires_grad = False

            # 5. Configurazione finale di runtime
            self.model_self_elicit.config.use_cache = False
            # Assicuriamoci di settarlo anche nel LM interno
            self.model_self_elicit.model.config.use_cache = False 
            self.model_self_elicit.eval()
        
        # Configurazione modello principale (inference veloce)
        self.model.config.use_cache = True
        self.model.model.config.use_cache = True # Qwen struttura nidificata
        self.model.eval()
        
        print("Qwen3-VL model loaded.")

        self.token_vis_start = "<|vision_start|>"
        self.token_vis_end = "<|vision_end|>"

    @torch.inference_mode()
    def extract_attention_bbox(
        self, 
        image: Image.Image, 
        question: str, 
        entity: str,
        layer_range: str = 'middle_half'
    ) -> Optional[Dict[str, Tuple[int, int, int, int]]]:
        """
        Extract bounding boxes from attention maps using entity-to-vision attention.
        
        This method performs a forward pass to extract attention from entity tokens
        to visual tokens, filters out sink tokens, and computes bboxes using multiple methods.
        
        Args:
            image: PIL Image
            question: Question text containing the entity
            entity: Entity to focus on (e.g., "building", "car")
            layer_range: Which layers to use for attention extraction
                        ('middle_half', 'last_half', 'first_half', 'all')
        
        Returns:
            Dictionary with multiple bbox methods:
            {
                'weighted_centroid': (x1, y1, x2, y2),
                'morphological_t0.3_k7': (x1, y1, x2, y2),
                ...
                'average': (x1, y1, x2, y2)
            }
            Returns None if extraction fails.
        """
        from qwen_vl_utils import process_vision_info
        # import numpy as np
        # import torch
        # import cv2
        # import types
        
        # Check if model supports this feature (needs sink dims)
        model_name = self.args.model_name
        if model_name not in SINK_DIMS:
            print(f"Warning: SINK_DIMS not defined for model {model_name}. Using default.")
            sink_dims = [318, 1874, 1819]  # Default to 3B model dims
        else:
            sink_dims = SINK_DIMS[model_name]

        SYSTEM_PROMPT = prompts.SYSTEM_PROMPT
        
        # 1. Prepare input
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": question}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.cpu().tolist()[0]
        
        # 2. Find entity tokens
        if entity is not None:
            entity_spans = self.find_text_token_spans(input_ids, entity)
            if not entity_spans:
                print(f"Warning: Entity '{entity}' not found in input. Cannot extract bbox.")
                return None
            entity_start, entity_end = entity_spans[0]
        else: # If teh entity is None we simply extract the attention weights correpsonding to the last token
            entity_start, entity_end = -1, len(input_ids)
        
        # Handle truncated entity
        if entity_end == -1:
            entity_end = len(input_ids)
        
        # 3. Vision tokens indices
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_start)
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_end)
        input_ids_tensor = inputs.input_ids[0]
        
        vision_start_positions = (input_ids_tensor == vision_start_id).nonzero(as_tuple=True)[0]
        vision_end_positions = (input_ids_tensor == vision_end_id).nonzero(as_tuple=True)[0]
        
        if len(vision_start_positions) == 0 or len(vision_end_positions) == 0:
            print("Warning: Could not find vision token boundaries.")
            return None
            
        vision_start_idx = vision_start_positions[0].item()
        vision_end_idx = vision_end_positions[0].item()
        
        # 5. Layer Selection
        n_layers = len(self.model.model.language_model.layers)
        if layer_range[0] == '[':
            try:
                start_layer, end_layer = map(int, layer_range[1:-1].split(','))
                if not (0 <= start_layer < end_layer <= n_layers):
                    raise ValueError
            except Exception:
                print(f"Invalid layer_range format: {layer_range}. Falling back to 'middle_half'.")
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
        else:
            if layer_range == 'all':
                start_layer, end_layer = 0, n_layers
            elif layer_range == 'last_half':
                start_layer, end_layer = n_layers // 2, n_layers
            elif layer_range == 'first_half':
                start_layer, end_layer = 0, n_layers // 2
            elif layer_range == 'middle_half':
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
            else:
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)

        # 6. Monkey patch for attention extraction from entity tokens
        original_forwards = {}
        for i, layer in enumerate(self.model.model.language_model.layers):
            if start_layer <= i < end_layer:
                original_forwards[i] = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(
                    create_qwen3_self_attn_forward(entity_start, entity_end), 
                    layer.self_attn
                )
        
        attention_data = {}
        sink_data = {}

        def create_attn_hook(layer_idx, vis_start, vis_end):
            def hook(module, input, output):
                if len(output) > 1 and output[1] is not None:
                    att_tensor = output[1]
                    entity_to_vision = att_tensor[0, :, :, vis_start + 1:vis_end]
                    per_head = entity_to_vision.mean(dim=1)
                    mean_heads = per_head.mean(dim=0)
                    attention_data[layer_idx] = {
                        'mean': mean_heads.detach().cpu().float().numpy()
                    }
            return hook

        def create_hidden_hook(layer_idx, vis_start, vis_end, sink_dims_list):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                vis_hidden = hidden_states[0, vis_start + 1:vis_end, :]
                
                sink_vals = vis_hidden[:, sink_dims_list]
                max_sink_val = sink_vals.abs().max(dim=1).values
                rms = torch.sqrt(vis_hidden.pow(2).mean(dim=1))
                sink_score = max_sink_val / (rms + 1e-6)
                
                sink_data[layer_idx] = {
                    'sink_scores': sink_score.detach().cpu().float().numpy()
                }
            return hook

        hooks = []
        layers = self.model.model.language_model.layers
        for i in range(start_layer, end_layer):
            h1 = layers[i].self_attn.register_forward_hook(
                create_attn_hook(i, vision_start_idx, vision_end_idx)
            )
            hooks.append(h1)
            h_sink = layers[i].register_forward_hook(
                create_hidden_hook(i, vision_start_idx, vision_end_idx, sink_dims)
            )
            hooks.append(h_sink)
        
        target_device = self.model.model.visual.patch_embed.proj.weight.device
        model_inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        try:
            self.model(**model_inputs, output_attentions=False)
        finally:
            for h in hooks:
                h.remove()
            for i, orig_forward in original_forwards.items():
                layers[i].self_attn.forward = orig_forward
            del model_inputs
            # torch.cuda.empty_cache()

        if not attention_data:
            print("Warning: No attention data extracted.")
            return None

        # 7. Aggregate attention and sink scores
        sorted_layers = sorted(attention_data.keys())
        mean_agg = np.array([attention_data[l]['mean'] for l in sorted_layers]).mean(axis=0)
        
        sorted_layers_sink = sorted(sink_data.keys())
        if sorted_layers_sink:
            sink_scores_agg = np.array([sink_data[l]['sink_scores'] for l in sorted_layers_sink]).mean(axis=0)
        else:
            sink_scores_agg = np.zeros_like(mean_agg)

        grid_thw = inputs.image_grid_thw[0]
        h_grid, w_grid = int(grid_thw[1]), int(grid_thw[2])

        # ---------------------------------------------------------
        # 8. Reshape to grid (Qwen2.5-VL / Qwen3-VL robust logic)
        # ---------------------------------------------------------
        merge_size = 2  # Spatial merge factor for Qwen vision models
        grid_h = h_grid // merge_size
        grid_w = w_grid // merge_size
        
        expected_vis_tokens = grid_h * grid_w
        n_vis_tokens = mean_agg.shape[0]
        
        if n_vis_tokens > expected_vis_tokens:
            # Filter out <|vision_newline|> tokens
            # Newlines are appended to every row of spatial tokens
            tokens_per_row = grid_w + 1
            indices = np.arange(n_vis_tokens)
            
            # Keep tokens that are NOT in the newline position
            mask = (indices + 1) % tokens_per_row != 0
            
            clean_mean_agg = mean_agg[mask][:expected_vis_tokens]
            clean_sink_scores = sink_scores_agg[mask][:expected_vis_tokens]
        else:
            clean_mean_agg = mean_agg[:expected_vis_tokens]
            clean_sink_scores = sink_scores_agg[:expected_vis_tokens]

        try:
            attn_map = clean_mean_agg.reshape((grid_h, grid_w))
            sink_map = clean_sink_scores.reshape((grid_h, grid_w))
        except ValueError as e:
            print(f"Warning: Could not reshape attention map. Expected {expected_vis_tokens} tokens, got {clean_mean_agg.shape[0]}. Error: {e}")
            return None
        # ---------------------------------------------------------

        # 9. Normalize attention map
        attn_map_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # 10. Filter using percentile threshold
        actual_threshold = np.percentile(sink_map, SINK_PERCENTILE) # Assicurati che SINK_PERCENTILE sia definito a monte
        is_sink_grid = sink_map >= actual_threshold
        
        # Also mark border tokens as sinks
        if is_sink_grid.shape[0] > 4 and is_sink_grid.shape[1] > 4:
            is_sink_grid[0, :] = True
            is_sink_grid[1, :] = True
            is_sink_grid[-1, :] = True
            is_sink_grid[-2, :] = True
            is_sink_grid[:, 0] = True
            is_sink_grid[:, 1] = True
            is_sink_grid[:, -1] = True
            is_sink_grid[:, -2] = True
        
        # Zero out sink tokens
        attn_map_filtered = attn_map_norm.copy()
        attn_map_filtered[is_sink_grid] = 0.0
        if attn_map_filtered.max() > 0:
            attn_map_filtered = attn_map_filtered / attn_map_filtered.max()
        
        # 11. Resize to image dimensions
        target_size = (image.size[0], image.size[1])
        attn_map_filtered_resized = cv2.resize(attn_map_filtered, target_size, interpolation=cv2.INTER_CUBIC)
        
        # ... qui immagino continui la tua logica per calcolare le bounding box usando i centroidi o la morfologia ...
        # return dict_with_bboxes        
        # 12. Extract bboxes using multiple methods
        # Weighted centroid
        try:
            bbox_weighted = extract_bbox_weighted_centroid(attn_map_filtered_resized, std_multiplier=2.0)
        except Exception:
            bbox_weighted = (0, 0, image.size[0], image.size[1])
        
        # Multiple morphological variants
        morphological_configs = [
            (0.3, 7),   # Original (baseline)
            (0.1, 7),   # Lower threshold (larger bbox)
            (0.3, 15),  # Larger kernel (more expansion)
            (0.1, 15),  # Both (most inclusive)
            (0.0, 7),
            (0.0, 15),
            (0.0, 31),
        ]
        
        results = {'weighted_centroid': bbox_weighted}
        
        for threshold, kernel_size in morphological_configs:
            key = f'morphological_t{threshold}_k{kernel_size}'
            try:
                bbox_morpho = extract_bbox_morphological(attn_map_filtered_resized, threshold=threshold, kernel_size=kernel_size)
                results[key] = bbox_morpho
            except Exception:
                results[key] = (0, 0, image.size[0], image.size[1])
        
        # Compute average between weighted_centroid and baseline morphological
        bbox_morpho_baseline = results['morphological_t0.3_k7']
        bbox_average = compute_average_bbox(bbox_weighted, bbox_morpho_baseline)
        results['average'] = bbox_average
        
        # Debug visualization if DEBUG env var is set
        if os.environ.get("DEBUG", "0") == "1":
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            
            # 1. Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            
            # 2. Raw attention map overlay
            axes[0, 1].imshow(image)
            attn_overlay_raw = cv2.resize(attn_map_norm, (image.size[0], image.size[1]), interpolation=cv2.INTER_CUBIC)
            axes[0, 1].imshow(attn_overlay_raw, alpha=0.6, cmap='jet')
            axes[0, 1].set_title("Raw Attention Map Overlay")
            axes[0, 1].axis('off')
            
            # 3. Filtered attention map overlay (without sinks)
            axes[1, 0].imshow(image)
            axes[1, 0].imshow(attn_map_filtered_resized, alpha=0.6, cmap='jet')
            axes[1, 0].set_title("Filtered Attention Map (No Sinks)")
            axes[1, 0].axis('off')
            
            # 4. Weighted centroid bbox on image
            axes[1, 1].imshow(image)
            x1, y1, x2, y2 = bbox_weighted
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none')
            axes[1, 1].add_patch(rect)
            axes[1, 1].set_title("Weighted Centroid BBox")
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            debug_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug_attention_bbox')
            os.makedirs(debug_dir, exist_ok=True)
            entity = entity.replace(" ", "_").replace("/", "_").replace("'", "") if entity is not None else 'None'
            debug_path = os.path.join(debug_dir, f'attn_bbox_debug_{entity}_{hash(question)}.png')
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Debug attention bbox plot saved to: {debug_path}")
        
        return results
  

class InferenceModelLLava1_5(InferenceModel):
    def __init__(self, args):
        super().__init__(args)
        
        print("LLaVA 1.5 model loaded.")

        self.model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(
            args.model_name,
            padding_side="left",
            trust_remote_code=True
        )

        self.tokenizer = self.processor.tokenizer

        language_model_ref = self.model.model.language_model

        if args.text_elicit:
            print("Configuring LLaVA 1.5 for Self-Elicit with Eager Attention...")

            num_layers = len(language_model_ref.layers)
            half_layers = num_layers // 2
            
            device_map = "balanced"

            # 1. Load separate model for Self-Elicit with Eager Attention to get correct attention_mask
            self.model_self_elicit = LlavaForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=args.model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True
            )

            # 2. Access model submodules for elicit
            elicit_lm = self.model_self_elicit.model.language_model
            elicit_visual = self.model_self_elicit.model.vision_tower

            # 3. Configure Hybrid Attention (Flash / Eager)
            # Copy config from first layer
            config_wo_attn = copy.deepcopy(elicit_lm.layers[0].self_attn.config)
            config_wo_attn._attn_implementation = "flash_attention_2"
            config_wo_attn.output_attentions = False
            config_w_attn = copy.deepcopy(elicit_lm.layers[0].self_attn.config)
            config_w_attn._attn_implementation = "eager"
            config_w_attn.output_attentions = True
            for i, layer in enumerate(elicit_lm.layers):
                if i >= half_layers:
                    layer.self_attn.config = config_w_attn
                else:
                    layer.self_attn.config = config_wo_attn
            # for block in elicit_visual.blocks:
            #     if hasattr(block.attn, 'config'):
            #         block.attn.config.output_attentions = False


            # Monkey Patching seconda metà dei layer
            n_layers = len(elicit_lm.layers)
            half_layers = n_layers // 2

            for i, layer in enumerate(elicit_lm.layers):
                if i >= half_layers:
                    layer.self_attn.config.output_attentions = True
                    remove_hook_from_module(layer, "attn_output_hook")
                    
                    # Monkey Patching con funzioni equivalenti per LLaVA 1.5
                    layer.forward = types.MethodType(llama_decoder_layer_forward, layer)
                    layer.self_attn.forward = types.MethodType(llama_self_attn_forward, layer.self_attn)
                else:
                    layer.self_attn.config.output_attentions = False
            for p in self.model_self_elicit.parameters():
                p.requires_grad = False

            # 4. Final runtime configuration
            self.model_self_elicit.config.use_cache = False
            self.model_self_elicit.model.config.use_cache = False # LLaVA nested structure
            self.model_self_elicit.eval()
        else:
            self.model.config.use_cache = True
            self.model.model.config.use_cache = True # LLaVA nested structure
            self.model.eval()

        print("LLaVA 1.5 model ready.")

    @torch.inference_mode()
    def extract_attention_bbox(
        self,
        image,
        question: str,
        entity: str,
        layer_range: str = 'middle_half',
    ):
        IMAGE_TOKEN_INDEX = self.processor.tokenizer.convert_tokens_to_ids('<image>')
        model_name = self.args.model_name
        if model_name not in SINK_DIMS:
            print(f"Warning: SINK_DIMS not defined for model {model_name}. Using default.")
            sink_dims = [318, 1874, 1819]  # Default to 3B model dims
        else:
            sink_dims = SINK_DIMS[model_name]

        SYSTEM_PROMPT = prompts.SYSTEM_PROMPT
        
        # 1. Prepare input
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": question}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        )

        inputs = inputs.to(self.model.device)
        
        input_ids_list = inputs.input_ids.cpu().tolist()[0]

        # ── 2. Compute number of vision tokens from processed image ───────────
        # images_tensor: [1, C, H, W]
        img_h = img_w = inputs.pixel_values.shape[-1]
        try:
            patch_size = self.model.model.vision_tower.vision_model.config.patch_size
        except AttributeError:
            patch_size = 14  # standard CLIP patch size
        h_grid = img_h // patch_size
        w_grid = img_w // patch_size
        n_vision_tokens = h_grid * w_grid

        # ── 3. Find where IMAGE_TOKEN_INDEX sits in raw input_ids ────────────
        try:
            img_token_pos = input_ids_list.index(IMAGE_TOKEN_INDEX)
        except ValueError:
            print("Warning: IMAGE_TOKEN_INDEX not found in input_ids.")
            return None

        # In the embedded sequence the single -200 placeholder is replaced by
        # n_vision_tokens actual tokens → positions after it shift accordingly.
        vision_start_idx = img_token_pos
        vision_end_idx   = img_token_pos + n_vision_tokens

        # ── 4. Find entity token span (adjusted for image expansion) ─────────
        if entity is not None:
            # Search in raw input_ids (before expansion)
            entity_ids = self.tokenizer.encode(entity, add_special_tokens=False)
            raw_start = None
            for i in range(len(input_ids_list) - len(entity_ids) + 1):
                if input_ids_list[i:i + len(entity_ids)] == entity_ids:
                    raw_start = i
                    break
            if raw_start is None:
                # Fallback: try find_text_token_spans
                spans = self.find_text_token_spans(
                    [t for t in input_ids_list if t != IMAGE_TOKEN_INDEX], entity
                )
                if not spans:
                    print(f"Warning: Entity '{entity}' not found in input. Cannot extract bbox.")
                    return None
                raw_start, raw_end_rel = spans[0]
                raw_end = raw_end_rel
                offset = n_vision_tokens - 1
                entity_start = raw_start + offset if raw_start > img_token_pos else raw_start
                entity_end   = raw_end   + offset if raw_end   > img_token_pos else raw_end
            else:
                raw_end = raw_start + len(entity_ids)

                # Adjust: tokens *after* the image placeholder shift by (n_vision_tokens - 1)
                offset = n_vision_tokens - 1
                entity_start = raw_start
                entity_end   = raw_end  
        else:
            # Use last token of the full embedded sequence
            total_len = len(input_ids_list) + (n_vision_tokens - 1)
            entity_start = total_len - 1
            entity_end   = total_len

        # ── 5. Layer selection ────────────────────────────────────────────────
        layers = self.model.model.language_model.layers
        n_layers = len(layers)
        if layer_range.startswith('['):
            try:
                start_layer, end_layer = map(int, layer_range[1:-1].split(','))
                if not (0 <= start_layer < end_layer <= n_layers):
                    raise ValueError
            except Exception:
                print(f"Invalid layer_range: {layer_range}. Falling back to 'middle_half'.")
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)
        else:
            if layer_range == 'all':
                start_layer, end_layer = 0, n_layers
            elif layer_range == 'last_half':
                start_layer, end_layer = n_layers // 2, n_layers
            elif layer_range == 'first_half':
                start_layer, end_layer = 0, n_layers // 2
            else:  # 'middle_half'
                start_layer, end_layer = n_layers // 4, n_layers - (n_layers // 4)

        # ── 6. Monkey-patch attention for entity→vision extraction ────────────
        original_forwards = {}
        for i in range(start_layer, end_layer):
            original_forwards[i] = layers[i].self_attn.forward
            layers[i].self_attn.forward = types.MethodType(
                create_llama_self_attn_forward(entity_start, entity_end),
                layers[i].self_attn,
            )

        attention_data = {}
        sink_data = {}

        def create_attn_hook(layer_idx, vis_start, vis_end):
            def hook(module, input, output):
                if len(output) > 1 and output[1] is not None:
                    att_tensor = output[1]  # [batch, heads, entity_len, seq_len]
                    entity_to_vision = att_tensor[0, :, :, vis_start:vis_end]
                    mean_heads = entity_to_vision.mean(dim=1).mean(dim=0)  # [n_vision]
                    attention_data[layer_idx] = {'mean': mean_heads.detach().cpu().float().numpy()}
            return hook

        def create_hidden_hook(layer_idx, vis_start, vis_end, sink_dims_list):
            def hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                vis_hidden = hs[0, vis_start:vis_end, :]
                sink_vals = vis_hidden[:, sink_dims_list]
                max_sink_val = sink_vals.abs().max(dim=1).values
                rms = torch.sqrt(vis_hidden.pow(2).mean(dim=1))
                sink_score = max_sink_val / (rms + 1e-6)
                sink_data[layer_idx] = {'sink_scores': sink_score.detach().cpu().float().numpy()}
            return hook

        hooks = []
        for i in range(start_layer, end_layer):
            hooks.append(layers[i].self_attn.register_forward_hook(
                create_attn_hook(i, vision_start_idx, vision_end_idx)
            ))
            hooks.append(layers[i].register_forward_hook(
                create_hidden_hook(i, vision_start_idx, vision_end_idx, sink_dims)
            ))

        # ── 7. Forward pass ───────────────────────────────────────────────────
        try:
            self.model(
                **inputs,
                output_attentions=False,
                use_cache=False,
            )
        finally:
            for h in hooks:
                h.remove()
            for i, orig in original_forwards.items():
                layers[i].self_attn.forward = orig
            # torch.cuda.empty_cache()

        if not attention_data:
            print("Warning: No attention data extracted.")
            return None

        # ── 8. Aggregate ──────────────────────────────────────────────────────
        sorted_layers = sorted(attention_data.keys())
        mean_agg = np.array([attention_data[l]['mean'] for l in sorted_layers]).mean(axis=0)

        sorted_layers_sink = sorted(sink_data.keys())
        if sorted_layers_sink:
            sink_scores_agg = np.array([sink_data[l]['sink_scores'] for l in sorted_layers_sink]).mean(axis=0)
        else:
            sink_scores_agg = np.zeros_like(mean_agg)

        # ── 9. Reshape to grid ────────────────────────────────────────────────
        # For LLaVA there is no merging: n_vision_tokens == h_grid * w_grid
        n_vis_tokens = mean_agg.shape[0]
        if n_vis_tokens == h_grid * w_grid:
            grid_shape = (h_grid, w_grid)
        else:
            side = int(np.sqrt(n_vis_tokens))
            grid_shape = (side, side)

        try:
            attn_map  = mean_agg.reshape(grid_shape)
            sink_map  = sink_scores_agg.reshape(grid_shape)
        except ValueError as e:
            print(f"Warning: Could not reshape attention map: {e}")
            return None

        # ── 10. Normalise & filter sinks ──────────────────────────────────────
        attn_map_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        actual_threshold = np.percentile(sink_map, SINK_PERCENTILE)
        is_sink_grid = sink_map >= actual_threshold

        if is_sink_grid.shape[0] > 4 and is_sink_grid.shape[1] > 4:
            is_sink_grid[0, :] = is_sink_grid[1, :] = True
            is_sink_grid[-1, :] = is_sink_grid[-2, :] = True
            is_sink_grid[:, 0] = is_sink_grid[:, 1] = True
            is_sink_grid[:, -1] = is_sink_grid[:, -2] = True

        attn_map_filtered = attn_map_norm.copy()
        attn_map_filtered[is_sink_grid] = 0.0
        if attn_map_filtered.max() > 0:
            attn_map_filtered /= attn_map_filtered.max()

        # ── 11. Resize to image dimensions ────────────────────────────────────
        target_size = (image.size[0], image.size[1])  # (W, H) for cv2
        attn_map_filtered_resized = cv2.resize(attn_map_filtered, target_size, interpolation=cv2.INTER_CUBIC)

        # ── 12. Extract bboxes ────────────────────────────────────────────────
        try:
            bbox_weighted = extract_bbox_weighted_centroid(attn_map_filtered_resized, std_multiplier=2.0)
        except Exception:
            bbox_weighted = (0, 0, image.size[0], image.size[1])

        morphological_configs = [(0.3, 7), (0.1, 7), (0.3, 15), (0.1, 15), (0.0, 7), (0.0, 15), (0.0, 31)]
        results = {'weighted_centroid': bbox_weighted}
        for threshold, kernel_size in morphological_configs:
            key = f'morphological_t{threshold}_k{kernel_size}'
            try:
                results[key] = extract_bbox_morphological(attn_map_filtered_resized, threshold=threshold, kernel_size=kernel_size)
            except Exception:
                results[key] = (0, 0, image.size[0], image.size[1])

        results['average'] = compute_average_bbox(bbox_weighted, results['morphological_t0.3_k7'])

        # ── 13. Debug visualisation ───────────────────────────────────────────
        if os.environ.get("DEBUG", "0") == "1":
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes[0, 0].imshow(image); axes[0, 0].set_title("Original Image"); axes[0, 0].axis('off')
            axes[0, 1].imshow(image)
            axes[0, 1].imshow(cv2.resize(attn_map_norm, target_size, interpolation=cv2.INTER_CUBIC), alpha=0.6, cmap='jet')
            axes[0, 1].set_title("Raw Attention Map"); axes[0, 1].axis('off')
            axes[1, 0].imshow(image); axes[1, 0].imshow(attn_map_filtered_resized, alpha=0.6, cmap='jet')
            axes[1, 0].set_title("Filtered Attention Map"); axes[1, 0].axis('off')
            axes[1, 1].imshow(image)
            x1, y1, x2, y2 = bbox_weighted
            axes[1, 1].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none'))
            axes[1, 1].set_title("Weighted Centroid BBox"); axes[1, 1].axis('off')
            plt.tight_layout()
            debug_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug_attention_bbox')
            os.makedirs(debug_dir, exist_ok=True)
            ent_str = (entity.replace(" ", "_").replace("/", "_").replace("'", "") if entity else 'None')
            plt.savefig(os.path.join(debug_dir, f'attn_bbox_debug_{ent_str}_{hash(question)}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved to", os.path.join(debug_dir, f'attn_bbox_debug_{ent_str}_{hash(question)}.png'))

        return results
        
            


def reconstruct_wiki_sections(knowledge_entry, section_index=-1, reconstruct=True):
    """Reconstruct the wiki sections from the knowledge entry class."""
    title = knowledge_entry.get('title')
    sections = []
    for it, section_title in enumerate(knowledge_entry.get('section_titles')):
        if it == int(section_index):
            if reconstruct:
                evidence_section = (
                    "# Wiki Article: "
                    + title
                    + "\n"
                    + "## Section Title: "
                    + section_title
                    + "\n"
                    + knowledge_entry.get('section_texts')[it]
                )
            else:
                evidence_section = knowledge_entry.get('section_texts')[it]
        elif (
            "external links" in section_title.lower()
            or "references" in section_title.lower()
        ):
            continue
        else:
            if reconstruct:
                sections.append(
                    "# Wiki Article: "
                    + title
                    + "\n"
                    + "## Section Title: "
                    + section_title
                    + "\n"
                    + knowledge_entry.get('section_texts')[it]
                )
            else:
                sections.append(knowledge_entry.get('section_texts')[it])

    if section_index != -1:
        return evidence_section, sections
    return sections

def use_omgm_retrieval(args, retriever, image_query, image_cropped=None):    
    top_k = retriever.I2T_faiss(image_query, top_k=args.top_k)
    if args.crop_in_retrieval and image_cropped is not None:
        top_k_cropped = retriever.I2T_faiss(image_cropped, top_k=args.top_k)
        # Combine and deduplicate results from original and cropped image
        combined_top_k = top_k + top_k_cropped
        seen_urls = set()
        unique_top_k = []
        for entry in combined_top_k:
            if entry['url'] not in seen_urls:
                unique_top_k.append(entry)
                seen_urls.add(entry['url'])
        # Order them based on the 'similarity' score (assuming higher is better)
        unique_top_k.sort(key=lambda x: x['similarity'], reverse=True)
        unique_top_k = unique_top_k[:args.top_k]
        top_k = unique_top_k
    wiki_urls = [entry['url'] for entry in top_k]

    top_k = top_k[:args.top_k]
    sections = []
    for retrieved_entry in top_k:
        sections.extend(reconstruct_wiki_sections(retrieved_entry['kb_entry'], reconstruct=args.reconstruct_omgm))
    docs_images = []

    return sections, docs_images, wiki_urls

def omgm_step2_retrieval(q2c_ranker, knowledge_base_dict, retrieval_results, data_id, question, step2_beta=0.2):
    top1_entry = knowledge_base_dict[retrieval_results[data_id]["retrieved_entities"][0]['url']]

    section_list = reconstruct_wiki_sections(top1_entry)
    
    q2s_scores, rank_idx = q2c_ranker.rank_entry_sections(question, section_list)
    q2s_scores = q2s_scores.tolist()
    
    step2_top1_entry_sec_sim = retrieval_results[data_id]['sec_sim'][0]
    assert len(step2_top1_entry_sec_sim) == len(q2s_scores)
    step3_scores = (
        step2_beta * torch.tensor(step2_top1_entry_sec_sim).to("cuda")
        + ( 1 - step2_beta ) * torch.tensor(q2s_scores).to("cuda")
    )
    _, index = torch.sort(step3_scores, descending=True)
    top1_section = section_list[index[0]]

    return top1_section

def omgm_step1_retrieval(knowledge_base_dict, data_id, omgm_step1_results, top_k, reconstruct=True):
    from OMGM.model import WikipediaKnowledgeBaseEntry
    retrieval_results = omgm_step1_results[data_id]
    wiki_urls = [entry['url'] for entry in retrieval_results['retrieved_entities'][:top_k]]
    kb_entries = []
    for wiki_url in wiki_urls:
        if wiki_url in knowledge_base_dict:
            kb_entry = knowledge_base_dict[wiki_url]
            kb_entries.append({
                'kb_entry': kb_entry
            })
        else:
            print(f"Warning: wiki_url {wiki_url} not found in knowledge base dict")

    kb_entries = kb_entries[:top_k]
    sections = []
    for entry in kb_entries:
        sections.extend(reconstruct_wiki_sections(entry['kb_entry'], reconstruct=reconstruct))

    return sections

def get_answer_root(args):
    root = args.output_root
    os.makedirs(root, exist_ok=True)
    model_name = args.model_name.replace('/', '_')
    if "Qwen" in args.model_name or (args.model_name == "Qwen/Qwen2.5-VL-7B-Instruct" or args.model_name == "Qwen/Qwen2.5-VL-3B-Instruct" or args.model_name == "Qwen/Qwen2.5-VL-32B-Instruct" or "Intern" in args.model_name):
        model_name = args.model_name.replace('/', '_')
    
    model_name += f"_{args.model_max_length}"

    answer_root = f"{args.dataset_name + ('_smallbbox' if args.small_bbox else '') + ('_few_shot' if args.few_shot_examples else '')}__{model_name}__{args.experiment_type}"

    if args.max_bbox_ratio is not None:
        answer_root += f"__maxbboxratio{str(args.max_bbox_ratio).replace('.', '')}"

    if args.experiment_type == "with_retrieval":
        if args.use_google_lens:
            answer_root += f"__google_lens"
        elif args.use_oracle:
            answer_root += f"__oracle"
        elif args.use_omgm:
            if args.crop_in_retrieval:
                answer_root += f"__omgm_eva-clip_crop_retrieval"
            elif args.omgm_step1:
                answer_root += f"__omgm_step1"
            elif args.omgm_step2:
                answer_root += f"__omgm_step2_fixed"
            elif args.omgm_retriever_crop:
                answer_root += f"__omgm_eva-clip_crop"
            else:
                answer_root += f"__omgm-eva-clip"
            if args.reconstruct_omgm:
                answer_root += f"_reconstruct"
        else:
            answer_root += f"__eva-clip"
        answer_root += f"_top{args.top_k}"

    if args.short_prompt:
        answer_root += '__short'
    if args.multiple_choice:
        answer_root += '__multiple_choice'
    if args.one_word:
        answer_root += '__one_word'

    if args.only_question:
        answer_root += '__only_question'

    if args.self_elicit:
        answer_root += f"__selfelicitcritic"
    if args.self_elicit_gen:
        answer_root += f"__selfelicitgen"
    if args.self_elicit_gen_passage:
        answer_root += f"__selfelicitgenpassage"
    if args.self_elicit_gen_sen2pas:
        answer_root += f"__selfelicitgensen2pas"
    if args.self_elicit_gen_all:
        answer_root += f"__selfelicitgenall"
    
    if args.self_elicit_gen or args.self_elicit_gen_passage:
        answer_root += str(f"_alpha{str(args.self_elicit_alpha).replace('.', '')}")
        
    if args.use_attention_bbox:
        answer_root += "__attnbbox"
        answer_root += f"_{args.attention_bbox_method}"
        answer_root += f"_{args.attention_bbox_layer_range.replace('[', 'q').replace(']', 'q').replace(' ', '')}"

    if args.self_elicit_image_markers:
        if args.self_elicit_image_markers_w_fallback:
            answer_root += "__selfelictimg_markers_w_fallback"
        else:
            answer_root += "__selfelictimg_markers"
    elif args.self_elicit_image_markers_w_fallback:
        answer_root += "__selfelictimg_markers_w_fallback"

    if args.self_elicit_image_bbox:
        answer_root += "__selfelictimg_bbox"
    if args.self_elicit_image_crop:
        answer_root += "__selfelictimg_crop"
    if args.self_elicit_image_only_crop:
        answer_root += "__selfelictimg_onlycrop"
    if args.self_elicit_image_add_crop:
        answer_root += "__selfelictimg_addcrop"
    if args.self_elicit_image_add_original:
        answer_root += "__selfelictimg_add_original"
    if args.self_elicit_image_w_bbox:
        answer_root += "__selfelictimg_w_bbox"
    if args.self_elicit_image_add_bbox:
        answer_root += "__selfelictimg_addbbox"
    if args.self_elicit_image_add_crop_markers:
        answer_root += "__selfelictimg_addcrop_markers"

    #REBUTTAL ARGS
    if args.cot:
        answer_root += "__cot"
    if args.eval_passages:
        answer_root += f"__eval_passages_{str(args.yes_prob_thr).replace('.', '')}"
    if args.gdino_bbox:
        answer_root += "__gdino_bbox"
    if args.re_rank_qwen:
        answer_root += f"__rerank_qwen{args.re_rank_top_k}"

    if args.attention_text_layer_range:
        answer_root += f"__attntext{args.attention_text_layer_range.replace('[', 'q').replace(']', 'q').replace(' ', '')}"

    if args.comp:
        answer_root += "__comp"

    if args.weighted_centroid_std_multiplier:
        answer_root += f"__weightedcentroidstd{str(args.weighted_centroid_std_multiplier).replace('.', '')}"
    if args.sink_tau:
        answer_root += f"__sinktau{str(args.sink_tau).replace('.', '')}"
    
    answer_root = os.path.join(root, answer_root)
    os.makedirs(answer_root, exist_ok=True)


    return answer_root

def sbatch_eval(args, answer_root):
    if os.environ.get("DEBUG", "0") == "1":
        print("DEBUG env var not set to 1; skipping sbatch submission.")
        return

    
    # Submit evaluation job via sbatch, pass answer_root as first argument to the submit script
    try:
        # Decide whether to submit: only submit when running as a single Slurm job
        # or when running as array task 0. If not running under Slurm, submit as well.
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        should_submit = False

        # Write a file in the directory on which we recap all information about the slurm job. IMPORTANT: Include logs file paths
        info_file_path = os.path.join(answer_root, "job_info.txt")
        with open(info_file_path, "w") as info_file:
            info_file.write(f"SLURM_JOB_ID: {slurm_job_id}\n")
            info_file.write(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}\n")
            info_file.write(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}\n")
            log_dir = f"/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/self-elicit/logs/{slurm_array_job_id if slurm_array_job_id else slurm_job_id}"
            info_file.write(f"Logs directory: {log_dir}\n")
            
            # Get log file paths from environment variables (set by sbatch --output and --error)
            stdout_file = os.environ.get("SLURM_STDOUTFILE")
            stderr_file = os.environ.get("SLURM_STDERRFILE")
            if stdout_file:
                info_file.write(f"STDOUT log file: {stdout_file}\n")
            if stderr_file:
                info_file.write(f"STDERR log file: {stderr_file}\n")
            
            if slurm_job_id:
                if slurm_array_task_id is not None:
                    info_file.write(f"Running as Slurm array task {slurm_array_task_id} of job {slurm_array_job_id}\n")
                else:
                    info_file.write(f"Running as single Slurm job {slurm_job_id}\n")
            else:
                info_file.write("Not running under Slurm\n")

        if slurm_job_id:
            # Under Slurm: submit only for single job or array task 0
            if slurm_array_task_id is None or str(slurm_array_task_id) == "0":
                should_submit = True
                print(f"Will submit evaluation job (SLURM_JOB_ID={slurm_job_id}, SLURM_ARRAY_JOB_ID={slurm_array_job_id}, SLURM_ARRAY_TASK_ID={slurm_array_task_id})")
            else:
                print(f"Skipping evaluation submission for array task {slurm_array_task_id} (only task 0 should submit).")
        else:
            # Not running under Slurm: submit by default
            should_submit = True
            print("Not running under Slurm; submitting evaluation job without dependency.")

        if should_submit:
            cmd = ["sbatch"]
            if args.dataset_name == "evqa":
                job_name = "evqa_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "evqa_eval", "submit_job.sh"))
            elif args.dataset_name == "infoseek":
                job_name = "infoseek_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "info_seek_eval", "eval.sh"))
            elif args.dataset_name == 'viquae':
                job_name = "viquae_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "viquae_eval", "eval.sh"))
            elif args.dataset_name == 'mrag':
                job_name = "mrag_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mrag_eval", "eval.sh"))
            elif args.dataset_name == 'okvqa':
                job_name = "okvqa_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "okvqa_eval", "eval.sh"))
            elif args.dataset_name == 'oven':
                job_name = "oven_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "oven_eval", "eval.sh"))
            elif args.dataset_name == 'blink':
                job_name = "blink_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "blink_eval", "eval.sh"))
            elif args.dataset_name == 'mmvp':
                job_name = "mmvp_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mmvp_eval", "eval.sh"))
            elif args.dataset_name == 'real_world_qa':
                job_name = "realworldqa_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "real_world_qa_eval", "eval.sh"))
            elif args.dataset_name == 'qbench':
                job_name = "qbench_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qbench_eval", "eval.sh"))
            elif args.dataset_name == 'vstar':
                job_name = "vstar_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "vstar_eval", "eval.sh"))
            elif args.dataset_name == 'ade':
                job_name = "ade_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ade_eval", "eval.sh"))
            elif args.dataset_name == 'omni':
                job_name = "omni_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "omni_eval", "eval.sh"))
            elif args.dataset_name == 'coco':
                job_name = "coco_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "coco_eval", "eval.sh"))
            elif args.dataset_name == 'textvqa':
                job_name = "textvqa_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "textvqa_eval", "eval.sh"))
            elif args.dataset_name == 'chartqa':
                job_name = "chartqa_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "chartqa_eval", "eval.sh"))
            elif args.dataset_name == 'ocrbench':
                job_name = "ocrbench_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ocrbench_eval", "eval.sh"))
            elif args.dataset_name == 'pope':
                job_name = "pope_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pope_eval", "eval.sh"))
            elif args.dataset_name == 'chair':
                job_name = "chair_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "chair_eval", "eval.sh"))
            elif args.dataset_name == 'amber':
                job_name = "amber_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "amber_eval", "eval.sh"))
            elif args.dataset_name == 'amber_disc':
                job_name = "amber_disc_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "amber_disc_eval", "eval.sh"))
            elif args.dataset_name == 'scienceqa':
                job_name = "scienceqa_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "scienceqa_eval", "eval.sh"))
            elif args.dataset_name == 'gqa':
                job_name = "gqa_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gqa_eval", "eval.sh"))
            elif args.dataset_name == 'mathvista':
                job_name = "mathvista_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mathvista_eval", "eval.sh"))
            elif args.dataset_name == 'ai2d':
                job_name = "ai2d_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ai2d_eval", "eval.sh"))
            elif args.dataset_name == 'mme':
                job_name = "mme_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mme_eval", "eval.sh"))
            elif args.dataset_name == 'mmebench_en':
                job_name = "mmebench_en_eval_" + os.path.basename(answer_root)
                cmd.extend(["--job-name", job_name])
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mmebench_en_eval", "eval.sh"))
            else:
                raise ValueError(f"Unknown dataset name: {args.dataset_name}")
        
            if slurm_job_id and (slurm_array_task_id is None or str(slurm_array_task_id) == "0"):
                # Use SLURM_ARRAY_JOB_ID for array jobs, SLURM_JOB_ID for single jobs
                job_id_for_dependency = slurm_array_job_id if slurm_array_job_id else slurm_job_id
                dependency = f"--dependency=afterok:{job_id_for_dependency}_*"
                cmd.append(dependency)
                print(f"Adding dependency on job id: {job_id_for_dependency}_*")
            
            # Set the account as the same this job has been submitted with, if available
            slurm_account = os.environ.get("SLURM_ACCOUNT")
            if slurm_account:
                cmd.extend(["--account", slurm_account])
                print(f"Setting sbatch account to: {slurm_account}")
            
            cmd.extend([script_path, answer_root])
            print(f"Submitting evaluation job: {' '.join(cmd)}")
            sbatch = subprocess.run(cmd, capture_output=True, text=True)
            if sbatch.returncode == 0:
                out = sbatch.stdout.strip()
                print(f"sbatch output: {out}")
                m = re.search(r"Submitted batch job (\d+)", out)
                if m:
                    print(f"Submitted job id: {m.group(1)}")
            else:
                print(f"sbatch failed (code {sbatch.returncode}). stderr: {sbatch.stderr}")
        else:
            print("Evaluation submission skipped.")
    except Exception as e:
        print(f"Error submitting sbatch job: {e}")

def main():
    global SINK_PERCENTILE
    args = get_args()
    answer_root = get_answer_root(args)
    print(f"Answer root: {answer_root}\n")
    
    
    if "Intern" in args.model_name:
        model = InferenceModelInternVL(args)
    elif "Qwen2-VL" in args.model_name:
        model = InferenceModelQwen2VL(args)
    elif "Qwen2.5-VL" in args.model_name:
        model = InferenceModelQwen2_5_VL(args)
    elif "Qwen3-VL" in args.model_name:
        model = InferenceModelQwen3_VL(args)
    elif 'llava_more' in args.model_name.lower():
        os.environ['TOKENIZER_PATH']=args.model_name
        from retrieval_module.retrieval_llava_more import InferenceModelLLavaMORE
        model = InferenceModelLLavaMORE(args)
    elif 'llava-1.5-7b-hf' in args.model_name:
        os.environ['TOKENIZER_PATH']=args.model_name
        model = InferenceModelLLava1_5(args)
    else:
        raise NotImplementedError(f"Model {args.model_name} not supported.")
    
    dataset, start_idx, end_idx = get_query_dataset(args)
    len_dataset = end_idx - start_idx
    
    if args.use_omgm and args.experiment_type == 'with_retrieval':
        from OMGM.model import ClipRetriever, WikipediaKnowledgeBase, WikipediaKnowledgeBaseEntry
        omgm_step1_results = None
        omgm_step2_results = None
        retriever = None
        q2c_ranker = None
        print("OMGM Retrieval Selected")
        if args.omgm_retriever:
            retriever = ClipRetriever(device="cuda:0", model='eva-clip')
            print("Knowledge Base Loading")
            knowledge_base_list = retriever.load_knowledge_base(args.wiki_KB)
            if args.dataset_name == 'infoseek':
                retriever.load_entity_faiss_index("/leonardo_scratch/large/userexternal/mmorini0/OMGM/infoseek.index")
            else:
                retriever.load_entity_faiss_index("/leonardo_scratch/large/userexternal/mmorini0/OMGM/encyclopedic.index")
        else:
            knowledge_base = WikipediaKnowledgeBase(args.wiki_KB)
            knowledge_base_list = knowledge_base.load_knowledge_base()
            if args.omgm_step1:
                with open(args.omgm_step1_result_file, "r") as f:
                    omgm_step1_results = ujson.load(f)
        
        print("Knowledge Base Loaded")
        knowledge_base_dict = {entry_info.url: entry_info for entry_info in knowledge_base_list}
        del knowledge_base_list

    elif not args.pre_computed_retrieval_path and args.experiment_type == 'with_retrieval' and not args.use_omgm:
        retriever = Retriever(args)
        
    if args.self_elicit_image_bbox or args.gdino_bbox:
        cropper = Cropper(args)
    if args.eval_passages:
        critic = CritiqueModel(model=model.model, processor=model.processor, args=args)

    if args.re_rank_qwen:
        from reranking_module.reranker import DocumentReranker

        reranker = DocumentReranker(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    
    if args.comp:
        with open('/leonardo/home/userexternal/mmorini0/mllms_know/bboxes_evqa.json', 'r') as f:
            comp_bboxes = json.load(f)

    if args.sink_tau:
        SINK_PERCENTILE = args.sink_tau

    num_image_tokens = ind_ids_len = out_ids_len = 0
    responses = []
    elicited_evidence = False
    for i, sample in tqdm.tqdm(enumerate(dataset), desc=f"{args.dataset_name} Retrieval", total=len_dataset):
        start = time.time()
        elapsed= text_elicit_time = image_elicit_time = find_span_time = 0
        context = None
        filtered_sections = []
        sections = []
        docs_images = []
        wiki_urls = []
        query = sample['question']
        image_query: PIL.Image = sample['image_query']

        num_evidence_sents = 0
        evidence_sents = []

        entity = sample.get('entity', None)
        image_cropped = sample.get('image_cropped', None)
        annotated_image = sample.get('annotated_image', None)
        bbox_2d = sample.get('bbox_2d', None)
        
        # Dynamic attention-based bbox extraction     
        if args.self_elicit_image_bbox:
            image_query, entity, found = cropper.detect_and_highlight(image_query, query, bbox=bbox_2d)
            if found and entity:
                entity = entity.split(' ')[1].replace(".","")

        data_id = sample['unique_id'] if args.dataset_name == "evqa" else sample['data_id']

        # Get evidence_section
        evidence_section = None
        if 'retrieval' in sample and 'evidence_section_id' in sample: # evqa data_loader
            try:
                evidence_section = sample['retrieval'][0]['section_texts'][int(sample['evidence_section_id'])]
                if args.use_omgm:
                    evidence_section, _ = reconstruct_wiki_sections(
                        knowledge_base_dict[sample['retrieval'][0]['url']],
                        section_index=sample['evidence_section_id'],
                        reconstruct=args.reconstruct_omgm
                    )
            except (KeyError, IndexError, ValueError):
                evidence_section = None

        image_elict_start_time = time.time()

        if args.use_attention_bbox:
            if args.dataset_name == 'viquae': # TODO: FIX THIS LOGIC
                doc = nlp(query)
                if len(doc.ents) > 0:
                    entity = str(doc.ents[-1])
            elif args.dataset_name == 'mrag':
                entity = extract_question_target(query.split('\n')[1])
            elif args.dataset_name in ['blink', 'mmvp', 'real_world_qa', 'qbench', 'vstar', 'ade', 'scienceqa', 'gqa', 'mme']:
                entity = extract_question_target(query.split('\n')[0])
            elif args.dataset_name == 'coco':
                entity = sample['target_class']
            elif args.dataset_name == 'ocrbench' or args.dataset_name == 'chair' or args.dataset_name == 'mathvista':
                entity = None # Per prendere ogni token della domanda
            elif args.dataset_name == 'omni':
                entity = None # Non runnare con omni, mettimao solo marker sull'intera immagine
            else: #textvqa
                entity = extract_question_target(query)
            
            # Extract bboxes using attention
            bboxes = model.extract_attention_bbox(
                image_query, 
                query, 
                entity,
                layer_range=args.attention_bbox_layer_range
            )

            # from visualize_attention_bbox_qualitatives import visualize_sample_bboxes
            # visualize_sample_bboxes(image_query, bboxes, data_id, entity, query, './debug_bboxes.png', None)
            
            if bboxes and args.attention_bbox_method in bboxes:
                bbox_2d = bboxes[args.attention_bbox_method]
                crop_area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
                # Apply max_bbox_ratio filter if specified
                if args.max_bbox_ratio is not None:
                    crop_area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
                    image_area = image_query.size[0] * image_query.size[1]
                    bbox_ratio = crop_area / image_area
                    if bbox_ratio > args.max_bbox_ratio:
                        bbox_2d = None
                        image_cropped = None
                        annotated_image = None
                
                if bbox_2d is not None and (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1]) > 0:
                    image_cropped = image_query.crop(bbox_2d)
                    annotated_image, _ = highlight_entity(image_query.copy(), bbox_2d)
            else:
                print(f"Warning: Could not extract bbox for sample {i}, method {args.attention_bbox_method} not found")
        
        if args.gdino_bbox:
            unique_id = sample['unique_id'] if 'unique_id' in sample else sample['data_id']
            bbox_2d = cropper.get_bbox_from_file(unique_id=unique_id)
            if bbox_2d is not None and (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1]) > 0:
                image_cropped = image_query.crop(bbox_2d)
                annotated_image, _ = highlight_entity(image_query.copy(), bbox_2d)

        if args.comp:
            if sample['unique_id'] in comp_bboxes:
                bbox_2d = comp_bboxes[sample['unique_id']]
                image_cropped = image_query.crop(bbox_2d)
                annotated_image, _ = highlight_entity(image_query.copy(), bbox_2d)

        image_elicit_end_time = time.time()
        image_elicit_time = image_elicit_end_time - image_elict_start_time

        evidence_in_context = False
        elicited_evidence = False
        try:
            if args.experiment_type == "no_retrieval":
                if args.only_question:
                    question = query
                else:
                    question = prompts.VQA_PROMPT.format(question=query, answer="")
            else:
                if args.use_google_lens:
                    dataset_image_id = sample['dataset_image_ids']
                    sections, wiki_urls = retriever.retrieve(image_query, dataset_image_id=dataset_image_id)
                    docs_images = []
                elif args.use_oracle:
                    if args.dataset_name == 'evqa' or args.use_omgm:
                        evidence_wiki_url = sample['wikipedia_url']
                        wiki_urls = evidence_wiki_url.split('|')
                         
                        sections = []
                        for wiki_url in wiki_urls:
                            if args.use_omgm:
                                sections.extend(reconstruct_wiki_sections(knowledge_base_dict[wiki_url], reconstruct=args.reconstruct_omgm))
                            else:
                                sections.extend(retriever.get_wiki_page_passages(wiki_url))
                        # sections = [evidence_section]
                    else:
                        sections = uniform_passages_of_sentences(sample['wikipedia_content'], n=300)
                elif args.use_omgm:
                    data_id_omgm = sample['unique_id'] if args.dataset_name == "evqa" else sample['data_id']
                    if args.omgm_retriever:
                        if args.omgm_retriever_crop:
                            sections, docs_images, wiki_urls = use_omgm_retrieval(args, retriever, image_cropped)    
                        else:
                            sections, docs_images, wiki_urls = use_omgm_retrieval(args, retriever, image_query, image_cropped=image_cropped)
                    elif args.omgm_step1:
                        sections = omgm_step1_retrieval(knowledge_base_dict, data_id_omgm, omgm_step1_results, args.top_k, args.reconstruct_omgm) 
                    elif args.omgm_step2:
                        section = omgm_step2_retrieval(q2c_ranker, knowledge_base_dict, omgm_step2_results, data_id_omgm, query, step2_beta=0.2)
                        sections = [section]
                else:
                    sections, docs_images, _ = retriever.retrieve(image_query, query=query)

                
                if args.eval_passages:
                    filtered_sections = []
                    yes_probs = []
                    # Batch evaluation
                    for j in range(0, len(sections), args.eval_passages_batch_size):
                        passages = sections[j:j+args.eval_passages_batch_size]
                        questions = [query] * len(passages)
                        images = [image_query] * len(passages)
                        results = critic.passages_relevance(passages, questions, images)
                        for k, result in enumerate(results):
                            if result['answer']: # Filtered opver the yes_prob threshold
                                filtered_sections.append(passages[k])
                                yes_probs.append(result['probs']['yes'])

                else:
                    filtered_sections = sections

                if args.re_rank_qwen:
                    filtered_sections, sorted_indices, sorted_scores = reranker.rerank_passages(
                        question=query,
                        image=image_query,
                        passages=filtered_sections
                    )
                    filtered_sections = filtered_sections[:args.re_rank_top_k]

                if evidence_section is not None and evidence_section in filtered_sections:
                    evidence_in_context = True

                context = PASSAGE_DELIMITER.join(filtered_sections)
                context = context + '.'
                if args.self_elicit_gen_all:
                    marker_impstart = '<START_IMPORTANT_TXT>'
                    marker_impend = '<END_IMPORTANT_TXT>'
                    context = marker_impstart + context + marker_impend
                
                if args.cot:
                    question = prompts.COT_PROMPT.format(question=query, context=context)
                else:
                    question = prompts.CONTEXT_VQA_PROMPT_training.format(question=query, context=context)

            if args.dataset_name == 'scienceqa':
                if sample["hint"] != "":
                    question = sample["hint"] + '\n' + question
                
                filtered_sections = [sample['hint']] if 'hint' in sample and sample['hint'] else []

            text_elicit_start_time = time.time()

            if args.text_elicit and len(filtered_sections) > 0:                        
                context_for_selfelicit = PASSAGE_DELIMITER.join(filtered_sections)
                question_for_selfelicit = prompts.CONTEXT_VQA_PROMPT_SELF_ELICIT.format(question=query, context=context_for_selfelicit)
                elicited_context, evidence_sents, evidence_spans, find_span_time = model.generate_self_elicit(question_for_selfelicit, image_query, context_for_selfelicit)
                # gc.collect()
                # torch.cuda.empty_cache()
                question = prompts.CONTEXT_VQA_PROMPT_training.format(question=query, context=elicited_context)

                num_evidence_sents = len(evidence_sents)

                if evidence_section is not None and evidence_in_context and any([sent.strip() in evidence_section for sent in evidence_sents]):
                    elicited_evidence = True

            text_elicit_end_time = time.time()
            text_elicit_time = text_elicit_end_time - text_elicit_start_time
            

            # if args.use_attention_bbox:
            #     if args.dataset_name == 'viquae': # TODO: FIX THIS LOGIC
            #         doc = nlp(query)
            #         if len(doc.ents) > 0:
            #             entity = str(doc.ents[-1])
            #     elif args.dataset_name == 'mrag':
            #         entity = extract_question_target(query.split('\n')[1])
            #     elif args.dataset_name in ['blink', 'mmvp', 'real_world_qa', 'qbench', 'vstar', 'ade', 'scienceqa', 'gqa', 'mme']:
            #         entity = extract_question_target(query.split('\n')[0])
            #     elif args.dataset_name == 'coco':
            #         entity = sample['target_class']
            #     elif args.dataset_name == 'ocrbench' or args.dataset_name == 'chair' or args.dataset_name == 'mathvista':
            #         entity = None # Per prendere ogni token della domanda
            #     elif args.dataset_name == 'omni':
            #         entity = None # Non runnare con omni, mettimao solo marker sull'intera immagine
            #     else: #textvqa
            #         entity = extract_question_target(query)
                
            #     # Extract bboxes using attention
            #     bboxes = model.extract_attention_bbox(
            #         image_query, 
            #         question if args.dataset_name != 'scienceqa' else query, 
            #         entity,
            #         layer_range=args.attention_bbox_layer_range
            #     )

            #     # from visualize_attention_bbox_qualitatives import visualize_sample_bboxes
            #     # visualize_sample_bboxes(image_query, bboxes, data_id, entity, query, './debug_bboxes.png', None)
                
            #     if bboxes and args.attention_bbox_method in bboxes:
            #         bbox_2d = bboxes[args.attention_bbox_method]
            #         crop_area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
            #         # Apply max_bbox_ratio filter if specified
            #         if args.max_bbox_ratio is not None:
            #             crop_area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
            #             image_area = image_query.size[0] * image_query.size[1]
            #             bbox_ratio = crop_area / image_area
            #             if bbox_ratio > args.max_bbox_ratio:
            #                 bbox_2d = None
            #                 image_cropped = None
            #                 annotated_image = None
                    
            #         if bbox_2d is not None and (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1]) > 0:
            #             image_cropped = image_query.crop(bbox_2d)
            #             annotated_image, _ = highlight_entity(image_query.copy(), bbox_2d)
            #     else:
            #         print(f"Warning: Could not extract bbox for sample {i}, method {args.attention_bbox_method} not found")
            
            output, ind_ids_len, out_ids_len, num_image_tokens = model.generate(question, image_query, entity=entity, bbox_2d=bbox_2d, image_cropped=image_cropped, annotated_image=annotated_image)

            if args.cot:
                # Extract what comes after 'Final answer:' as the final answer, otherwise keep the full output
                final_answer_marker = "Final Answer:"
                if final_answer_marker.lower() in output.lower():
                    output = output.split(final_answer_marker)[-1].strip()  

            end = time.time()
            elapsed = end - start

            # Debug: salva informazioni del sample se DEBUG=1
            if os.environ.get('DEBUG') == '1':
                debug_dir = os.path.join('./debug_samples', f"{args.dataset_name}_{data_id}")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Salva dati testuali in JSON
                debug_data = {
                    'data_id': data_id,
                    'question': query,
                    'context': context if context else None,
                    'elicited_context': elicited_context if args.text_elicit and len(filtered_sections) > 0 else None,
                    'elicited_sents': evidence_sents,
                    'num_elicited_sents': num_evidence_sents,
                    'model_output': output,
                    'ground_truth': sample.get('answer', sample.get('answers', None)),
                    'entity': entity,
                    'bbox_2d': bbox_2d,
                }
                
                with open(os.path.join(debug_dir, 'debug_info.json'), 'w', encoding='utf-8') as f:
                    ujson.dump(debug_data, f, indent=2, ensure_ascii=False)
                
                # Raccogli tutte le immagini disponibili per il PDF
                images_for_pdf = []
                if image_query is not None:
                    img_rgb = image_query.convert('RGB')
                    images_for_pdf.append(img_rgb)
                    img_rgb.save(os.path.join(debug_dir, 'image_original.png'))
                
                if image_cropped is not None:
                    img_rgb = image_cropped.convert('RGB')
                    images_for_pdf.append(img_rgb)
                    img_rgb.save(os.path.join(debug_dir, 'image_cropped.png'))
                
                if annotated_image is not None:
                    img_rgb = annotated_image.convert('RGB')
                    images_for_pdf.append(img_rgb)
                    img_rgb.save(os.path.join(debug_dir, 'image_annotated.png'))
                
                # Salva tutte le immagini in un unico PDF
                if len(images_for_pdf) > 0:
                    images_for_pdf[0].save(
                        os.path.join(debug_dir, 'images_combined.pdf'),
                        save_all=True,
                        append_images=images_for_pdf[1:] if len(images_for_pdf) > 1 else []
                    )

        except Exception as e:
            data_id = sample['unique_id'] if args.dataset_name == "evqa" else sample['data_id']
            print(f"Error processing sample {i} with data_id: ({data_id})\n{e}")
            output = "Error"
            gc.collect()
            torch.cuda.empty_cache()
        
        if args.dataset_name == "evqa":
            reference_list = sample['answer']
            if isinstance(reference_list, str):
                reference_list = reference_list.split('|')
            responses.append({
                "data_id": sample['unique_id'],
                "unique_id": sample['unique_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": output,
                "reference": reference_list,
                "question_type": sample['question_type'],
                "evidence_in_context": evidence_in_context,
                "elicited_evidence": elicited_evidence,
                "num_passages": len(filtered_sections) if args.experiment_type == "with_retrieval" else 0,
                "num_elicited_sents": num_evidence_sents,
                "elicited_sents": evidence_sents if args.text_elicit else [],
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'text_elicit_time': text_elicit_time,
                'ind_ids_len': ind_ids_len,
                'out_ids_len': out_ids_len,
                'find_span_time': find_span_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == "infoseek":
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "prediction": output,
                "reference": sample['answer'],
                "evidence_in_context": evidence_in_context,
                "elicited_evidence": elicited_evidence,
                "num_passages": len(filtered_sections) if args.experiment_type == "with_retrieval" else 0,
                "elicited_sents": evidence_sents if args.text_elicit else [],
                "num_elicited_sents": num_evidence_sents,
                "elapsed_time": elapsed,
                'ind_ids_len': ind_ids_len,
                'out_ids_len': out_ids_len,
                'image_elicit_time': image_elicit_time,
                'text_elicit_time': text_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == "viquae":
            responses.append({
                "data_id": sample['data_id'],
                "unique_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": output,
                "prediction": output,
                "reference": sample['reference'],
                "evidence_in_context": evidence_in_context,
                "num_passages": len(filtered_sections) if args.experiment_type == "with_retrieval" else 0,
                "elapsed_time": elapsed,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'mrag':
            sample.pop('image')
            sample.pop('image_query')
            sample.pop('retrieved_images')
            sample.pop('gt_images')
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "output": output,
                "reference": sample['answer'],
                "evidence_in_context": evidence_in_context,
                "num_passages": len(filtered_sections) if args.experiment_type == "with_retrieval" else 0,
                "entity": entity,
                "elapsed_time": elapsed,
                'num_image_tokens': num_image_tokens,
                **sample
            })
        elif args.dataset_name == 'okvqa':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                "evidence_in_context": evidence_in_context,
                "num_passages": len(filtered_sections) if args.experiment_type == "with_retrieval" else 0,
                "elapsed_time": elapsed,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'oven':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                "evidence_in_context": evidence_in_context,
                "num_passages": len(filtered_sections) if args.experiment_type == "with_retrieval" else 0,
                "elapsed_time": elapsed,
                'num_image_tokens': num_image_tokens
            })
        elif args.dataset_name == 'blink':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'mmvp':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'real_world_qa':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'qbench':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                'type': sample['type'],
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'vstar':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'ade':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'omni' or args.dataset_name == 'coco':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'textvqa' or args.dataset_name == 'chartqa':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'ocrbench' or args.dataset_name == 'pope':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
                'ind_ids_len': ind_ids_len,
                'out_ids_len': out_ids_len
            })
        elif args.dataset_name == 'chair':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                # "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                'image_id': sample['image_id'],
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'amber' or args.dataset_name == 'amber_disc':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "image_path": sample['image_path'] if 'image_path' in sample else "",
                # "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'ind_ids_len': ind_ids_len,
                'out_ids_len': out_ids_len,
                'num_image_tokens': num_image_tokens,
            })

        elif args.dataset_name == 'scienceqa':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "answers": sample['answer'],
                'prediction': output,
                'choices': sample['choices'] if 'choices' in sample else [],
                'text_answer': sample['text_answer'] if 'text_answer' in sample else "",
                'category': sample['category'],
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'gqa':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "answers": sample['answer'],
                'prediction': output,
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'mathvista':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                'type': sample['type'] if 'type' in sample else "",
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'ai2d':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "answers": sample['answer'],
                'prediction': output,
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })
        elif args.dataset_name == 'mme':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })

        elif args.dataset_name == 'mmebench_en':
            responses.append({
                "data_id": sample['data_id'],
                "question": sample['question'],
                "answers": sample['answer'],
                'prediction': output,
                'category': sample['category'] if 'category' in sample else "",
                'source_id': sample['source_id'] if 'source_id' in sample else "",
                "elapsed_time": elapsed,
                'image_elicit_time': image_elicit_time,
                'num_image_tokens': num_image_tokens,
            })

    sbatch_eval(args, answer_root)

    part = int(os.environ.get('PART', '0'))
    os.makedirs(answer_root, exist_ok=True)
    answers_file = os.path.join(answer_root, f'split_{part}.json')
    print("Answer file:", answers_file)

    with open(answers_file, "w") as f:
        f.write(ujson.dumps(responses))
    


        

if __name__ == "__main__":
    main()

