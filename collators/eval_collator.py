from typing import Dict, Sequence
import torch

from .base import BaseDataCollator

def get_images(new_messages):
    images = []
    for msg in new_messages:
        for content in msg["content"]:
            if content["type"] == "image":
                images.append(content["image"])
    return images

class EvalDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        new_messages = []
        ids = []

        for item in messages:
            new_messages.append(item[0])
            ids.append(item[1])

        prompting = self.processor.apply_chat_template(new_messages)
        images = get_images(new_messages)

        if all(img is None for img in images):
            inputs = self.processor(
                text=prompting,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=prompting,
                images=[images],
                padding=True,
                return_tensors="pt",
            )

        input_ids = inputs['input_ids']
        
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        attention_mask = inputs.get('attention_mask', None)
        pixel_values = inputs.get('pixel_values', None)
        image_grid_thw = inputs.get('image_grid_thw', None)
        pixel_values_videos = inputs.get('pixel_values_videos', None)
        video_grid_thw = inputs.get('video_grid_thw', None)

        has_hard_negative = False

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
            has_hard_negative=has_hard_negative,
            ids=ids
        )