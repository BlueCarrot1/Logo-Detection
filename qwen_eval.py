from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
qwen2_vl_2b_model_id = "/root/autodl-tmp/model/qwen2-vl-2b"
qwen2_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen2_vl_2b_model_id, 
                                                        device_map={"": device1}, torch_dtype=torch.bfloat16, 
                                                        trust_remote_code=True).eval() 
qwen2_processor = AutoProcessor.from_pretrained(qwen2_vl_2b_model_id)


GROUNDING_PROMPT = """请你输出图片当中anker logo对应的位置坐标。你的输出要求是:
1、每个bounding box的格式是[x1,y1,x2,y2],其中(x1,y1)表示图片的左上角坐标，(x2,y2)表示图片的右下角坐标。表示的都是图片的绝对像素点位置。
2、如果图片中有多个logo，那么请以列表的形式输出，即[[x1,y1,x2,y2],[x1,y1,x2,y2], ...
3、在抽取时，不要遗漏任何信息，也不要多抽取到其他的位置信息。
"""

def Qwen2_vl_grounding(image_path, model, processor, ):
    messages = [{"role": "user","content": [{"type": "image","image": image_path,},
            {"type": "text", "text": GROUNDING_PROMPT},],}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device1)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text