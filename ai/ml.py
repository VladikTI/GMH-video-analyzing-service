# pip install decord flash_attn einops
import math
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import ImageFilter
from transformers import AutoModelForImageSegmentation
from PIL import Image, ImageFilter
import numpy as np
import cv2
from skimage.measure import label, regionprops
import requests
import json
from PIL import Image, ImageFilter
import numpy as np
import cv2
from skimage.measure import label, regionprops



MAIN_PATH = ''
SECOND_PATH = ''

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

path = "OpenGVLab/InternVL2-8B"
device_map = split_model('InternVL2-8B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False,)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

heatMap_images = []


#visual LLM
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

generation_config = dict(max_new_tokens=512, do_sample=True)

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, segment_duration=180, overlap_duration=30, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    total_duration = max_frame / fps
    segment_indices = []
    
    start_time = 0
    while start_time < total_duration:
        end_time = min(start_time + segment_duration, total_duration)
        segment_indices.append((start_time, end_time))
        start_time += (segment_duration - overlap_duration)

    pixel_values_list = []
    num_patches_list = []
    transform = build_transform(input_size=input_size)

    for start, end in segment_indices:
        frame_indices = get_index((start, end), fps, max_frame, num_segments=num_segments)
        segment_pixel_values = []

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            if (frame_index == 5):
                heatMap_images.append(img)
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            segment_pixel_values.append(pixel_values)

        if segment_pixel_values:
            segment_pixel_values = torch.cat(segment_pixel_values)
            pixel_values_list.append(segment_pixel_values)
            num_patches_list.append(segment_pixel_values.shape[0])

    return pixel_values_list, num_patches_list, fps, segment_indices 


video_path = MAIN_PATH 



pixel_values_list, num_patches_list, fps, segment_indices = load_video(video_path, num_segments=8,max_num=1)

responses = []
for idx, pixel_values in enumerate(pixel_values_list):
    pixel_values = pixel_values.to(torch.bfloat16).cuda()  
    video_prefix = ''.join([
        f'Frame{frame_idx + 1} at {segment_indices[idx][0] + round((segment_indices[idx][1]-segment_indices[idx][0])-(((segment_indices[idx][1]-segment_indices[idx][0])/8)*(8-frame_idx)),2)}second: <image>\n' 
        for frame_idx in range(num_patches_list[idx])
    ])   
    question = video_prefix + '''analyze the frames and make a list of the key objects and items  that affect the most, make a list of all the key events, a list of all the symbols that occur in the frames, for example, flags of countries or brand logos, give the most accurate answer, think it over qualitatively, specify the frame to which the scene or object belongs, determine the types of scenes, for example, dialogue or action. Pay attention to whether the scenes contain content related to violence, marginal behavior, pornographic scenes, semi-legal things in the Russian Federation, such as religion and illegal, according to the type of prohibited symbols,also return the most significant frame time '''
    response = model.chat(tokenizer, pixel_values, question, generation_config,
                                   num_patches_list=[num_patches_list[idx]], history=None)
    print(f'User: {question}\nAssistant: {response}')
    responses.append(f'User: {question}\nAssistant: {response}')




audio = AudioSegment.from_file(MAIN_PATH)  

audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)


# Сохранение файла
audio.export(SECOND_PATH, format="wav")



#транскрибация
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,attn_implementation="eager"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe(SECOND_PATH,return_timestamps=True)


transcribation = []
count = -1
for segment in result['chunks']:
    start_time = segment['timestamp'][0]
    
    if start_time == 0:
        count+=1
    end_time = segment['timestamp'][1]
    
    start_time+= 30*count
    end_time+= 30* count
    text = segment['text']
    print(f"[{start_time:.2f} - {end_time:.2f}]: {text}")
    transcribation.append(f"[{start_time:.2f} - {end_time:.2f}]: {text}")


audio = AudioSegment.from_file(MAIN_PATH)
raw_data = audio.raw_data
audio_np = np.frombuffer(raw_data, dtype=np.int16)

#Часть звуков
sounds = [
    "Диалог",
    "Музыка",
    "голос",
    "Тишина",
    "Фоновый шум",
]
audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
output = audio_classifier(audio_np, candidate_labels=sounds)
print(output[:3])
sounds = []
for i in output:
    if(i['score']>=0.2):
        sounds.append(i['label'])


prompts=['''объедини несколько отрезков времени в несколько ключевых промежутков времени, суммаризиуй отрезки в несколько больших отрезков,  выдели  ключевые объекты, выдели ключевые символики, определи название ключевого периода времени, например флаг страны или название бренда это символика, верни ответ в формате json
в нем должны быть начало ключевого периода времени, конец ключевого периода времени, список ключевых объектов, список ключевых символик, вот текст для анализа'''
, '''твоя задача выделить из этого текста разметки видео  время начала фрагмента и время конца фрагмента,  для этого тебе нужно проанализировать время начала кадров и дальнейшее описание, найти время самого значимого кадра, так же тебе нужно выявить  ключевые объекты в видео, ключевые события в видео и ключевые символики, например флаг страны или логотип бренда это символика, так же определи для каждой сцены ее тип, так же проверь содержится ли в сцене 
Треш – любое насилие, большое кол-во нецензурных высказываний, маргинальное, опасное поведение
    Порнография – секс сцены, обнаженные люди, сексуализированный контент
    Gray – серая зона: всё полулегальное в РФ и любой неоднозначный контент, например, религия
    Black – любое видео, содержащее запрещенную или незаконную в РФ символику, слова, объекты или звуки
верни ответ в формата json, важно, верни ответ в формате json, вот текст:''',
        '''твоя задача упаковать всю информацию в json файл, все поля обязательны 
    key_intervals: [{start_time: '...', end_time: '...', title: '...', objects: [...], events: [...], sounds: [...], music: [...], symbols: [...], interest_point: '...'}, ...],
    tags: {category: '...', objects: [object1, object2, ...], events: [...], sounds: [...], music: [...], symbols: [...], voice_transcription: '...'},
    tonality_objects: {object1: ['emotion1', 'emotion2', ...], object2: [...], ...},
    tonality_events: {event1: ['emotion1', 'emotion2', ...], event2: [...], ...},
    tonality_sounds: {sound1: ['emotion1', 'emotion2', ...], sound2: [...], ...},
    tonality_music: {music1: ['emotion1', 'emotion2', ...], music2: [...], ...},
    tonality_symbols: {symbol1: ['emotion1', 'emotion2', ...], symbol2: [...], ...},
    key_intervals - ключевой промежуток времени, start_time - начало ключевого промежутка времени, end_time - конец ключевого промежутка времени, title - название ключевого промежутка времени, sounds - основные звуки из видео,  symbols - основные символики, tags - суммаризированая разметка видео, все основные теги
     в tonality_objects твоя задача определить тональность видео к объектам, тональность это негативное, позитивное или нейтральное отношение к чему-то, tonality_events - тональность к ключевым событиям, tonality_sounds - тональность к звукам,tonality_music - тональность к музыке, tonality_symbols - тональность к символам
     ТВОЙ ОТВЕТ ОБЯЗАТЕЛЬНО ДОЛЖНО БЫТЬ В ТАКОМ ФОРМАТЕ
    ''']



#requests to gigachat
giga_responses = [[],[]]
def req_toGigaChat(prompt_type,text):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    payload = json.dumps({
    "model": "GigaChat",
    "messages": [
    {
    "role": "user",
    "content": f"{prompts[prompt_type]} {text}"
    }
    ],
    "n": 1,
    "stream": False,
    "update_interval": 0
    })
    headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': 'Bearer eyJjdHkiOiJqd3QiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.Uq7874R4MhGbSgKIq9x-eIAegeCTbl6EFZANvnHGCSws1T61Csrpqx1JGlw1R-umX2Dy3A4RwjglfGrWRhfr0ZWLy5EfMNlse6L3dwEJBNKwAwfITHWmn4AeUO1B-xwy8ZsIjgRxpXtU6wGFvnY3WiaxOeQsa8BDhHEL-86Mh4A16TCOAloQOlVoHPCRVOeP_tUNl96rDqc07Ez-nf_thTJ7ZejIPQVfRUoKUEXCwwwTwB8WkvUvVTxF7fSlcw9EVeguwFkZ5iIzm6muAeyHq9eZu9RjKUSdhgkYWJgfXg06PdFcyYcm_rHqB2y-1sRj2HH-tc8YisDgdWYXY13NtA.yEEgHrgNgBP261qh4q0o0g.lW0igvn1eKi8AJMicnEOShNPsOcpz-JFRDtYAy30No5vqHfX0EnYtKG0eAezL7zkhGOC85fpi2k0TpmyAiZ3o5q4p3GDMgxXReKAYEoi-kIG0f0aHlSiC1y9qnhjrUoBTCpU-8sIl_QXUjHtvp0ZU7JdVxgqAjaGGPzuOR9IcnvudJg4dMIv7lfZWQbwlYLJx3F3tQkah7EIYCH-GTB3PfXgvdJoEjCLfElvfKA_xml_YTJGi-L9n2_8YffT2fzTpW3-giPuL0FNOj2kmRV-5wMdl1J23CETAf3EniLoOIlKgIsuTYRyRIt2xJ7rRGCQ62GPGWTi1l_kEAvnzoVjqCYVW6VNmQxtivfCjUabUlIj5o0JMCeJOVVOOwiEc7XnWoE9YNI7OpRoizUbsAhZ_RuRUe9znGJiWWotPLLaeiKhtokUXZvAOTnQuGE2zuBKooe9tdBIH9jclb6Qdi8ts3YTem_LxHrIjpwcSDJmwdhHzPSSTqnFYKhBqtnFl2YGZwVKwGKKphahyCLzfoUFrksqKZe0YkmamtAfD18pTSobzZHiLVOtJ69vyImoKWww7j-2x249bW6ZWPTTC4BjYEhie2ccN63ZaQrJQgnh0bGej7g81T-mP6vrv000i_bNNfFbKGAm-T6PgjiRD2QP2J7ejwEahixLDbMWzkpTJ6waFWqsiOwgERnoLMHZMSypwaIgcFwgc21FRbx0JNufLA6oF-k0ULAWsFQcbYcKv-4.cgvPZIyMV3cD-ktdynbPLsVLf8Bqv7OIDnN6y6EOo8A'
    }
    giga_response = requests.request("POST", url, headers=headers, data=payload,verify=False)
    giga_responses[prompt_type].append(giga_response.text)
    print(giga_response.text)


for i in responses:
    req_toGigaChat(1,i)
count = 0
text_to_chad = ''
for i in transcribation:
    text_to_chad += i
    count+=1
    if (count == 15):
        req_toGigaChat(0,text_to_chad)
        count = 0
        text_to_chad = ''
if (count != 0):
    req_toGigaChat(0,text_to_chad)



summar = ''
for i in giga_responses:
    for j in i:
        summar += j
for i in sounds:
    summar += i + " "


# output result

json_result = req_toGigaChat(2,summar)






#модель для heatmap - point of interest
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)

torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()




def extract_object(birefnet, image_np):
    image = Image.fromarray(image_np).convert('RGB')

    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_images = transform_image(image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)

    # Маска
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask




#point of interest

def apply_heatmap(image, mask, rect_size=5, blur_radius=25, expansion_radius=75, distance_threshold=250):
    mask_np = np.array(mask) / 255.0 
    image_np = np.array(image)

    kernel = np.ones((expansion_radius, expansion_radius), np.uint8)
    expanded_mask = cv2.dilate(mask_np.astype(np.uint8), kernel)

    contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    final_mask = np.zeros_like(mask_np)

    for contour in contours:
        cv2.drawContours(final_mask, [contour], -1, (1), thickness=cv2.FILLED)

    labeled_mask = label(final_mask)
    regions = regionprops(labeled_mask)


    for region in regions:
        center_distance = np.linalg.norm(np.array(region.centroid) - np.array(mask_np.shape) // 2)
        if center_distance > distance_threshold:
            final_mask[labeled_mask == region.label] = 0

    heatmap = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.float32)

    height, width = mask_np.shape
    y_indices, x_indices = np.indices((height, width))
    center = np.array([width // 2, height // 2])
    distances = np.linalg.norm(np.stack([x_indices, y_indices], axis=-1) - center, axis=-1)
    max_distance = np.linalg.norm(center)
    alpha = np.clip(1 - (distances / max_distance), 0, 1)


    for y in range(0, height, rect_size):
        for x in range(0, width, rect_size):
            rect_mask = final_mask[y:y+rect_size, x:x+rect_size]
            if np.any(rect_mask):
                rect_h = min(rect_size, height - y)
                rect_w = min(rect_size, width - x)

                red = (255 * alpha[y:y+rect_h, x:x+rect_w]) * rect_mask[:rect_h, :rect_w] 
                heatmap[y:y+rect_h, x:x+rect_w, 0] += red


                heatmap[y:y+rect_h, x:x+rect_w, 1] = 0
                heatmap[y:y+rect_h, x:x+rect_w, 2] = 0

    heatmap_pil = Image.fromarray(heatmap.astype('uint8')).filter(ImageFilter.GaussianBlur(blur_radius))

    blended = (image_np * 0.5 + np.array(heatmap_pil) * 0.5).astype('uint8')

    return Image.fromarray(blended)


original_image, mask_image = extract_object(birefnet, imagepath='/kaggle/input/video-moscow-hack/key_frames/keyframes_007.jpg')


heatmap_image = apply_heatmap(original_image, mask_image)
heatmap_image.save("heatmap_image.png")

heatmap_image.show()













