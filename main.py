# -*- coding: utf-8 -*-
"""
功能：
提供两种输入模式，可通过右下角按钮切换：
    1. 文字模式: 点击输入框或按 Tab 激活，输入文本，按 Enter 或 Tab 提交。
    2. 语音模式: 按 'Space' 键开始/停止录音，识别结果提交给 Agent。
语音识别结果会短暂显示在屏幕右上角。
"""

import threading
import asyncio
import queue
import json
import os
from pathlib import Path
import time


import pygame
import pygame.freetype


import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer

try:
    from digital_human_agents_v2 import DigitalHumanAgentSystem
    import pygame_Alpha as pg_base
except ImportError as e:
    print(f"[致命错误] 无法导入必要的自定义模块: {e}")
    print("请确保 'digital_human_agents_v2.py' 和 'pygame_Alpha_base.py' 文件存在于同一目录。")
    exit()

ACTION_MAP = {
    "拱手礼": "greet",
    "跳舞":   "dance",
    "常态":   "normal",
    "行礼":   "dun",
    "向前走":   "forward",
    "向后走":   "back",
    "空闲":   "idle",
}
FPS = 45

REC_FS = 16000
REC_CHANNELS = 1
REC_BLOCKSIZE = 4000
VOSK_MODEL_PATH = "model-cn/vosk-model-cn-0.22"


UI_COLORS = {
    "background": (245, 235, 220),
    "red": (180, 30, 30),
    "dark_red": (100, 10, 10),
    "light_red": (220, 120, 120),
    "gold": (217, 164, 91),
    "button_text": (100, 10, 10),
    "text": (50, 20, 10),
    "text_light": (245, 245, 240),
    "input_bg": (240, 217, 181, 200),
    "output_bg": (245, 225, 190, 220),
    "border": (180, 80, 60),
    "corner": (160, 60, 40),
    "recording_active": (0, 180, 0),
    "recording_idle": (120, 120, 120),
    "hint_text": (100, 80, 60, 180),
    "recognized_text": (255, 255, 255),
}

FONT_PATH = "font.ttf"

FALLBACK_FONT = "simhei"
INPUT_HINT = "点击或按Tab输入问题..."
RECOGNIZED_TEXT_DURATION = FPS * 4
OUTPUT_FADE_DURATION = FPS // 3

action_queue = queue.Queue()
dh_system = None

is_recording = False
audio_buffers = []
rec_stream = None
vosk_model = None
last_recognized_text = ""
recognized_text_timer = 0

def initialize_agent_system():
    """初始化 DigitalHumanAgentSystem。"""
    global dh_system
    try:
        action_paths = {
            k: f"actions/{v}.json" for k, v in ACTION_MAP.items() if v
        }
        valid_paths = {}
        for name, path in action_paths.items():
            if Path(path).exists():
                valid_paths[name] = path
            else:
                print(f"[警告] 动作文件未找到，将忽略: {path}")

        if not valid_paths:
             print("[错误] 未找到任何有效的动作文件。请检查 'actions' 目录和 ACTION_MAP")
             dh_system = None
        else:
             print(f"初始化 Agent 系统，使用动作: {list(valid_paths.keys())}")
             dh_system = DigitalHumanAgentSystem(valid_paths)

    except Exception as e:
        print(f"[错误] 初始化 DigitalHumanAgentSystem 失败: {e}")
        dh_system = None

def initialize_vosk_model():
    global vosk_model
    model_path = Path(VOSK_MODEL_PATH)
    if not model_path.exists():
        print(f"[错误] Vosk 模型路径未找到: '{VOSK_MODEL_PATH}'")
        print("请下载 Vosk 中文模型并解压到指定路径。语音识别功能将不可用。")
        vosk_model = None
        return

    try:
        print(f"加载 Vosk 模型: {VOSK_MODEL_PATH} ...")
        vosk_model = Model(str(model_path))
        print("Vosk 模型加载成功。")
    except Exception as e:
        print(f"[错误] 加载 Vosk 模型失败: {e}")
        vosk_model = None

def initialize_fonts():
    global main_font, input_font, status_font
    try:
        pygame.freetype.init()
        main_font = pygame.freetype.Font(FONT_PATH, 22)
        input_font = pygame.freetype.Font(FONT_PATH, 18)
        status_font = pygame.freetype.Font(FONT_PATH, 16)
        print(f"自定义字体 '{FONT_PATH}' 加载成功")
    except Exception as e:
        print(f"[警告] 加载自定义字体失败 ({FONT_PATH}): {e}.")
        print(f"尝试使用系统字体 '{FALLBACK_FONT}'...")
        try:
            main_font = pygame.freetype.SysFont(FALLBACK_FONT, 24)
            input_font = pygame.freetype.SysFont(FALLBACK_FONT, 20)
            status_font = pygame.freetype.SysFont(FALLBACK_FONT, 18)
            print(f"系统字体 '{FALLBACK_FONT}' 加载成功")
        except Exception as e_fallback:
            print(f"[错误] 加载系统字体 '{FALLBACK_FONT}' 也失败: {e_fallback}.")
            print("将使用 Pygame 默认字体")
            main_font = pygame.freetype.SysFont(None, 26)
            input_font = pygame.freetype.SysFont(None, 22)
            status_font = pygame.freetype.SysFont(None, 20)

def create_styled_box(width, height, bg_color, border_color, corner_color, border_radius=8, border_width=2, corner_size=12):

    # 确保宽度和高度是整数且为正
    width, height = max(1, int(width)), max(1, int(height))
    surface = pygame.Surface((width, height), pygame.SRCALPHA)

    pygame.draw.rect(surface, bg_color, (0, 0, width, height), border_radius=border_radius)

    pygame.draw.rect(surface, border_color, (0, 0, width, height), width=border_width, border_radius=border_radius)

    line_width = border_width

    corner_size = min(corner_size, width // 2, height // 2, border_radius + line_width)

    if corner_size > line_width:
        # 左上角
        pygame.draw.line(surface, corner_color, (0, corner_size), (0, 0 + line_width//2), line_width)
        pygame.draw.line(surface, corner_color, (0 + line_width//2, 0), (corner_size, 0), line_width)
        # 右上角
        pygame.draw.line(surface, corner_color, (width - corner_size, 0), (width - line_width//2, 0), line_width)
        pygame.draw.line(surface, corner_color, (width, 0 + line_width//2), (width, corner_size), line_width)
        # 左下角
        pygame.draw.line(surface, corner_color, (0, height - corner_size), (0, height - line_width//2), line_width)
        pygame.draw.line(surface, corner_color, (0 + line_width//2, height), (corner_size, height), line_width)
        # 右下角
        pygame.draw.line(surface, corner_color, (width - corner_size, height), (width - line_width//2, height), line_width)
        pygame.draw.line(surface, corner_color, (width, height - line_width//2), (width, height - corner_size), line_width)

    return surface

def create_input_box(width, height):
    return create_styled_box(width, height, UI_COLORS["input_bg"], UI_COLORS["border"], UI_COLORS["corner"], border_radius=8, corner_size=12)

def create_output_box(width, height):

    return create_styled_box(width, height, UI_COLORS["output_bg"], UI_COLORS["border"], UI_COLORS["corner"], border_radius=10, corner_size=15)



def process_and_enqueue(text: str):

    global last_recognized_text, recognized_text_timer
    if not dh_system:
        print("[警告] Agent 系统未初始化，无法处理输入。")
        action_queue.put((['idle'], "Agent系统似乎出了一些问题。"))
        return

    text = text.strip()
    if not text:
        print("[Agent] 输入为空，忽略。")
        return

    try:

        acts, reply = asyncio.run(dh_system.process_input(text))
        print(f"[Agent] 输入: '{text}' -> 回复: '{reply}' | 动作: {acts}")

        if not isinstance(acts, list):
            print(f"[警告] Agent 返回的动作不是列表: {acts}, 使用 'idle' 代替。")
            acts = ['idle']
        if not acts:
             acts = ['idle']

        action_queue.put((acts, reply))

        last_recognized_text = ""
        recognized_text_timer = 0

    except Exception as e:
        print(f"[Agent] 调用失败: {e}")
        action_queue.put((['idle'], "抱歉，我处理时遇到点麻烦。"))

def audio_callback(indata: np.ndarray, frames: int, time_info, status):

    if status:
        print(f"[录音警告] {status}")
    if is_recording:
        audio_buffers.append(indata.copy())

def recognize_and_enqueue(pcm_data: np.ndarray):

    global last_recognized_text, recognized_text_timer
    if not vosk_model:
        print("[警告] Vosk 模型未加载，无法进行语音识别。")
        last_recognized_text = "语音识别不可用"
        recognized_text_timer = RECOGNIZED_TEXT_DURATION # 显示消息
        return

    try:
        print(f"开始识别 {len(pcm_data) / REC_FS:.2f} 秒的音频...")
        recognizer = KaldiRecognizer(vosk_model, REC_FS)
        recognizer.AcceptWaveform(pcm_data.tobytes())
        result_json = recognizer.FinalResult()
        res = json.loads(result_json)
        recognized_text = res.get("text", "").strip()

        print(f"[Vosk 离线识别] 结果: '{recognized_text if recognized_text else '<无结果>'}'")

        last_recognized_text = recognized_text if recognized_text else "未识别到内容"
        recognized_text_timer = RECOGNIZED_TEXT_DURATION

        if recognized_text:

            process_and_enqueue(recognized_text)
        # else:
            # action_queue.put((['thinking'], ""))

    except json.JSONDecodeError as e:
        print(f"[识别错误] 无法解析 Vosk 返回的 JSON: {result_json} - {e}")
        last_recognized_text = "识别结果解析错误"
        recognized_text_timer = RECOGNIZED_TEXT_DURATION
    except Exception as e:
        print(f"[识别失败] 处理音频时发生错误: {e}")
        last_recognized_text = "识别过程出错"
        recognized_text_timer = RECOGNIZED_TEXT_DURATION

def toggle_recording():
    global is_recording, rec_stream, audio_buffers
    if not vosk_model:
        print("[警告] Vosk 模型未加载，无法启动录音。")
        return

    if not is_recording:
        print("[录音] 开始...")
        audio_buffers = []
        try:
            rec_stream = sd.InputStream(
                samplerate=REC_FS,
                channels=REC_CHANNELS,
                blocksize=REC_BLOCKSIZE,
                dtype="int16",
                callback=audio_callback
            )
            rec_stream.start()
            is_recording = True
            print("[录音] 录音流已启动。按 'Space' 停止。")
        except sd.PortAudioError as e:
             print(f"[录音错误] PortAudio 错误: {e}")
             print("请检查音频设备是否连接/可用，或尝试重启程序。")
             is_recording = False
             rec_stream = None
        except Exception as e:
            print(f"[录音错误] 无法启动录音流: {e}")
            is_recording = False
            rec_stream = None
    else:
        print("[录音] 停止...")
        is_recording = False
        if rec_stream:
            try:
                rec_stream.stop()
                rec_stream.close()
                print("[录音] 录音流已停止并关闭。")
            except Exception as e:
                print(f"[录音错误] 关闭录音流时出错: {e}")
            finally:
                 rec_stream = None

        if not audio_buffers:
            print("[录音] 未录制到有效音频数据。")
            last_recognized_text = "未录到声音"
            recognized_text_timer = RECOGNIZED_TEXT_DURATION // 2
            return

        try:
            full_audio_data = np.concatenate(audio_buffers, axis=0)
            audio_buffers = []

            print(f"[录音] 音频数据合并完毕 ({len(full_audio_data)} 样本), 准备识别...")

            threading.Thread(
                target=recognize_and_enqueue,
                args=(full_audio_data,),
                daemon=True
            ).start()

        except ValueError as e:
            print(f"[录音处理错误] 合并音频块失败 (可能为空或格式错误): {e}")
            audio_buffers = []
        except Exception as e:
            print(f"[录音处理错误] 处理音频数据时发生未知错误: {e}")
            audio_buffers = []


def load_frames(json_list):
    frames = []
    init_joints = {}
    first = True
    w = h = 0

    for p_str in json_list:
        p = Path(p_str)
        if not p.exists():
            print(f"[WARN] 动作文件未找到: {p_str}")
            continue

        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] 无法读取或解析动作文件 {p_str}: {e}")
            continue

        cur = data.get('frames', [])
        if not cur:

            print(f"[WARN] 动作文件 {p_str} 没有帧数据。")
            continue

        video_info = data.get('video_info', {})
        w, h = video_info.get('resolution', (w, h))

        for fr in cur:
            j = fr.get('joints', {})
            if 'right_elbow' in j:     j['right_elbow']['y'] += 15
            if 'left_elbow'  in j:     j['left_elbow']['y']  += 15
            if 'right_shoulder' in j and 'left_shoulder' in j:
                mid = (j['right_shoulder']['x'] + j['left_shoulder']['x']) / 2
                j['right_shoulder']['x'], j['left_shoulder']['x'] = mid - 30, mid
                j['left_shoulder']['x'] = mid

        if first:
            first_frame = cur[0] if cur else {}
            pelvis = first_frame.get('joints', {}).get('pelvis')

            if pelvis:
                offx, offy = w//2 - pelvis['x'], h//2 - pelvis['y']
                for fr in cur:
                     for pos in fr.get('joints', {}).values():
                        pos['x'] += offx; pos['y'] += offy
                first_frame_joints_centered = cur[0].get('joints', {})
                for name,pos in first_frame_joints_centered.items():
                    init_joints[name] = {'x':pos['x'],'y':pos['y']}
                first = False

            else:
                print(f"[WARN] 首段动作文件 {p_str} 的第一帧没有 'pelvis' 关节，无法居中和记录基线。后续拼接可能受影响。")
                first = False

        else:

            last_pelvis = frames[-1].get('joints', {}).get('pelvis') if frames else None
            start_pelvis = cur[0].get('joints', {}).get('pelvis')

            if last_pelvis and start_pelvis:
                dx, dy = last_pelvis['x']-start_pelvis['x'], last_pelvis['y']-start_pelvis['y']
                for fr in cur:
                    for pos in fr.get('joints', {}).values():
                        pos['x'] += dx; pos['y'] += dy
            elif frames:
                 print(f"[WARN] 动作文件 {p_str} 无法与上一段对齐：缺少骨盆关节信息。")

        frames.extend(cur)

    if not init_joints and frames:
         first_valid_joints = next((fr.get('joints', {}) for fr in frames if fr.get('joints')), None)
         if first_valid_joints:
             for name,pos in first_valid_joints.items():
                 init_joints[name] = {'x':pos['x'],'y':pos['y']}
             print("[WARN] 未能使用骨盆关节设置初始基线 (init_joints)，使用首个有效帧的关节作为近似基线。动画定位和缩放可能不准确。")
         else:
              print("[ERROR] 无法确定初始关节基线。请检查动作文件是否包含关节数据。")


    return frames, init_joints, w, h

def pygame_loop():
    global is_recording, last_recognized_text, recognized_text_timer, input_text, typing

    print("初始化 Pygame 及资源...")
    pygame.init()
    initialize_fonts()
    print("加载 Idle 动作...")
    try:
        idle_action_key = "空闲" if "空闲" in ACTION_MAP else ("常态" if "常态" in ACTION_MAP else None)
        if not idle_action_key or not ACTION_MAP.get(idle_action_key):
            raise ValueError("ACTION_MAP 中未定义有效的 '空闲' 或 '常态' 动作。")

        idle_json_name = ACTION_MAP[idle_action_key]
        idle_file = Path(f"actions/{idle_json_name}.json")
        if not idle_file.exists():
            raise FileNotFoundError(f"必需的 Idle/Normal 动画文件未找到: {idle_file}")

        idle_frames, initial_joints_idle_original, w, h = load_frames([str(idle_file)])

        if not idle_frames:
            raise ValueError("Idle 动画加载失败或为空。")
        print(f"Idle 动作加载成功 ({len(idle_frames)} 帧), 窗口尺寸: {w}x{h}")

    except Exception as e:
        print(f"[致命错误] 加载 Idle 动画失败: {e}")
        pygame.quit()
        return

    print("设置 Pygame 窗口和资源...")
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("皮影数字人 (文本: Tab/回车, 语音: Space)")
    try:
        bg_image_path = pg_base.BACKGROUND_IMG
        background_img = pygame.image.load(bg_image_path).convert()
        bg = pygame.transform.smoothscale(background_img, (w, h))
        print(f"背景图片 '{bg_image_path}' 加载成功。")
    except Exception as e:
        print(f"[警告] 加载背景图片失败 ({pg_base.BACKGROUND_IMG}): {e}。使用纯色背景。")
        bg = pygame.Surface((w, h))
        bg.fill(UI_COLORS["background"])

    try:
        mats = pg_base.load_materials()
        for part, img in mats.items():
             if part in pg_base.PART_PARAM:
                 mw, mh = img.get_size()
                 pg_base.PART_PARAM[part][0] = mw // 2
                 pg_base.PART_PARAM[part][1] = 0
        print(f"角色素材加载成功 ({len(mats)} 个部位)。")
    except Exception as e:
        print(f"[错误] 加载角色素材失败: {e}")
        mats = {}

    clock = pygame.time.Clock()


    input_box_height = 45
    input_box_width = min(550, w - 180)
    input_box_x = (w - input_box_width - 130) // 2
    input_box_y = h - input_box_height - 10
    input_box_img = create_input_box(input_box_width, input_box_height)
    input_box_rect = pygame.Rect(input_box_x, input_box_y, input_box_width, input_box_height)
    input_text_offset_x = 15
    input_text_offset_y = (input_box_height - input_font.get_sized_height(18)) // 2

    button_height = input_box_height - 10
    button_width = 110
    button_x = input_box_x + input_box_width + 10
    button_y = input_box_y + 5
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

    output_box_img = None
    output_box_rect = None
    output_visible = False
    output_fade_timer = 0

    # 状态和位置初始化
    state = 'idle'
    idle_idx = 0
    action_idx = 0
    action_frames = []
    action_initial_joints = {}
    text_reply = ""
    typing = False
    input_text = ""
    input_mode = "text" # 当前输入模式: "text" 或 "voice"

    initial_pelvis_original = initial_joints_idle_original.get('pelvis', {'x': 0, 'y': 0})
    center_x, center_y = w // 2, h // 2
    initial_offset_x = center_x - initial_pelvis_original['x']
    initial_offset_y = center_y - initial_pelvis_original['y']

    last_pelvis_abs = {
        'x': idle_frames[-1]['joints']['pelvis']['x'] + initial_offset_x,
        'y': idle_frames[-1]['joints']['pelvis']['y'] + initial_offset_y
    } if 'pelvis' in idle_frames[-1].get('joints', {}) else {'x': center_x, 'y': center_y}

    current_offset = {'x': initial_offset_x, 'y': initial_offset_y}

    print("初始化完成，开始主循环...")
    running = True
    while running:
        delta_time = clock.tick(FPS) / 1000.0 

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break

            if ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if input_mode == "text":
                        if input_box_rect.collidepoint(ev.pos):
                            if not typing:
                                typing = True
                                input_text = ""
                                pygame.key.start_text_input()
                                print("输入框已激活。")
                        elif typing:
                            typing = False
                            pygame.key.stop_text_input()
                            print("输入框已取消激活。")


                    if button_rect.collidepoint(ev.pos):
                        if input_mode == "text":
                            input_mode = "voice"
                            print("切换到 -> 语音模式")
                            if typing:
                                typing = False
                                pygame.key.stop_text_input()
                                input_text = ""
                        else:
                            input_mode = "text"
                            print("切换到 -> 文字模式")
                            if is_recording:
                                toggle_recording()
                        last_recognized_text = ""
                        recognized_text_timer = 0

            elif ev.type == pygame.KEYDOWN:
                if input_mode == "voice" and ev.key == pygame.K_SPACE:
                    toggle_recording()

                elif input_mode == "text" and ev.key == pygame.K_TAB:
                     typing = not typing
                     if typing:
                         input_text = ""
                         pygame.key.start_text_input()
                         print("输入框已激活 (Tab)。")
                     else:
                         pygame.key.stop_text_input()
                         print("输入框已取消激活 (Tab)。")
                         if input_text.strip():
                             print(f"提交文本 (Tab): {input_text.strip()}")
                             threading.Thread(target=process_and_enqueue, args=(input_text.strip(),), daemon=True).start()
                         input_text = ""

                elif typing and input_mode == "text":
                    if ev.key == pygame.K_RETURN or ev.key == pygame.K_KP_ENTER:
                        print(f"提交文本 (Enter): {input_text.strip()}")
                        pygame.key.stop_text_input()
                        typing = False
                        if input_text.strip():
                            threading.Thread(target=process_and_enqueue, args=(input_text.strip(),), daemon=True).start()
                        input_text = ""
                    elif ev.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif ev.key == pygame.K_ESCAPE:
                        print("取消输入 (Escape)。")
                        typing = False
                        pygame.key.stop_text_input()
                        input_text = ""

            elif ev.type == pygame.TEXTINPUT and typing and input_mode == "text":
                input_text += ev.text

        if not running: break
        if state == 'idle' and not action_queue.empty():
            acts, reply = action_queue.get()
            text_reply = reply

            json_list = []
            valid_action_found = False
            for action_name in acts:
                action_name_clean = action_name.strip().strip("'\"")
                p = Path(action_name_clean)

                potential_path = None
                if p.suffix.lower() == '.json':
                    potential_path = p
                elif p.parent != Path('.'):
                    potential_path = p.with_suffix('.json')
                else:

                    direct_path = Path(f"actions/{action_name_clean}.json")
                    if direct_path.exists():
                        potential_path = direct_path
                    else:
                        mapped_filename = ACTION_MAP.get(action_name_clean)
                        if mapped_filename:
                            mapped_path = Path(f"actions/{mapped_filename}.json")
                            if mapped_path.exists():
                                potential_path = mapped_path
                            else:
                                print(f"[警告] 动作 '{action_name_clean}' 映射的文件未找到: {mapped_path}")
                        else:
                             print(f"[警告] 动作 '{action_name_clean}' 既不是有效文件名也不是映射名。")

                if potential_path and potential_path.exists():
                     json_list.append(str(potential_path))
                     valid_action_found = True
                elif potential_path:
                     print(f"[警告] 构造的动作文件路径不存在: {potential_path}")


            if json_list and valid_action_found:
                print(f"加载新动作序列: {json_list}")
                loaded_frames, first_segment_joints_original, loaded_w, loaded_h = load_frames(json_list)

                if loaded_frames:
                    action_frames = loaded_frames
                    action_initial_joints = first_segment_joints_original

                    raw_start_pelvis = action_initial_joints.get('pelvis', {'x': loaded_w // 2, 'y': loaded_h // 2})

                    action_offset = {
                        'x': last_pelvis_abs['x'] - raw_start_pelvis['x'],
                        'y': last_pelvis_abs['y'] - raw_start_pelvis['y'],
                    }
                    current_offset = action_offset

                    last_frame_pelvis_relative = action_frames[-1]['joints'].get('pelvis', raw_start_pelvis)
                    last_pelvis_abs = {
                        'x': last_frame_pelvis_relative['x'] + action_offset['x'],
                        'y': last_frame_pelvis_relative['y'] + action_offset['y'],
                    }
                    state = 'action'
                    action_idx = 0
                    output_visible = bool(text_reply)
                    output_fade_timer = OUTPUT_FADE_DURATION if output_visible else 0
                    print(f"切换到 Action 状态, {len(action_frames)} 帧。新的 last_pelvis_abs: ({last_pelvis_abs['x']:.1f}, {last_pelvis_abs['y']:.1f})")
                else:
                    print("[警告] 动作加载失败 (返回空帧列表)，返回 Idle。")
                    state = 'idle'
                    idle_idx = 0
                    text_reply = "抱歉，我好像动不了了。"
                    output_visible = True
                    output_fade_timer = OUTPUT_FADE_DURATION

            else:
                print("[警告] 请求的动作未找到有效的动作文件，保持 Idle。")
                if not valid_action_found and acts:
                     text_reply = "我好像不认识这个动作。"
                     output_visible = True
                     output_fade_timer = OUTPUT_FADE_DURATION

        frame_joint_data_original = None
        if state == 'idle':
            if not idle_frames:
                 print("[错误] Idle 帧丢失!")
                 time.sleep(0.1)
                 continue

            frame_joint_data_original = idle_frames[idle_idx]['joints']
            idle_idx = (idle_idx + 1) % len(idle_frames)
            idle_origin_pelvis = initial_joints_idle_original.get('pelvis', {'x': w // 2, 'y': h // 2})
            idle_global_offset = {
                'x': last_pelvis_abs['x'] - idle_origin_pelvis['x'],
                'y': last_pelvis_abs['y'] - idle_origin_pelvis['y'],
            }
            current_offset = idle_global_offset

            if output_visible and not text_reply:
                if output_fade_timer > 0:
                    output_fade_timer -= 1
                else:
                    output_visible = False

        elif state == 'action':
            if not action_frames:
                 print("[错误] Action 帧丢失!")
                 state = 'idle'
                 idle_idx = 0
                 continue

            frame_joint_data_original = action_frames[action_idx]['joints']
            action_idx += 1

            if action_idx >= len(action_frames):
                print("Action 播放完毕，返回 Idle。")
                state = 'idle'
                idle_idx = 0

        joints_for_render = {}
        if frame_joint_data_original:
            joints_for_render = {
                joint_name: {
                    'x': original_pos['x'] + current_offset['x'],
                    'y': original_pos['y'] + current_offset['y'],
                }
                for joint_name, original_pos in frame_joint_data_original.items()
                if original_pos
            }
        else:
             print("[警告] 当前帧缺少关节点数据！")
             pass


        screen.blit(bg, (0, 0))

        if mats and joints_for_render:
            pg_base.render(
                screen, bg, mats,
                joints_for_render,
                initial_joints_idle_original,
                frame_idx=0
            )
        elif not mats:
             pass

        if output_visible and text_reply:
            max_text_width = w * 0.65
            padding_x = 20
            padding_y = 15
            line_spacing = 6

            wrapped_lines = []
            current_line = ""
            for char in text_reply:
                test_line = current_line + char
                rect = main_font.get_rect(test_line)
                if rect.width < max_text_width:
                    current_line = test_line
                else:
                    wrapped_lines.append(current_line)
                    current_line = char
            if current_line:
                wrapped_lines.append(current_line)

            text_rects = [main_font.get_rect(line) for line in wrapped_lines]
            total_text_height = sum(r.height for r in text_rects) + max(0, len(wrapped_lines) - 1) * line_spacing
            max_line_width = max(r.width for r in text_rects) if text_rects else 0

            output_box_width_needed = max_line_width + 2 * padding_x
            output_box_height_needed = total_text_height + 2 * padding_y

            final_output_box_width = min(max(output_box_width_needed, 150), w - 40)
            final_output_box_height = min(max(output_box_height_needed, 50), h // 2.5)
            final_output_box_x = (w - final_output_box_width) // 2
            final_output_box_y = 20
            output_box_rect = pygame.Rect(final_output_box_x, final_output_box_y, final_output_box_width, final_output_box_height)

            output_box_img = create_output_box(final_output_box_width, final_output_box_height)
            alpha = int(min(1.0, output_fade_timer / OUTPUT_FADE_DURATION) * UI_COLORS["output_bg"][3])
            output_box_img.set_alpha(alpha)

            screen.blit(output_box_img, output_box_rect.topleft)

            if alpha > 200:
                current_y = final_output_box_y + padding_y
                for i, line in enumerate(wrapped_lines):
                     main_font.render_to(screen,
                                         (final_output_box_x + padding_x, current_y),
                                         line,
                                         UI_COLORS["text"])
                     current_y += text_rects[i].height + line_spacing

        elif output_visible and output_fade_timer > 0:
            output_fade_timer -= 1
            if output_box_img and output_box_rect:
                 alpha = int(min(1.0, output_fade_timer / OUTPUT_FADE_DURATION) * UI_COLORS["output_bg"][3])
                 output_box_img.set_alpha(alpha)
                 screen.blit(output_box_img, output_box_rect.topleft)
            if output_fade_timer <= 0:
                 output_visible = False
                 output_box_img = None
                 output_box_rect = None


        if input_mode == "text":
             screen.blit(input_box_img, input_box_rect.topleft)
             text_render_x = input_box_rect.x + input_text_offset_x
             text_render_y = input_box_rect.y + input_text_offset_y

             if typing:
                 input_font.render_to(screen, (text_render_x, text_render_y), input_text, UI_COLORS["text"])
                 if int(time.time() * 2) % 2 == 0:
                     text_surface, text_rect = input_font.render(input_text)
                     cursor_x = text_render_x + text_rect.width + 1
                     cursor_y1 = text_render_y
                     cursor_y2 = text_render_y + input_font.get_sized_height(18)
                     pygame.draw.line(screen, UI_COLORS["dark_red"], (cursor_x, cursor_y1), (cursor_x, cursor_y2), 2)
             elif not input_text:
                 input_font.render_to(screen, (text_render_x, text_render_y), INPUT_HINT, UI_COLORS["hint_text"])

        button_color = UI_COLORS["gold"] if input_mode == "voice" else UI_COLORS["button_text"]
        pygame.draw.rect(screen, button_color, button_rect, border_radius=8)
        button_text = "语音模式" if input_mode == "voice" else "文字模式"
        btn_text_color = UI_COLORS["text_light"]
        text_surf, text_rect = input_font.render(button_text, btn_text_color)
        text_rect.center = button_rect.center
        screen.blit(text_surf, text_rect)

        if input_mode == "voice":
             rec_indicator_x = button_rect.left + 150
             rec_indicator_y = button_rect.centery
             rec_color = UI_COLORS["recording_active"] if is_recording else UI_COLORS["recording_idle"]
             pygame.draw.circle(screen, rec_color, (rec_indicator_x, rec_indicator_y), 7)
             rec_status_text = "录音中..." if is_recording else "按 '空格' 开始录音"
             status_font.render_to(screen,
                                  (rec_indicator_x + 12, rec_indicator_y - 10),
                                  rec_status_text,
                                  rec_color)

        if recognized_text_timer > 0:
             recognized_text_timer -= 1
             if last_recognized_text:
                 rec_text_surf, rec_text_rect = status_font.render(f"识别: {last_recognized_text}", UI_COLORS["recognized_text"])
                 rec_text_rect.topright = (w - 15, 15)
                 screen.blit(rec_text_surf, rec_text_rect)

        pygame.display.flip()


    print("退出 Pygame 主循环...")
    if rec_stream:
        try:
            if not rec_stream.closed:
                rec_stream.stop()
                rec_stream.close()
                print("录音流已停止并关闭。")
        except Exception as e:
            print(f"[警告] 关闭录音流时出错: {e}")
    try:
        sd.stop()
        print("Sounddevice 已停止。")
    except Exception as e:
        print(f"[警告] 停止 Sounddevice 时出错: {e}")

    pygame.quit()
    print("Pygame 已退出。程序结束。")




last_check_time = time.time()

if __name__ == "__main__":
    print("程序启动...")
    initialize_agent_system()
    initialize_vosk_model()

    try:
        pygame_loop()
    except Exception as main_e:
        print(f"\n--- 主程序发生未处理的异常 ---")
        print(f"错误类型: {type(main_e).__name__}")
        print(f"错误信息: {main_e}")
        import traceback
        traceback.print_exc()
        print("------------------------------")


    finally:
        if pygame.get_init():
            pygame.quit()
            print("在 finally 块中强制退出 Pygame。")

