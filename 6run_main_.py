import os
import soundfile as sf
from time import perf_counter
from tools.i18n.i18n import I18nAuto
from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()
change_gpt_weights("pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
change_sovits_weights("pretrained_models/gsv-v2final-pretrained/s2G2333k.pth")

def synthesize(
    target_text,
    output_path,
    idx
):
    # Set languages to English
    ref_language = "英文"
    target_language = "英文"


    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path="6_output.wav",
        prompt_text=None,
        prompt_language=i18n(ref_language),
        text=target_text,
        text_language=i18n(target_language),
        inp_refs=None
    )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, f"{idx+1}_output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")
    else:
        print("Synthesis failed or returned no data.")

# Example of how to call the synthesize function directly
if __name__ == '__main__':
    target_text = "This is the target text to synthesize."
    output_path = "/home/user/app/outputs/Files/outputs/gpts"
    sentences = [
    "This is testing",
    "Coffee keeps me energized throughout work.",
    "The cat chased the mouse across the kitchen floor in excitement.",
    "After the long hike up the mountain, the group finally reached the summit, where they enjoyed breathtaking views of the valley below and rested before descending again.",
    "The children eagerly waited for the school bell to ring, signaling the start of summer vacation. As soon as it rang, they ran out, cheering loudly, with plans of camping trips, swimming lessons, and ice cream parties. Their parents smiled, knowing that their kids were ready to embrace new adventures and memories.",
    "On a quiet Sunday afternoon, Sarah decided to try her hand at baking for the first time. She pulled out an old family recipe for chocolate chip cookies, which her grandmother had passed down. As the cookies baked, the warm, sweet aroma filled the house, making her feel nostalgic. When they were done, Sarah shared them with her neighbors, who praised her efforts. That simple act of kindness turned her day around. She realized that small gestures could make a big difference, and she decided to bake more often, not just for herself but to spread joy throughout her community.",
    "It was a rainy evening when Mark found himself sitting alone in a cozy cafe, sipping on a hot cup of tea. He had come to the city for a business trip, but after finishing his meetings, he felt the need to unwind. The soft hum of conversations around him and the gentle clinking of cups created a relaxing atmosphere. As he stared out the window, watching raindrops race down the glass, a stranger sat at the table next to him and smiled warmly. They struck up a conversation about travel, books, and their favorite cities. Mark found himself opening up more than he usually did, enjoying the spontaneity of the moment. The conversation felt effortless, as if they had known each other for years. When it was time to leave, they exchanged contact information, promising to meet again in the future. Walking back to his hotel, Mark reflected on how unexpected moments could sometimes be the most meaningful. Life had a way of surprising you when you least expected it. The rain had stopped, and the city lights reflected beautifully in the puddles on the street. Mark felt grateful for the unplanned evening, realizing that sometimes, the best connections happened by chance."]

    # Using the new sentences in the TTS model.
    latency = []
    for idx,text in enumerate(sentences):
        st = perf_counter()
        # Ensure the output directory exists
        # os.makedirs(output_path, exist_ok=True)
        
        # Call the synthesize function
        synthesize(
            target_text=text,
            output_path=output_path,
            idx=idx
        )
        end = perf_counter() - st
        print(end)
        latency.append(end)
        
    print(latency)
