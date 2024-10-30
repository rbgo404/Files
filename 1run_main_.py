import os
import soundfile as sf

from tools.i18n.i18n import I18nAuto
from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()

def synthesize(
    target_text,
    output_path
):
    # Set languages to English
    ref_language = "英文"
    target_language = "英文"

    change_gpt_weights("pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
    change_sovits_weights("pretrained_models/gsv-v2final-pretrained/s2G2333k.pth")

    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path="6_output.wav",
        prompt_text=None,
        prompt_language=i18n(ref_language),
        text=target_text,
        text_language=i18n(target_language)
    )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")
    else:
        print("Synthesis failed or returned no data.")

# Example of how to call the synthesize function directly
if __name__ == '__main__':
    target_text = "This is the target text to synthesize."
    output_path = "/home/user/app/outputs"

    # Ensure the output directory exists
    # os.makedirs(output_path, exist_ok=True)

    # Call the synthesize function
    synthesize(
        target_text=target_text,
        output_path=output_path
    )
