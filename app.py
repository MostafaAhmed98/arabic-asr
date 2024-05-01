import gradio as gr
from pathlib import Path
import nemo
import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf


base_path = str(Path(__file__).parent)

# Converting the original wav to the same sr
def convert_wav_to_16k(input_wav_path, output_file_path, sr=16000):
  y, s = librosa.load(input_wav_path, sr=sr)
  sf.write(output_file_path, y, s)
  print(f'"{input_wav_path}" has been converted to {s}Hz')
  return output_file_path

def loading_nemo_and_prediction(processed_wav):
  arabic_asr = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path="pretrained_model/conformer_ctc_small_60e_adamw_30wtr_32wv_40wte.nemo") # loading the model from a path
  prediction = arabic_asr.transcribe(paths2audio_files=[processed_wav])
  return prediction

def predict(uploaded_wav):
  out_path = base_path + "/converted.wav"
  audio_conversion = convert_wav_to_16k(uploaded_wav, out_path)
  prediction_text = loading_nemo_and_prediction(audio_conversion)
  return prediction_text[0]
   


demo = gr.Interface(fn=predict, inputs=gr.Audio(value='str',label="Audio file", max_length=10, show_download_button=False, interactive=True, type="filepath"), outputs=gr.Text())
demo.launch(debug=True, share=True)