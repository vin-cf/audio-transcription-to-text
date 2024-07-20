import os

from dotenv import load_dotenv
from flask import Flask, request, render_template
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio

from utils import audio, files


load_dotenv()

# Set environment variable to avoid parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

models = [
    "openai/whisper-base",
    "openai/whisper-large-v3"
]


def load_audio(file_path: str) -> dict:
    """
    Load the audio file with torchaudio, ensuring only the left channel is returned
    to handle the following error from pipeline()
    ValueError: We expect a single channel audio input for AutomaticSpeechRecognitionPipeline
    @param file_path:
    @return:
    """
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform[0] if audio._is_stereo(file_path) else waveform.squeeze()
    return {'array': waveform.numpy(), 'sampling_rate': sample_rate}


def process_file(file_path: str, model_id: str, retry: bool = True) -> str:
    try:
        # Check if CUDA is available and set the device and torch_dtype accordingly
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load the model with the specified parameters
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        model.to(device)

        # Load the processor
        processor = AutoProcessor.from_pretrained(model_id)
        from transformers.pipelines import Pipeline

        # Create the pipeline
        pipe: Pipeline = pipeline(
            'automatic-speech-recognition',
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Load the audio file
        sample = load_audio(file_path)
        # Run the pipeline on the loaded audio sample
        result = pipe(sample)  # Explicitly annotate the result as a dictionary
        if isinstance(result, dict):
            transcription = result['text']
            return render_template('result.html', transcription=transcription)
        else:
            # Handle error case appropriately
            return 'Error: Invalid result from pipeline'
    except torch.cuda.OutOfMemoryError:
        if retry:
            torch.cuda.empty_cache()
            print('CUDA out of memory. Trying again.')
            return process_file(file_path, model_id, False)
        else:
            return 'Out of memory error despite clearing cuda cache'


@app.route('/')
def upload_form():
    return render_template('upload.html', models=models)


@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'model' not in request.form:
        return 'No file or model selected'
    file = request.files['file']
    model_id = request.form['model']
    if not file.filename:
        return 'No selected file'
    filename = files.generate_secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(file_path):
        print(f'File already uploaded, using {file_path}')
        transcription = process_file(file_path, model_id)
    else:
        file.save(file_path)
        transcription = process_file(file_path, model_id)

    return f'Transcription: {transcription}'


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5002, debug=True)


def _is_stereo(file_path) -> bool:
    """
    Return true if the file has more than one channel
    @param file_path:
    @return: bool
    """
    return torchaudio.info(file_path).num_channels > 1
