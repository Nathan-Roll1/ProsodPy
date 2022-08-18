<div id="header" align="left">
  <img src="https://raw.githubusercontent.com/Nathan-Roll1/ProsodPy/main/Paper/prosodpy_logo_3.png" width="550"/>
</div>

## ProsodPy - Audio Preprocessing, Modelling, and Inference (in Python)

### Getting Started
Start by downloading the github module:
```python
!git clone https://github.com/Nathan-Roll1/ProsodPy
```

Import ProsodPy
```python
import ProsodPy.ProsodPy as PP
```
### Predicting Intonation Unit Boundaries

[full tutorial](https://github.com/Nathan-Roll1/ProsodPy/blob/main/Tutorials/inference.ipynb)

Load the necessary resources
```python
# load feed order for OLS -------------------------------------
with open('ProsodPy/Models/Metadata/feed_order.pickle', 'rb') as handle:
  feed_order = PP.pickle.load(handle)
#--------------------------------------------------------------

# load NNs
models = PP.os.listdir('ProsodPy/Models/Model Files')

# load OLS model
with open('ProsodPy/Models/Model Files/OLS.pickle', 'rb') as handle:
  OLS = PP.pickle.load(handle)
  
# specify boundary break point
h = 512

# ... and 1/4|3/4 points
lower_mid, upper_mid = int(h-h/2),int(h+h/2)
```

point to an audio file
```python
# load test file --------------------------------------------
import urllib.request

# if remote, specify path to audio file
audio_path = 'https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav'

# download sample audio file
urllib.request.urlretrieve(audio_path, 'gettysburg.wav')

# load the audio array with a sample rate of 8 kHz
audio_array = PP.librosa.load('gettysburg.wav',sr=8000)[0]
# -----------------------------------------------------------
```
generate MFCC and potential (heuristic) boundaries (dont change parameters!!)
``` python
boundary_mfcc, __, boundaries = PP.MFCC_preprocess(
    audio_array, PP.np.array([0]), hop_length=16, n_mfcc = 15, n_fft=743, n_frames = 1024
    )

boundary_mfcc = boundary_mfcc.transpose(0,2,1)
```
make inferences on CNNs and RNNs
```python
# initialize prediction dictionary
outputs_dict = {}

# for each MFCC subset...
for k, s in tqdm(set_indices.items()):

  # get the corresponding RNN & CNN
  mods = [x for x in models if (k in x)&(not f'h_{k}' in x)]

  # for each of those models...
  for m in mods:

    # load the model
    r = tf.keras.models.load_model(f'ProsodPy/Models/Model Files/{m}')

    # generate inference and add to prediction dictionary
    if 'cnn' in m:
      t = 'cnn'
      outputs_dict[f'{k}_{t}'] = r.predict(PP.np.expand_dims(eval(f'boundary_mfcc{s}'),3))
    else:
      t = 'rnn'
      outputs_dict[f'{k}_{t}'] = r.predict(eval(f'boundary_mfcc{s}'))
```
