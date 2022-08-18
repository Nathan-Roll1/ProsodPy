<div id="header" align="left">
  <img src="https://raw.githubusercontent.com/Nathan-Roll1/ProsodPy/main/Paper/prosodpy_logo_3.png" width="550"/>
</div>

## ProsodPy - Audio Preprocessing, Modelling, and Inference (in Python)
### Companion to: 
A Deep Learning Approach to Automatic Prosodic Segmentation in Untranscribed Discourse,<br>
Roll and Graham (2022), unpublished

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
<div align="left">
  <img src="https://raw.githubusercontent.com/Nathan-Roll1/ProsodPy/main/Paper/sample_prediction.png" width="550"/>
</div>

[full tutorial](https://github.com/Nathan-Roll1/ProsodPy/blob/main/Tutorials/inference.ipynb)

Initialize sets (don't touch!)
```python
set_indices = {'pre':'[:,:512]',
               'h_pre':'[:,256:512]',
               'bound':'[:,lower_mid:upper_mid]',
               'h_bound':'[:,lower_mid+128:upper_mid-128]',
               'post':'[:,-512:][:,::-1]',
               'h_post':'[:,-256:][:,::-1]'}
```
Load the necessary resources

```python
!pip install tensorflow==2.7.0
import tensorflow as tf

# load model feed order
with open('ProsodPy/Models/Metadata/feed_order.pickle', 'rb') as handle:
  feed_order = PP.pickle.load(handle)

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

Load test file (or use your own!)
```python
import urllib.request

# if remote, specify path to audio file
audio_path = 'https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav'

# download sample audio file
urllib.request.urlretrieve(audio_path, 'gettysburg.wav')

# load the audio array with a sample rate of 8 kHz
audio_array = PP.librosa.load('gettysburg.wav',sr=8000)[0]
```
Generate MFCC and potential (heuristic) boundaries (dont change parameters!!)
``` python
boundary_mfcc, __, boundaries = PP.MFCC_preprocess(
    audio_array, PP.np.array([0]), hop_length=16, n_mfcc = 15, n_fft=743, n_frames = 1024
    )

boundary_mfcc = boundary_mfcc.transpose(0,2,1)
```
Make inferences on CNNs and RNNs
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

Create prediction dataframe
```python
# initialize DF
df = PP.pd.DataFrame()

# add each prediction set to the data frame
for c,d in outputs_dict.items():
  df[c] = [x[0] for x in d]

# index DF at boundary position
df.index = boundaries
```
Make predictions
```python
df['pred'] = OLS.predict(df[feed_order].values)
```

plot predictions on subset (change i to plot another 2-second segment)
```python
PP.waveform_plot(audio_array, boundaries, df, i=5, threshold = 0.5)
```

Export predictions
``` python
# get prediction column from dataframe
output = df['pred']

# timestamp back into seconds
output.index = output.index/8000

# save to path
PP.pd.DataFrame(output).to_csv('boundaries.csv')
```
<meta name="google-site-verification" content="H5zkmDgWfUROaKPr1fa2uXFXw9BPh_DnRNjgvAmnoq0" />
